// Our own NR engine, in place of NVIDIA's NGX feature: a frame of the Ampere engine (nrbench) recorded
// as precompiled cubins and launches (trace.py -> export_flat.py), replayed on the game's command
// list through NVAPI's cubin shaders. No translation at run time: the kernels are sm_86 machine code,
// their parameters packed once at load against the D3D12 buffers that hold the engine's memory.
//
//   engine in:  an atlas region (the display-referred proxy, RGBA16F) -> the engine's input image,
//               float32 [3, field] at k/255, mirrored out to the field              (engine_in)
//   engine:     the recorded launches, a UAV barrier after each
//   engine out: the engine's composed frame -> the same region of a raw answer atlas (engine_out)
//   history:    once both regions ran, the change the model made (raw - input) blended with last
//               frame's, reprojected and clamped -> the answer atlas            (engine_temporal)
//
// The program lives in <exe dir>\nr-engine (program.txt, kernels\, segments\). It is loaded on a
// thread of its own (about 1 GB of buffers); until it is ready the NGX feature keeps running.
// The engine is one network at one field size, run once per region, so split foveation snaps both
// regions to the field while it is on.

#pragma once

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <d3d12.h>

#include "../build/shaders/engine_in_dxil.h"
#include "../build/shaders/engine_out_dxil.h"
#include "../build/shaders/engine_temporal_dxil.h"

namespace nr_engine {

// NVAPI entry points (nvapi_interface.h ids), resolved through dxvk-nvapi's nvapi_QueryInterface.
typedef void* (__cdecl* QueryInterface_t)(uint32_t);
typedef int (__cdecl* Initialize_t)();
typedef int (__cdecl* CreateCubinEx_t)(ID3D12Device*, const void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                       uint32_t, const char*, void**);
typedef int (__cdecl* LaunchCubin_t)(ID3D12GraphicsCommandList*, void*, uint32_t, uint32_t, uint32_t,
                                     const void*, uint32_t);
constexpr uint32_t kNvInitialize = 0x0150e828;
constexpr uint32_t kNvCreateCubinEx = 0x3151211b;   // NvAPI_D3D12_CreateCubinComputeShaderEx
constexpr uint32_t kNvLaunchCubin = 0x5c52bb86;     // NvAPI_D3D12_LaunchCubinShader

constexpr uint32_t kRingSize = 8;       // frames of descriptors in flight
constexpr uint32_t kConstants = 15;     // EngineConstants in engine_temporal.hlsl (in/out read the first 6)
constexpr uint32_t kUses = 5;           // descriptor sets per frame: in and out of two runs, the history pass
constexpr uint32_t kSlots = 6;          // t0-t3, u1-u2
constexpr uint64_t kStaging = 64ull << 20;

enum Status : int { kIdle, kLoading, kReady, kFailed };

struct Kernel {
  std::string name, file;
  uint32_t block[3] = {};
  uint32_t shared = 0;
  void* shader = nullptr;
};
struct Param {
  int kind;          // 0 pointer, 1 u32, 2 f32, 3 u64
  uint32_t seg;
  uint64_t off;
  uint64_t value;    // u32/f32 bits or u64
};
struct Launch {
  uint32_t kernel, grid[3];
  std::vector<Param> params;
  std::vector<uint8_t> packed;   // CUDA layout (each parameter aligned to its size), filled at load
};

struct Program {
  std::vector<Kernel> kernels;
  std::vector<uint64_t> seg_size;
  std::vector<std::string> seg_file;
  std::vector<bool> seg_used;
  std::vector<Launch> launches;
  uint32_t in_seg = 0, out_seg = 0;
  uint64_t in_off = 0, out_off = 0, in_bytes = 0, out_bytes = 0;
  uint32_t field_w = 0, field_h = 0;
};

struct State {
  std::atomic<int> status{kIdle};
  bool parsed = false;
  bool parse_failed = false;
  std::string dir;
  Program prog;
  std::vector<ID3D12Resource*> segs;
  uint64_t bytes = 0;            // device memory the program holds
  double load_ms = 0.0;
  char error[160] = {};

  LaunchCubin_t launch = nullptr;
  ID3D12RootSignature* root_sig = nullptr;
  ID3D12PipelineState* in_pso = nullptr;
  ID3D12PipelineState* out_pso = nullptr;
  ID3D12PipelineState* temporal_pso = nullptr;
  ID3D12DescriptorHeap* heap = nullptr;
  uint32_t descriptor_size = 0;

  // Atlas-sized textures (UNORDERED_ACCESS between passes): the engine's raw answers and the
  // change history, ping-ponged.
  ID3D12Resource* raw = nullptr;
  ID3D12Resource* hist[2] = {};
  uint32_t atlas_w = 0, atlas_h = 0;
  uint32_t flip = 0;
  bool hist_valid = false;
  struct Grave { ID3D12Resource* r; int frames; };
  std::vector<Grave> graves;
};

inline State g;

inline void Fail(const char* what, long code = 0) {
  std::snprintf(g.error, sizeof(g.error), code != 0 ? "%s (0x%lx)" : "%s", what, (unsigned long)code);
  ngx_probe::Logf("nr-engine: %s", g.error);
  g.status = kFailed;
}

inline std::vector<char> ReadAll(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// program.txt (export_flat.py): K kernels, S segments, L launches, I/O the image and frame, F the field.
inline bool Parse() {
  if (g.parsed) return true;
  if (g.parse_failed) return false;
  char exe[MAX_PATH] = {};
  GetModuleFileNameA(nullptr, exe, MAX_PATH);
  if (char* slash = std::strrchr(exe, '\\')) *slash = 0;
  g.dir = std::string(exe) + "\\nr-engine\\";

  std::ifstream f(g.dir + "program.txt");
  if (!f) {
    g.parse_failed = true;
    Fail("no program.txt in the nr-engine folder beside the exe");
    return false;
  }
  Program& p = g.prog;
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream s(line);
    std::string tag;
    s >> tag;
    if (tag == "K") {
      Kernel k;
      s >> k.name >> k.file >> k.block[0] >> k.block[1] >> k.block[2] >> k.shared;
      p.kernels.push_back(k);
    } else if (tag == "S") {
      uint64_t n = 0;
      std::string file;
      s >> n >> file;
      p.seg_size.push_back(n);
      p.seg_file.push_back(file);
    } else if (tag == "L") {
      Launch L;
      uint32_t n = 0;
      s >> L.kernel >> L.grid[0] >> L.grid[1] >> L.grid[2] >> n;
      for (uint32_t i = 0; i < n; ++i) {
        std::string t;
        s >> t;
        Param q = {};
        if (t == "ptr") {
          q.kind = 0;
          s >> q.seg >> q.off;
        } else if (t == "u32" || t == "s32" || t == "b32") {
          int64_t v = 0;
          s >> v;
          q.kind = 1;
          q.value = (uint32_t)v;
        } else if (t == "f32") {
          float v = 0.0f;
          s >> v;
          uint32_t bits;
          std::memcpy(&bits, &v, 4);
          q.kind = 2;
          q.value = bits;
        } else if (t == "u64" || t == "s64" || t == "b64") {
          int64_t v = 0;
          s >> v;
          q.kind = 3;
          q.value = (uint64_t)v;
        } else {
          g.parse_failed = true;
          Fail(("unknown parameter type " + t).c_str());
          return false;
        }
        L.params.push_back(q);
      }
      p.launches.push_back(std::move(L));
    } else if (tag == "I") {
      s >> p.in_seg >> p.in_off >> p.in_bytes;
    } else if (tag == "O") {
      s >> p.out_seg >> p.out_off >> p.out_bytes;
    } else if (tag == "F") {
      s >> p.field_w >> p.field_h;
    }
  }
  const uint64_t image = 3ull * p.field_w * p.field_h * 4;
  if (p.kernels.empty() || p.launches.empty() || p.field_w == 0 || p.in_bytes != image ||
      p.out_bytes != image || p.in_seg >= p.seg_size.size() || p.out_seg >= p.seg_size.size()) {
    g.parse_failed = true;
    Fail("program.txt is incomplete (re-export it with export_flat.py)");
    return false;
  }
  // Only the segments a launch touches are created: the trace also catches buffers the engine
  // allocated but this frame never reads.
  p.seg_used.assign(p.seg_size.size(), false);
  p.seg_used[p.in_seg] = p.seg_used[p.out_seg] = true;
  for (const Launch& L : p.launches) {
    if (L.kernel >= p.kernels.size()) {
      g.parse_failed = true;
      Fail("a launch names an unknown kernel");
      return false;
    }
    for (const Param& q : L.params)
      if (q.kind == 0) {
        if (q.seg >= p.seg_size.size()) {
          g.parse_failed = true;
          Fail("a launch names an unknown segment");
          return false;
        }
        p.seg_used[q.seg] = true;
      }
  }
  g.parsed = true;
  ngx_probe::Logf("nr-engine: program %zu kernels, %zu launches, field %ux%u", p.kernels.size(),
                  p.launches.size(), p.field_w, p.field_h);
  return true;
}

// The engine's field when its program is there to run (0 otherwise): the size foveation lays the
// regions out at.
inline uint32_t FieldW() { return Parse() ? g.prog.field_w : 0; }
inline uint32_t FieldH() { return Parse() ? g.prog.field_h : 0; }

inline D3D12_RESOURCE_DESC BufferDesc(uint64_t size, bool uav) {
  D3D12_RESOURCE_DESC d = {};
  d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  d.Width = size;
  d.Height = d.DepthOrArraySize = d.MipLevels = 1;
  d.SampleDesc.Count = 1;
  d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  d.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
  return d;
}

// The pipelines around the engine: [0] SRV table t0-t3, [1] UAV table u1-u2, [2] root UAV u0 (the
// engine's image or frame), [3] constants b0, static linear s0.
inline bool CreatePipelines(ID3D12Device* device) {
  D3D12_DESCRIPTOR_RANGE srv = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4, 0, 0, 0};
  D3D12_DESCRIPTOR_RANGE uav = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 1, 0, 0};
  D3D12_ROOT_PARAMETER params[4] = {};
  params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  params[0].DescriptorTable = {1, &srv};
  params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  params[1].DescriptorTable = {1, &uav};
  params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
  params[2].Descriptor = {0, 0};
  params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  params[3].Constants = {0, 0, kConstants};
  D3D12_STATIC_SAMPLER_DESC sampler = {};
  sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
  sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
  sampler.MaxLOD = D3D12_FLOAT32_MAX;
  sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  D3D12_ROOT_SIGNATURE_DESC desc = {4, params, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE};
  ID3DBlob* blob = nullptr;
  ID3DBlob* error = nullptr;
  if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error))) {
    if (error != nullptr) error->Release();
    return false;
  }
  const HRESULT hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                                 IID_PPV_ARGS(&g.root_sig));
  blob->Release();
  if (FAILED(hr)) return false;
  D3D12_COMPUTE_PIPELINE_STATE_DESC pso = {};
  pso.pRootSignature = g.root_sig;
  pso.CS = {engine_in_dxil, sizeof(engine_in_dxil)};
  if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&g.in_pso)))) return false;
  pso.CS = {engine_out_dxil, sizeof(engine_out_dxil)};
  if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&g.out_pso)))) return false;
  pso.CS = {engine_temporal_dxil, sizeof(engine_temporal_dxil)};
  if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&g.temporal_pso)))) return false;
  D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kRingSize * kUses * kSlots,
                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
  if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g.heap)))) return false;
  g.descriptor_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  return true;
}

// The loader thread: NVAPI, the buffers (filled through a staging buffer on a queue of its own), the
// cubin shaders, the packed launches. The device is free-threaded; nothing here touches the game's
// command lists.
inline void Load(ID3D12Device* device) {
  const ULONGLONG t0 = GetTickCount64();
  Program& p = g.prog;

  HMODULE nvapi = LoadLibraryA("nvapi64.dll");
  auto query = nvapi != nullptr ? (QueryInterface_t)GetProcAddress(nvapi, "nvapi_QueryInterface") : nullptr;
  auto init = query != nullptr ? (Initialize_t)query(kNvInitialize) : nullptr;
  auto create = query != nullptr ? (CreateCubinEx_t)query(kNvCreateCubinEx) : nullptr;
  g.launch = query != nullptr ? (LaunchCubin_t)query(kNvLaunchCubin) : nullptr;
  if (init == nullptr || create == nullptr || g.launch == nullptr) {
    device->Release();
    return Fail("NVAPI cubin entry points missing (PROTON_ENABLE_NVAPI=1, dxvk-nvapi)");
  }
  init();

  if (!CreatePipelines(device)) {
    device->Release();
    return Fail("copy pipelines");
  }

  // Segments: default-heap buffers, created zeroed; those with contents copied in chunk by chunk.
  ID3D12CommandQueue* queue = nullptr;
  ID3D12CommandAllocator* alloc = nullptr;
  ID3D12GraphicsCommandList* list = nullptr;
  ID3D12Fence* fence = nullptr;
  ID3D12Resource* staging = nullptr;
  HANDLE event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
  uint64_t fence_value = 0;
  D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_DIRECT};
  D3D12_HEAP_PROPERTIES def = {D3D12_HEAP_TYPE_DEFAULT}, up = {D3D12_HEAP_TYPE_UPLOAD};
  const D3D12_RESOURCE_DESC sd = BufferDesc(kStaging, false);
  void* staging_map = nullptr;
  bool ok = SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) &&
            SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) &&
            SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr,
                                                IID_PPV_ARGS(&list))) &&
            SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))) &&
            SUCCEEDED(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &sd,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                      IID_PPV_ARGS(&staging))) &&
            SUCCEEDED(staging->Map(0, nullptr, &staging_map));
  auto submit = [&]() {
    list->Close();
    ID3D12CommandList* ls[] = {list};
    queue->ExecuteCommandLists(1, ls);
    queue->Signal(fence, ++fence_value);
    fence->SetEventOnCompletion(fence_value, event);
    WaitForSingleObject(event, 60000);
    alloc->Reset();
    list->Reset(alloc, nullptr);
  };

  g.segs.assign(p.seg_size.size(), nullptr);
  for (size_t i = 0; ok && i < p.seg_size.size(); ++i) {
    if (!p.seg_used[i]) continue;
    const bool has_file = p.seg_file[i] != "-";
    const D3D12_RESOURCE_DESC d = BufferDesc(p.seg_size[i], true);
    if (FAILED(device->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &d,
                                               has_file ? D3D12_RESOURCE_STATE_COPY_DEST
                                                        : D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                               nullptr, IID_PPV_ARGS(&g.segs[i])))) {
      ok = false;
      std::snprintf(g.error, sizeof(g.error), "segment %zu (%llu bytes)", i, (unsigned long long)p.seg_size[i]);
      break;
    }
    g.bytes += p.seg_size[i];
    if (!has_file) continue;
    const std::vector<char> data = ReadAll(g.dir + p.seg_file[i]);
    if (data.size() != p.seg_size[i]) {
      ok = false;
      std::snprintf(g.error, sizeof(g.error), "%s: %zu bytes, expected %llu", p.seg_file[i].c_str(),
                    data.size(), (unsigned long long)p.seg_size[i]);
      break;
    }
    for (uint64_t at = 0; at < data.size(); at += kStaging) {
      const uint64_t n = data.size() - at < kStaging ? data.size() - at : kStaging;
      std::memcpy(staging_map, data.data() + at, n);
      list->CopyBufferRegion(g.segs[i], at, staging, 0, n);
      submit();
    }
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {g.segs[i], D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
    list->ResourceBarrier(1, &b);
    submit();
  }
  if (staging != nullptr) staging->Release();
  if (list != nullptr) list->Release();
  if (alloc != nullptr) alloc->Release();
  if (fence != nullptr) fence->Release();
  if (queue != nullptr) queue->Release();
  CloseHandle(event);
  if (!ok) {
    device->Release();
    return Fail(g.error[0] != 0 ? g.error : "upload objects");
  }

  for (Kernel& k : p.kernels) {
    const std::vector<char> cubin = ReadAll(g.dir + k.file);
    const int st = cubin.empty() ? -1
                                 : create(device, cubin.data(), (uint32_t)cubin.size(), k.block[0],
                                          k.block[1], k.block[2], k.shared, k.name.c_str(), &k.shader);
    if (st != 0) {
      device->Release();
      return Fail(("cubin shader " + k.name).c_str(), st);
    }
  }

  // The launches' parameters, packed once: the buffers never move.
  for (Launch& L : p.launches) {
    for (const Param& q : L.params) {
      const size_t size = q.kind == 0 || q.kind == 3 ? 8 : 4;
      while (L.packed.size() % size) L.packed.push_back(0);
      uint64_t v = q.kind == 0 ? g.segs[q.seg]->GetGPUVirtualAddress() + q.off : q.value;
      const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
      L.packed.insert(L.packed.end(), b, b + size);
    }
  }
  device->Release();
  g.load_ms = (double)(GetTickCount64() - t0);
  ngx_probe::Logf("nr-engine: ready -- %zu cubins, %zu launches, %.0f MiB in %.1f s", p.kernels.size(),
                  p.launches.size(), g.bytes / 1048576.0, g.load_ms / 1000.0);
  g.status = kReady;
}

// Start loading (once) when the program is there. True once the engine can run.
inline bool Ready(ID3D12GraphicsCommandList* cmd) {
  const int st = g.status.load();
  if (st == kReady) return true;
  if (st != kIdle || !Parse()) return false;
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return false;
  g.status = kLoading;
  ngx_probe::Log("nr-engine: loading");
  std::thread(Load, device).detach();   // Load releases the device
  return false;
}

inline const char* StatusText() {
  switch (g.status.load()) {
    case kLoading: return "loading";
    case kReady: return "ready";
    case kFailed: return g.error;
    default: return g.parse_failed ? g.error : "idle";
  }
}

inline void Barrier(ID3D12GraphicsCommandList* cmd, ID3D12Resource* r, D3D12_RESOURCE_STATES a,
                    D3D12_RESOURCE_STATES b) {
  D3D12_RESOURCE_BARRIER x = {};
  x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  x.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, a, b};
  cmd->ResourceBarrier(1, &x);
}

// Textures the GPU may still be using wait out the descriptor ring before they are released.
inline void TickGraves() {
  for (size_t i = 0; i < g.graves.size();) {
    if (--g.graves[i].frames > 0) { ++i; continue; }
    g.graves[i].r->Release();
    g.graves[i] = g.graves.back();
    g.graves.pop_back();
  }
}

inline bool EnsureTextures(ID3D12Device* device, uint32_t w, uint32_t h) {
  if (g.raw != nullptr && g.atlas_w == w && g.atlas_h == h) return true;
  for (ID3D12Resource** t : {&g.raw, &g.hist[0], &g.hist[1]}) {
    if (*t != nullptr) g.graves.push_back({*t, (int)kRingSize + 2});
    *t = nullptr;
  }
  D3D12_HEAP_PROPERTIES heap = {D3D12_HEAP_TYPE_DEFAULT};
  D3D12_RESOURCE_DESC desc = {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  desc.Width = w;
  desc.Height = h;
  desc.DepthOrArraySize = desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
  desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  for (ID3D12Resource** t : {&g.raw, &g.hist[0], &g.hist[1]})
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                               IID_PPV_ARGS(t)))) {
      Fail("history textures");
      return false;
    }
  g.atlas_w = w;
  g.atlas_h = h;
  g.hist_valid = false;
  return true;
}

// One use's descriptors (t0-t3 from `srv`, u1-u2 from `uav`; null entries read as zero), then the
// pipeline, tables, root UAV and constants.
inline void Bind(ID3D12GraphicsCommandList* cmd, ID3D12Device* device, ID3D12PipelineState* pso,
                 uint32_t frame, uint32_t use, ID3D12Resource* const (&srv)[4],
                 ID3D12Resource* const (&uav)[2], D3D12_GPU_VIRTUAL_ADDRESS root_uav,
                 const uint32_t* constants) {
  const uint32_t base = ((frame % kRingSize) * kUses + use) * kSlots;
  D3D12_CPU_DESCRIPTOR_HANDLE cpu = g.heap->GetCPUDescriptorHandleForHeapStart();
  D3D12_GPU_DESCRIPTOR_HANDLE gpu = g.heap->GetGPUDescriptorHandleForHeapStart();
  cpu.ptr += (SIZE_T)base * g.descriptor_size;
  gpu.ptr += (UINT64)base * g.descriptor_size;
  for (int i = 0; i < 4; ++i) {
    D3D12_SHADER_RESOURCE_VIEW_DESC v = {};
    v.Format = srv[i] != nullptr ? srv[i]->GetDesc().Format : DXGI_FORMAT_R16G16B16A16_FLOAT;
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.Texture2D.MipLevels = 1;
    D3D12_CPU_DESCRIPTOR_HANDLE h = cpu;
    h.ptr += (SIZE_T)i * g.descriptor_size;
    device->CreateShaderResourceView(srv[i], &v, h);
  }
  for (int i = 0; i < 2; ++i) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC v = {};
    v.Format = uav[i] != nullptr ? uav[i]->GetDesc().Format : DXGI_FORMAT_R16G16B16A16_FLOAT;
    v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    D3D12_CPU_DESCRIPTOR_HANDLE h = cpu;
    h.ptr += (SIZE_T)(4 + i) * g.descriptor_size;
    device->CreateUnorderedAccessView(uav[i], nullptr, &v, h);
  }
  D3D12_GPU_DESCRIPTOR_HANDLE gpu_uav = gpu;
  gpu_uav.ptr += 4ull * g.descriptor_size;
  cmd->SetDescriptorHeaps(1, &g.heap);
  cmd->SetComputeRootSignature(g.root_sig);
  cmd->SetPipelineState(pso);
  cmd->SetComputeRootDescriptorTable(0, gpu);
  cmd->SetComputeRootDescriptorTable(1, gpu_uav);
  cmd->SetComputeRootUnorderedAccessView(2, root_uav);
  cmd->SetComputeRoot32BitConstants(3, kConstants, constants, 0);
}

// One engine run on the region (x, y, w, h) of the atlas `src` (NON_PIXEL_SHADER_RESOURCE, left so):
// its answer lands in the same region of the raw atlas, for Temporal to finish. The region fits the
// field; the rest of the field is its mirror. `use` tells this frame's two runs apart.
inline bool Run(ID3D12GraphicsCommandList* cmd, ID3D12Resource* src, uint32_t x, uint32_t y, uint32_t w,
                uint32_t h, uint32_t frame, uint32_t use) {
  const Program& p = g.prog;
  if (g.status.load() != kReady || w > p.field_w || h > p.field_h || w == 0 || h == 0) return false;
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return false;
  const D3D12_RESOURCE_DESC atlas = src->GetDesc();
  if (use == 0) TickGraves();
  if (!EnsureTextures(device, (uint32_t)atlas.Width, atlas.Height)) {
    device->Release();
    return false;
  }

  D3D12_RESOURCE_BARRIER all = {};
  all.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;   // null resource: every UAV access before vs after
  const uint32_t c[kConstants] = {x, y, w, h, p.field_w, p.field_h};
  Bind(cmd, device, g.in_pso, frame, (use & 1) * 2, {src, nullptr, nullptr, nullptr}, {nullptr, nullptr},
       g.segs[p.in_seg]->GetGPUVirtualAddress() + p.in_off, c);
  cmd->Dispatch((p.field_w + 7) / 8, (p.field_h + 7) / 8, 1);
  cmd->ResourceBarrier(1, &all);
  for (const Launch& L : p.launches) {
    if (g.launch(cmd, p.kernels[L.kernel].shader, L.grid[0], L.grid[1], L.grid[2], L.packed.data(),
                 (uint32_t)L.packed.size()) != 0) {
      // Recorded launches cannot be taken back; what is left of the frame is garbage, so stop
      // using the engine and let NGX take over from the next frame.
      Fail(("LaunchCubinShader " + p.kernels[L.kernel].name).c_str());
      break;
    }
    cmd->ResourceBarrier(1, &all);
  }
  Bind(cmd, device, g.out_pso, frame, (use & 1) * 2 + 1, {nullptr, nullptr, nullptr, nullptr}, {g.raw, nullptr},
       g.segs[p.out_seg]->GetGPUVirtualAddress() + p.out_off, c);
  cmd->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
  cmd->ResourceBarrier(1, &all);
  device->Release();
  return true;
}

// After this frame's runs: the answer atlas (UNORDERED_ACCESS) = input + the model's change, blended
// with the reprojected history. `proxy` and `motion` are the atlas inputs (NON_PIXEL_SHADER_RESOURCE);
// `history` the weight of the history (0 = none); `reset` drops it; `low_ran` whether the low-res
// region was run this frame (if not, it is left alone).
inline void Temporal(ID3D12GraphicsCommandList* cmd, ID3D12Resource* proxy, ID3D12Resource* motion,
                     ID3D12Resource* answer, uint32_t crop_w, uint32_t crop_h, uint32_t low_x,
                     uint32_t low_w, uint32_t low_h, uint32_t frame, float history, bool reset,
                     bool low_ran) {
  ID3D12Device* device = nullptr;
  if (g.raw == nullptr || FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return;
  constexpr auto kUav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  constexpr auto kSrv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  history = history < 0.0f ? 0.0f : (history > 0.95f ? 0.95f : history);
  const float alpha = reset || !g.hist_valid ? 1.0f : 1.0f - history;
  uint32_t c[kConstants] = {0, 0, 0, 0, g.prog.field_w, g.prog.field_h, crop_w, crop_h, low_x, low_w,
                            low_h, 0, low_ran ? 1u : 0u, g.atlas_w, g.atlas_h};
  std::memcpy(&c[11], &alpha, 4);
  ID3D12Resource* const prev = g.hist[g.flip];
  ID3D12Resource* const next = g.hist[g.flip ^ 1];
  Barrier(cmd, g.raw, kUav, kSrv);
  Barrier(cmd, prev, kUav, kSrv);
  Bind(cmd, device, g.temporal_pso, frame, 4, {proxy, motion, prev, g.raw}, {answer, next},
       g.segs[g.prog.in_seg]->GetGPUVirtualAddress(), c);
  cmd->Dispatch((g.atlas_w + 7) / 8, (g.atlas_h + 7) / 8, 1);
  Barrier(cmd, g.raw, kSrv, kUav);
  Barrier(cmd, prev, kSrv, kUav);
  device->Release();
  g.flip ^= 1;
  g.hist_valid = true;
}

}  // namespace nr_engine
