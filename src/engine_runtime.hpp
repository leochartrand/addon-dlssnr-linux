// Our own NR engine, in place of NVIDIA's NGX feature: frames of the Ampere engine (nrbench) recorded
// as precompiled cubins and launches (trace.py -> export_flat.py), replayed on the game's command
// list through NVAPI's cubin shaders. No translation at run time: the kernels are sm_86 machine code,
// their parameters packed once at load against the D3D12 buffers that hold the engine's memory.
//
// One program per split region, each at its own field: <exe dir>\nr-engine\crop and \low (program.txt,
// kernels\, segments\; without \low the crop's program runs both). They load on a thread of their own;
// until they are ready the NGX features keep running. Split foveation snaps the regions to the fields
// while the engine is on.
//
// Per region and frame, NVIDIA's temporal loop (engine_common.hlsli):
//   engine in:  the atlas region -> the engine's frame and history inputs (the previous answer,
//               reprojected)                                                      (engine_in)
//   engine:     the recorded launches, a UAV barrier after each
//   engine out: the head -> the answer atlas, blended with the history by the network's own logit;
//               also the next history                                             (engine_out)

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
constexpr uint32_t kConstants = 10;     // EngineConstants in engine_common.hlsli
constexpr uint32_t kUses = 4;           // descriptor sets per frame: in and out of each region
constexpr uint32_t kSlots = 5;          // t0-t2, u1-u2
constexpr uint64_t kStaging = 64ull << 20;
enum Region : uint32_t { kCrop, kLow, kRegions };

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
  std::string dir;
  std::vector<Kernel> kernels;
  std::vector<uint64_t> seg_size;
  std::vector<std::string> seg_file;
  std::vector<bool> seg_used;
  std::vector<Launch> launches;
  uint32_t in_seg = 0, out_seg = 0, hist_seg = 0;
  uint64_t in_off = 0, out_off = 0, hist_off = 0, in_bytes = 0, out_bytes = 0;
  uint32_t field_w = 0, field_h = 0;
  bool temporal = false;         // takes a history and returns the head (H and T lines)
  float blend_scale = 0.0f;
  std::vector<ID3D12Resource*> segs;
  uint64_t bytes = 0;            // device memory it holds

  D3D12_GPU_VIRTUAL_ADDRESS Va(uint32_t seg, uint64_t off) const { return segs[seg]->GetGPUVirtualAddress() + off; }
};

struct State {
  std::atomic<int> status{kIdle};
  bool parsed = false;
  bool parse_failed = false;
  Program progs[kRegions];
  int count = 0;                 // programs found (1: the crop's runs both regions)
  double load_ms = 0.0;
  char error[160] = {};

  LaunchCubin_t launch = nullptr;
  ID3D12RootSignature* root_sig = nullptr;
  ID3D12PipelineState* in_pso = nullptr;
  ID3D12PipelineState* out_pso = nullptr;
  ID3D12DescriptorHeap* heap = nullptr;
  uint32_t descriptor_size = 0;

  // The answers of the last frame and this one (UNORDERED_ACCESS between frames), atlas-sized; a
  // region has a history when last frame ran it.
  ID3D12Resource* hist[2] = {};
  uint32_t atlas_w = 0, atlas_h = 0;
  uint32_t flip = 0;
  bool ran_last[kRegions] = {};
  bool ran_now[kRegions] = {};
  bool reset = false;
  bool in_frame = false;
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

// program.txt (export_flat.py): K kernels, S segments, L launches, I/O/H the frame, head and history,
// F the field, T the blend cap. False with g.error set when it is unusable; `missing` when absent.
inline bool ParseOne(Program& p, const std::string& dir, bool* missing) {
  p.dir = dir;
  std::ifstream f(dir + "program.txt");
  *missing = !f;
  if (!f) return false;
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
          std::snprintf(g.error, sizeof(g.error), "%sprogram.txt: unknown parameter type %s", dir.c_str(), t.c_str());
          return false;
        }
        L.params.push_back(q);
      }
      p.launches.push_back(std::move(L));
    } else if (tag == "I") {
      s >> p.in_seg >> p.in_off >> p.in_bytes;
    } else if (tag == "O") {
      s >> p.out_seg >> p.out_off >> p.out_bytes;
    } else if (tag == "H") {
      uint64_t bytes = 0;
      s >> p.hist_seg >> p.hist_off >> bytes;
      p.temporal = true;
    } else if (tag == "F") {
      s >> p.field_w >> p.field_h;
    } else if (tag == "T") {
      s >> p.blend_scale;
    }
  }
  const uint64_t plane = 4ull * p.field_w * p.field_h;
  const size_t nseg = p.seg_size.size();
  if (!p.temporal) {
    std::snprintf(g.error, sizeof(g.error), "%sprogram.txt has no history input (re-trace it)", dir.c_str());
    return false;
  }
  if (p.kernels.empty() || p.launches.empty() || p.field_w == 0 || p.in_bytes != 3 * plane ||
      p.out_bytes != 4 * plane || p.in_seg >= nseg || p.out_seg >= nseg || p.hist_seg >= nseg) {
    std::snprintf(g.error, sizeof(g.error), "%sprogram.txt is incomplete (re-export it)", dir.c_str());
    return false;
  }
  // Only the segments a launch touches are created.
  p.seg_used.assign(nseg, false);
  p.seg_used[p.in_seg] = p.seg_used[p.out_seg] = p.seg_used[p.hist_seg] = true;
  for (const Launch& L : p.launches) {
    bool ok = L.kernel < p.kernels.size();
    for (const Param& q : L.params)
      if (q.kind == 0) {
        ok = ok && q.seg < nseg;
        if (q.seg < nseg) p.seg_used[q.seg] = true;
      }
    if (!ok) {
      std::snprintf(g.error, sizeof(g.error), "%sprogram.txt names an unknown kernel or segment", dir.c_str());
      return false;
    }
  }
  return true;
}

inline bool Parse() {
  if (g.parsed) return true;
  if (g.parse_failed) return false;
  char exe[MAX_PATH] = {};
  GetModuleFileNameA(nullptr, exe, MAX_PATH);
  if (char* slash = std::strrchr(exe, '\\')) *slash = 0;
  const std::string root = std::string(exe) + "\\nr-engine\\";
  bool missing = false;
  if (!ParseOne(g.progs[kCrop], root + "crop\\", &missing)) {
    g.parse_failed = true;
    Fail(missing ? "no nr-engine\\crop\\program.txt beside the exe" : g.error);
    return false;
  }
  g.count = 1;
  if (ParseOne(g.progs[kLow], root + "low\\", &missing)) {
    g.count = 2;
  } else if (!missing) {
    g.parse_failed = true;
    Fail(g.error);
    return false;
  }
  for (int r = 0; r < g.count; ++r)
    ngx_probe::Logf("nr-engine: %s program %zu kernels, %zu launches, field %ux%u, blend cap %.4f",
                    r == kCrop ? "crop" : "low", g.progs[r].kernels.size(), g.progs[r].launches.size(),
                    g.progs[r].field_w, g.progs[r].field_h, g.progs[r].blend_scale);
  g.parsed = true;
  return true;
}

// The program that runs a region.
inline const Program& Prog(uint32_t region) { return g.progs[region < (uint32_t)g.count ? region : kCrop]; }

// A region's field when the programs are there to run (0 otherwise): the size foveation lays it out at.
inline uint32_t FieldW(uint32_t region) { return Parse() ? Prog(region).field_w : 0; }
inline uint32_t FieldH(uint32_t region) { return Parse() ? Prog(region).field_h : 0; }

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

// The pipelines around the engine: [0] SRV table t0-t2, [1] UAV table u1-u2, [2] root UAV u0 (the
// engine's frame or head), [3] root UAV u3 (its history input), [4] constants b0, static linear s0.
inline bool CreatePipelines(ID3D12Device* device) {
  D3D12_DESCRIPTOR_RANGE srv = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0, 0};
  D3D12_DESCRIPTOR_RANGE uav = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 1, 0, 0};
  D3D12_ROOT_PARAMETER params[5] = {};
  params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  params[0].DescriptorTable = {1, &srv};
  params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  params[1].DescriptorTable = {1, &uav};
  params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
  params[2].Descriptor = {0, 0};
  params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
  params[3].Descriptor = {3, 0};
  params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  params[4].Constants = {0, 0, kConstants};
  D3D12_STATIC_SAMPLER_DESC sampler = {};
  sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
  sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
  sampler.MaxLOD = D3D12_FLOAT32_MAX;
  sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  D3D12_ROOT_SIGNATURE_DESC desc = {5, params, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE};
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
  D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kRingSize * kUses * kSlots,
                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
  if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g.heap)))) return false;
  g.descriptor_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  return true;
}

// The loader's own queue, for filling the buffers through a staging buffer.
struct Uploader {
  ID3D12CommandQueue* queue = nullptr;
  ID3D12CommandAllocator* alloc = nullptr;
  ID3D12GraphicsCommandList* list = nullptr;
  ID3D12Fence* fence = nullptr;
  ID3D12Resource* staging = nullptr;
  void* map = nullptr;
  HANDLE event = nullptr;
  uint64_t value = 0;

  bool Init(ID3D12Device* device) {
    event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_DIRECT};
    D3D12_HEAP_PROPERTIES up = {D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_RESOURCE_DESC sd = BufferDesc(kStaging, false);
    return SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) &&
           SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) &&
           SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr,
                                               IID_PPV_ARGS(&list))) &&
           SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))) &&
           SUCCEEDED(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &sd,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                     IID_PPV_ARGS(&staging))) &&
           SUCCEEDED(staging->Map(0, nullptr, &map));
  }
  void Submit() {
    list->Close();
    ID3D12CommandList* ls[] = {list};
    queue->ExecuteCommandLists(1, ls);
    queue->Signal(fence, ++value);
    fence->SetEventOnCompletion(value, event);
    WaitForSingleObject(event, 60000);
    alloc->Reset();
    list->Reset(alloc, nullptr);
  }
  ~Uploader() {
    for (IUnknown* u : {(IUnknown*)staging, (IUnknown*)list, (IUnknown*)alloc, (IUnknown*)fence, (IUnknown*)queue})
      if (u != nullptr) u->Release();
    if (event != nullptr) CloseHandle(event);
  }
};

// One program's buffers, cubin shaders and packed launches. Leaves g.error set on failure.
inline bool LoadProgram(ID3D12Device* device, Program& p, CreateCubinEx_t create, Uploader& up) {
  D3D12_HEAP_PROPERTIES def = {D3D12_HEAP_TYPE_DEFAULT};
  p.segs.assign(p.seg_size.size(), nullptr);
  for (size_t i = 0; i < p.seg_size.size(); ++i) {
    if (!p.seg_used[i]) continue;
    const bool has_file = p.seg_file[i] != "-";
    const D3D12_RESOURCE_DESC d = BufferDesc(p.seg_size[i], true);
    if (FAILED(device->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &d,
                                               has_file ? D3D12_RESOURCE_STATE_COPY_DEST
                                                        : D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                               nullptr, IID_PPV_ARGS(&p.segs[i])))) {
      std::snprintf(g.error, sizeof(g.error), "%s segment %zu (%llu bytes)", p.dir.c_str(), i,
                    (unsigned long long)p.seg_size[i]);
      return false;
    }
    p.bytes += p.seg_size[i];
    if (!has_file) continue;
    const std::vector<char> data = ReadAll(p.dir + p.seg_file[i]);
    if (data.size() != p.seg_size[i]) {
      std::snprintf(g.error, sizeof(g.error), "%s%s: %zu bytes, expected %llu", p.dir.c_str(),
                    p.seg_file[i].c_str(), data.size(), (unsigned long long)p.seg_size[i]);
      return false;
    }
    for (uint64_t at = 0; at < data.size(); at += kStaging) {
      const uint64_t n = data.size() - at < kStaging ? data.size() - at : kStaging;
      std::memcpy(up.map, data.data() + at, n);
      up.list->CopyBufferRegion(p.segs[i], at, up.staging, 0, n);
      up.Submit();
    }
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {p.segs[i], D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
    up.list->ResourceBarrier(1, &b);
    up.Submit();
  }

  for (Kernel& k : p.kernels) {
    const std::vector<char> cubin = ReadAll(p.dir + k.file);
    const int st = cubin.empty() ? -1
                                 : create(device, cubin.data(), (uint32_t)cubin.size(), k.block[0],
                                          k.block[1], k.block[2], k.shared, k.name.c_str(), &k.shader);
    if (st != 0) {
      std::snprintf(g.error, sizeof(g.error), "%s cubin shader %s (%d)", p.dir.c_str(), k.name.c_str(), st);
      return false;
    }
  }

  // The launches' parameters, packed once: the buffers never move.
  for (Launch& L : p.launches) {
    for (const Param& q : L.params) {
      const size_t size = q.kind == 0 || q.kind == 3 ? 8 : 4;
      while (L.packed.size() % size) L.packed.push_back(0);
      uint64_t v = q.kind == 0 ? p.Va(q.seg, q.off) : q.value;
      const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
      L.packed.insert(L.packed.end(), b, b + size);
    }
  }
  return true;
}

// The loader thread: NVAPI, the pipelines, every program. The device is free-threaded; nothing here
// touches the game's command lists.
inline void Load(ID3D12Device* device) {
  const ULONGLONG t0 = GetTickCount64();
  HMODULE nvapi = LoadLibraryA("nvapi64.dll");
  auto query = nvapi != nullptr ? (QueryInterface_t)GetProcAddress(nvapi, "nvapi_QueryInterface") : nullptr;
  auto init = query != nullptr ? (Initialize_t)query(kNvInitialize) : nullptr;
  auto create = query != nullptr ? (CreateCubinEx_t)query(kNvCreateCubinEx) : nullptr;
  g.launch = query != nullptr ? (LaunchCubin_t)query(kNvLaunchCubin) : nullptr;
  bool ok = init != nullptr && create != nullptr && g.launch != nullptr;
  if (!ok) std::snprintf(g.error, sizeof(g.error), "NVAPI cubin entry points missing (PROTON_ENABLE_NVAPI=1)");
  if (ok) init();
  if (ok && !CreatePipelines(device)) {
    ok = false;
    std::snprintf(g.error, sizeof(g.error), "engine pipelines");
  }
  uint64_t bytes = 0;
  {
    Uploader up;
    if (ok && !up.Init(device)) {
      ok = false;
      std::snprintf(g.error, sizeof(g.error), "upload objects");
    }
    for (int r = 0; ok && r < g.count; ++r) {
      ok = LoadProgram(device, g.progs[r], create, up);
      bytes += g.progs[r].bytes;
    }
  }
  device->Release();
  if (!ok) return Fail(g.error);
  g.load_ms = (double)(GetTickCount64() - t0);
  ngx_probe::Logf("nr-engine: ready -- %d program(s), %.0f MiB in %.1f s", g.count, bytes / 1048576.0,
                  g.load_ms / 1000.0);
  g.status = kReady;
}

// Start loading (once) when the programs are there. True once the engine can run.
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

inline bool EnsureHistory(ID3D12Device* device, uint32_t w, uint32_t h) {
  if (g.hist[0] != nullptr && g.atlas_w == w && g.atlas_h == h) return true;
  for (ID3D12Resource*& t : g.hist) {
    if (t != nullptr) g.graves.push_back({t, (int)kRingSize + 2});
    t = nullptr;
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
  for (ID3D12Resource*& t : g.hist)
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                               IID_PPV_ARGS(&t)))) {
      Fail("history textures");
      return false;
    }
  g.atlas_w = w;
  g.atlas_h = h;
  g.ran_last[kCrop] = g.ran_last[kLow] = false;
  return true;
}

// One use's descriptors (t0-t2 from `srv`, u1-u2 from `uav`; null entries read as zero), then the
// pipeline, tables, root UAVs and constants.
inline void Bind(ID3D12GraphicsCommandList* cmd, ID3D12Device* device, ID3D12PipelineState* pso,
                 uint32_t frame, uint32_t use, ID3D12Resource* const (&srv)[3],
                 ID3D12Resource* const (&uav)[2], D3D12_GPU_VIRTUAL_ADDRESS u0,
                 D3D12_GPU_VIRTUAL_ADDRESS u3, const uint32_t* constants) {
  const uint32_t base = ((frame % kRingSize) * kUses + use) * kSlots;
  D3D12_CPU_DESCRIPTOR_HANDLE cpu = g.heap->GetCPUDescriptorHandleForHeapStart();
  D3D12_GPU_DESCRIPTOR_HANDLE gpu = g.heap->GetGPUDescriptorHandleForHeapStart();
  cpu.ptr += (SIZE_T)base * g.descriptor_size;
  gpu.ptr += (UINT64)base * g.descriptor_size;
  for (int i = 0; i < 3; ++i) {
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
    h.ptr += (SIZE_T)(3 + i) * g.descriptor_size;
    device->CreateUnorderedAccessView(uav[i], nullptr, &v, h);
  }
  D3D12_GPU_DESCRIPTOR_HANDLE gpu_uav = gpu;
  gpu_uav.ptr += 3ull * g.descriptor_size;
  cmd->SetDescriptorHeaps(1, &g.heap);
  cmd->SetComputeRootSignature(g.root_sig);
  cmd->SetPipelineState(pso);
  cmd->SetComputeRootDescriptorTable(0, gpu);
  cmd->SetComputeRootDescriptorTable(1, gpu_uav);
  cmd->SetComputeRootUnorderedAccessView(2, u0);
  cmd->SetComputeRootUnorderedAccessView(3, u3);
  cmd->SetComputeRoot32BitConstants(4, kConstants, constants, 0);
}

// Before a frame's runs: `atlas` is the input atlas (its size is the history's); `reset` drops every
// region's history. Leaves last frame's answers readable until EndFrame.
inline bool BeginFrame(ID3D12GraphicsCommandList* cmd, ID3D12Resource* atlas, bool reset) {
  if (g.status.load() != kReady) return false;
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return false;
  TickGraves();
  const D3D12_RESOURCE_DESC d = atlas->GetDesc();
  const bool ok = EnsureHistory(device, (uint32_t)d.Width, d.Height);
  device->Release();
  if (!ok) return false;
  g.reset = reset;
  g.ran_now[kCrop] = g.ran_now[kLow] = false;
  Barrier(cmd, g.hist[g.flip], D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  g.in_frame = true;
  return true;
}

// One engine run on a region (x, y, w, h) of the atlas `src` with its motion vectors `motion` (both
// NON_PIXEL_SHADER_RESOURCE, left so): the answer lands in the same region of `dst` (UNORDERED_ACCESS).
// `history` scales the network's blend weight (1 = NVIDIA's, 0 = none).
inline bool Run(ID3D12GraphicsCommandList* cmd, uint32_t region, ID3D12Resource* src, ID3D12Resource* motion,
                ID3D12Resource* dst, uint32_t x, uint32_t y, uint32_t w, uint32_t h, uint32_t frame,
                float history) {
  if (!g.in_frame || region >= kRegions) return false;
  const Program& p = Prog(region);
  if (w > p.field_w || h > p.field_h || w == 0 || h == 0) return false;
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return false;

  history = history < 0.0f ? 0.0f : (history > 1.0f ? 1.0f : history);
  const bool valid = !g.reset && g.ran_last[region];
  const float blend = p.blend_scale * history;
  uint32_t c[kConstants] = {x, y, w, h, p.field_w, p.field_h, g.atlas_w, g.atlas_h, valid ? 1u : 0u, 0};
  std::memcpy(&c[9], &blend, 4);
  ID3D12Resource* const prev = g.hist[g.flip];
  ID3D12Resource* const next = g.hist[g.flip ^ 1];
  const D3D12_GPU_VIRTUAL_ADDRESS hist_in = p.Va(p.hist_seg, p.hist_off);

  D3D12_RESOURCE_BARRIER all = {};
  all.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;   // null resource: every UAV access before vs after
  Bind(cmd, device, g.in_pso, frame, region * 2, {src, motion, prev}, {nullptr, nullptr},
       p.Va(p.in_seg, p.in_off), hist_in, c);
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
  Bind(cmd, device, g.out_pso, frame, region * 2 + 1, {src, motion, prev}, {dst, next},
       p.Va(p.out_seg, p.out_off), hist_in, c);
  cmd->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
  cmd->ResourceBarrier(1, &all);
  device->Release();
  g.ran_now[region] = true;
  return true;
}

// After a frame's runs: this frame's answers become the history.
inline void EndFrame(ID3D12GraphicsCommandList* cmd) {
  if (!g.in_frame) return;
  Barrier(cmd, g.hist[g.flip], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  g.flip ^= 1;
  g.ran_last[kCrop] = g.ran_now[kCrop];
  g.ran_last[kLow] = g.ran_now[kLow];
  g.in_frame = false;
}

}  // namespace nr_engine
