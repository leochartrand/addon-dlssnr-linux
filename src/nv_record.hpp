#pragma once
// Recorder of NVIDIA's own NR kernels (experiments). Under dxvk-nvapi NVIDIA's runtime runs its network as cubin
// compute shaders on the D3D12 command list (CreateCubinComputeShaderWithName / Ex / ExV2, then LaunchCubinShader --
// the path our engine uses; CreateCuModule is not provided there); with openNR's DLL those are NVIDIA's kernels
// lowered to SM86. This hooks nvapi_QueryInterface and wraps those entry points:
//   shaders:  every cubin created goes to nv-record\shader-<n>.cubin beside the exe, with its handle, block, shared
//             memory and name to nv-record\shaders.txt;
//   launches: RecordNV=<n> in the live ini records every launch inside the next n NR evaluates (the runner marks
//             them with Scope) to nv-record\launches.txt: "E <evaluate>", then "L <handle> <grid> <bytes> <hex>".
//   interfaces.txt lists every NVAPI interface id asked for. Nothing is changed: every call goes through as made.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <d3d12.h>
#include <detours.h>
#include <reshade.hpp>

#include "diag.hpp"
#include "ngx_probe.hpp"

namespace nv_record {

using QueryInterface = void*(__cdecl*)(uint32_t);
using CreateWithName = int32_t(__cdecl*)(ID3D12Device*, const void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                         const char*, void**);
using CreateEx = int32_t(__cdecl*)(ID3D12Device*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                   const char*, void**);
struct CreateParams {   // NVAPI_D3D12_CREATE_CUBIN_SHADER_PARAMS
  size_t size_in, size_out;
  ID3D12Device* device;
  const void* cubin;
  uint32_t size, bx, by, bz, shared;
  const char* name;
  uint32_t flags;
  void* shader;
};
using CreateExV2 = int32_t(__cdecl*)(CreateParams*);
using Launch = int32_t(__cdecl*)(ID3D12GraphicsCommandList*, void*, uint32_t, uint32_t, uint32_t, const void*,
                                 uint32_t);

constexpr uint32_t kCreateWithName = 0x1dc7261f;   // NvAPI_D3D12_CreateCubinComputeShaderWithName
constexpr uint32_t kCreateEx = 0x3151211b;         // NvAPI_D3D12_CreateCubinComputeShaderEx
constexpr uint32_t kCreateExV2 = 0x299f5fdc;       // NvAPI_D3D12_CreateCubinComputeShaderExV2
constexpr uint32_t kLaunch = 0x5c52bb86;           // NvAPI_D3D12_LaunchCubinShader

inline QueryInterface real_query = nullptr;
inline CreateWithName real_with_name = nullptr;
inline CreateEx real_ex = nullptr;
inline CreateExV2 real_ex_v2 = nullptr;
inline Launch real_launch = nullptr;

inline std::mutex m;
inline std::string dir;            // nv-record\ beside the exe
inline uint32_t shaders = 0;
inline std::atomic<int> requested{0};   // RecordNV: evaluates still to record
inline thread_local bool in_nr = false;
inline uint32_t evaluates = 0;

// The last recorded evaluate's chain, for the replay test (ReplayNV): each launch's shader, grid and parameters.
struct Recorded {
  void* shader;
  uint32_t gx, gy, gz;
  std::vector<uint8_t> params;
};
inline std::vector<Recorded> chain, building;
inline bool replay = false;      // ReplayNV=1: the runner launches the recorded chain itself instead of evaluating
inline std::vector<std::pair<void*, std::string>> names;   // shader handle -> name (as created)
// ProfileNV=<n>: n replays with a GPU timestamp after every launch, written to nv-record\profile.txt.
inline int profile_requested = 0;
inline ID3D12QueryHeap* qheap = nullptr;
inline ID3D12Resource* qread = nullptr;
inline int pending = -1;          // replays until the last profiled one's timestamps are safe to read
inline std::vector<double> sums;  // per launch, ms summed over the profiled replays
inline int profiled = 0;
inline uint32_t counter = 0;     // the chain's frame counter (launch 1, offset 208: the noise seed)

inline FILE* Open(const char* name, const char* mode) {
  return std::fopen((dir + name).c_str(), mode);
}

// The runner marks each NR evaluate it makes; launches outside one are not recorded.
struct Scope {
  bool on;
  Scope() : on(requested.load() > 0) {
    if (!on) return;
    in_nr = true;
    std::lock_guard<std::mutex> g(m);
    building.clear();
    if (FILE* f = Open("launches.txt", "a")) {
      std::fprintf(f, "E %u\n", evaluates++);
      std::fclose(f);
    }
  }
  ~Scope() {
    if (!on) return;
    in_nr = false;
    {
      std::lock_guard<std::mutex> g(m);
      if (!building.empty()) chain.swap(building);
      if (chain.size() > 1 && chain[1].params.size() >= 212) std::memcpy(&counter, chain[1].params.data() + 208, 4);
    }
    if (--requested <= 0) {
      requested = 0;
      ngx_probe::Log("nv-record: recording done");
    }
  }
};

// A cubin shader as created: its bytes to shader-<n>.cubin, the handle and launch shape to shaders.txt
// ("S <handle> <n> <bytes> <bx> <by> <bz> <shared> <name>").
inline void Created(void* shader, const void* cubin, uint32_t size, uint32_t bx, uint32_t by, uint32_t bz,
                    uint32_t shared, const char* name) {
  std::lock_guard<std::mutex> g(m);
  char file[64];
  std::snprintf(file, sizeof(file), "shader-%u.cubin", shaders);
  if (FILE* f = Open(file, "wb")) {
    std::fwrite(cubin, 1, size, f);
    std::fclose(f);
  }
  names.emplace_back(shader, name != nullptr ? name : "-");
  if (FILE* f = Open("shaders.txt", "a")) {
    std::fprintf(f, "S %p %u %u %u %u %u %u %s\n", shader, shaders, size, bx, by, bz, shared,
                 name != nullptr ? name : "-");
    std::fclose(f);
  }
  ++shaders;
}

inline int32_t __cdecl HookWithName(ID3D12Device* d, const void* c, uint32_t n, uint32_t bx, uint32_t by,
                                    uint32_t bz, const char* name, void** out) {
  const int32_t r = real_with_name(d, c, n, bx, by, bz, name, out);
  if (r == 0 && out != nullptr) Created(*out, c, n, bx, by, bz, 0, name);
  return r;
}

inline int32_t __cdecl HookEx(ID3D12Device* d, const void* c, uint32_t n, uint32_t bx, uint32_t by, uint32_t bz,
                              uint32_t shared, const char* name, void** out) {
  const int32_t r = real_ex(d, c, n, bx, by, bz, shared, name, out);
  if (r == 0 && out != nullptr) Created(*out, c, n, bx, by, bz, shared, name);
  return r;
}

inline int32_t __cdecl HookExV2(CreateParams* p) {
  const int32_t r = real_ex_v2(p);
  if (r == 0 && p != nullptr) Created(p->shader, p->cubin, p->size, p->bx, p->by, p->bz, p->shared, p->name);
  return r;
}

// "L <handle> <gx> <gy> <gz> <bytes> <hex of the parameter block>"
inline int32_t __cdecl HookLaunch(ID3D12GraphicsCommandList* cmd, void* shader, uint32_t gx, uint32_t gy,
                                  uint32_t gz, const void* params, uint32_t size) {
  if (in_nr) {
    std::lock_guard<std::mutex> g(m);
    const auto* pb = static_cast<const uint8_t*>(params);
    building.push_back({shader, gx, gy, gz, std::vector<uint8_t>(pb, pb + (pb != nullptr ? size : 0))});
    if (FILE* f = Open("launches.txt", "a")) {
      std::fprintf(f, "L %p %u %u %u %u ", shader, gx, gy, gz, size);
      const auto* b = static_cast<const uint8_t*>(params);
      for (uint32_t i = 0; b != nullptr && i < size; ++i) std::fprintf(f, "%02x", b[i]);
      std::fputc('\n', f);
      std::fclose(f);
    }
  }
  return real_launch(cmd, shader, gx, gy, gz, params, size);
}

// The replay test: the recorded chain launched on `cmd` as NVIDIA's runtime launched it, the frame counter
// advanced (or held, with `hold`: the recorded frame again, for a bit-exact check), a UAV barrier after each launch
// (the safe order; NVIDIA's runtime may overlap its chained kernels). False when there is nothing to replay.
// Every buffer the device creates (ReShade's resource events): its GPU address range, so an address in a launch's
// parameters can be traced to the buffer that holds it and copied out whole (DumpNV).
struct Buffer {
  ID3D12Resource* res;
  uint64_t va, size;
};
inline std::vector<Buffer> buffers;
inline std::mutex buffers_m;

inline void OnInitResource(reshade::api::device* device, const reshade::api::resource_desc& desc,
                           const reshade::api::subresource_data*, reshade::api::resource_usage,
                           reshade::api::resource resource) {
  if (desc.type != reshade::api::resource_type::buffer || device->get_api() != reshade::api::device_api::d3d12) return;
  auto* r = reinterpret_cast<ID3D12Resource*>(resource.handle);
  const uint64_t va = r != nullptr ? r->GetGPUVirtualAddress() : 0;
  if (va == 0) return;
  std::lock_guard<std::mutex> g(buffers_m);
  buffers.push_back({r, va, desc.buffer.size});
}

inline void OnDestroyResource(reshade::api::device*, reshade::api::resource resource) {
  std::lock_guard<std::mutex> g(buffers_m);
  auto* r = reinterpret_cast<ID3D12Resource*>(resource.handle);
  for (size_t i = 0; i < buffers.size(); ++i)
    if (buffers[i].res == r) {
      buffers[i] = buffers.back();
      buffers.pop_back();
      return;
    }
}

inline const Buffer* Holding(uint64_t va) {
  for (const Buffer& b : buffers)
    if (va >= b.va && va < b.va + b.size) return &b;
  return nullptr;
}

// DumpNV=<k>: on the next replay, right after launch k (k >= the chain's length: after the last one, every buffer
// any launch names), each buffer that launch's parameters point into is copied to a readback buffer; four replays
// later they are written to nv-record\dump\<k>-<buffer address>.bin, and listed in dump\index.txt
// ("<k> <param offset> <param value> <buffer address> <bytes>").
inline int dump_requested = -1;
struct Pending {
  ID3D12Resource* readback;
  uint64_t va, size;
  int launch;
};
inline std::vector<Pending> dumps;
inline int dump_wait = -1;

inline void CopyOut(ID3D12GraphicsCommandList* cmd, int k, const std::vector<uint8_t>& params) {
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device)))) return;
  std::lock_guard<std::mutex> g(buffers_m);
  FILE* index = Open("dump\\index.txt", "a");
  for (size_t off = 0; off + 8 <= params.size(); off += 8) {
    uint64_t v = 0;
    std::memcpy(&v, params.data() + off, 8);
    const Buffer* b = v > 0x10000 ? Holding(v) : nullptr;
    if (b == nullptr) continue;
    bool seen = false;
    for (const Pending& d : dumps) seen |= d.va == b->va && d.launch == k;
    if (index != nullptr) std::fprintf(index, "%d %zu %llx %llx %llu\n", k, off, (unsigned long long)v,
                                       (unsigned long long)b->va, (unsigned long long)b->size);
    if (seen) continue;
    D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = b->size;
    rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* rb = nullptr;
    if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&rb)))) continue;
    cmd->CopyBufferRegion(rb, 0, b->res, 0, b->size);
    dumps.push_back({rb, b->va, b->size, k});
  }
  if (index != nullptr) std::fclose(index);
  device->Release();
}

inline void WriteDumps() {
  if (dump_wait < 0 || --dump_wait > 0) return;
  dump_wait = -1;
  for (Pending& d : dumps) {
    void* p = nullptr;
    D3D12_RANGE range = {0, (SIZE_T)d.size};
    if (SUCCEEDED(d.readback->Map(0, &range, &p))) {
      char name[96];
      std::snprintf(name, sizeof(name), "dump\\%d-%llx.bin", d.launch, (unsigned long long)d.va);
      if (FILE* f = Open(name, "wb")) {
        std::fwrite(p, 1, (size_t)d.size, f);
        std::fclose(f);
      }
      D3D12_RANGE none = {0, 0};
      d.readback->Unmap(0, &none);
    }
    d.readback->Release();
  }
  ngx_probe::Logf("nv-record: %zu buffers dumped", dumps.size());
  dumps.clear();
}

inline const char* NameOf(void* shader) {
  for (const auto& n : names)
    if (n.first == shader) return n.second.c_str();
  return "?";
}

// The timestamps of the profiled replay four replays ago (no longer in flight): added up per launch; after the last
// requested one, the averages go to profile.txt.
inline void CollectProfile() {
  if (pending < 0 || --pending > 0) return;
  pending = -1;
  const size_t n = chain.size() + 1;
  uint64_t* t = nullptr;
  D3D12_RANGE range = {0, n * 8};
  if (FAILED(qread->Map(0, &range, reinterpret_cast<void**>(&t)))) return;
  const double freq = (double)diag::timestamp_freq.load();
  sums.resize(chain.size(), 0.0);
  for (size_t i = 0; i < chain.size() && freq > 0; ++i) sums[i] += (double)(t[i + 1] - t[i]) * 1000.0 / freq;
  D3D12_RANGE none = {0, 0};
  qread->Unmap(0, &none);
  ++profiled;
  if (profile_requested > 0) return;
  if (FILE* f = Open("profile.txt", "w")) {
    double total = 0.0;
    for (size_t i = 0; i < chain.size(); ++i) total += sums[i] / profiled;
    std::fprintf(f, "# %d replays, %.3f ms per replay (launch, ms, grid, name)\n", profiled, total);
    for (size_t i = 0; i < chain.size(); ++i)
      std::fprintf(f, "%3zu %8.4f %5u %5u %3u %s\n", i, sums[i] / profiled, chain[i].gx, chain[i].gy, chain[i].gz,
                   NameOf(chain[i].shader));
    std::fclose(f);
  }
  ngx_probe::Logf("nv-record: profile of %d replays written", profiled);
  sums.clear();
  profiled = 0;
}

inline bool Replay(ID3D12GraphicsCommandList* cmd, bool hold) {
  std::lock_guard<std::mutex> g(m);
  if (chain.empty() || real_launch == nullptr) return false;
  CollectProfile();
  WriteDumps();
  const int dump_at = dump_wait < 0 ? dump_requested : -1;
  if (dump_at >= 0) {
    dump_requested = -1;
    CreateDirectoryA((dir + "dump").c_str(), nullptr);
  }
  if (!hold) ++counter;
  // Timestamps for this replay when a profile is asked for and the last one has been read.
  bool timed = false;
  const uint32_t n = (uint32_t)chain.size() + 1;
  if (profile_requested > 0 && pending < 0) {
    ID3D12Device* device = nullptr;
    if (qheap == nullptr && SUCCEEDED(cmd->GetDevice(IID_PPV_ARGS(&device)))) {
      D3D12_QUERY_HEAP_DESC qd = {D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 256, 0};
      device->CreateQueryHeap(&qd, IID_PPV_ARGS(&qheap));
      D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_READBACK};
      D3D12_RESOURCE_DESC rd = {};
      rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
      rd.Width = 256 * 8;
      rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
      rd.SampleDesc.Count = 1;
      rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
      device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                      IID_PPV_ARGS(&qread));
      device->Release();
    }
    timed = qheap != nullptr && qread != nullptr && n <= 256;
    if (timed) {
      --profile_requested;
      cmd->EndQuery(qheap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
    }
  }
  D3D12_RESOURCE_BARRIER all = {};
  all.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  std::vector<uint8_t> p;
  for (size_t i = 0; i < chain.size(); ++i) {
    const Recorded& r = chain[i];
    p = r.params;
    if (i == 1 && p.size() >= 212) std::memcpy(p.data() + 208, &counter, 4);
    if (real_launch(cmd, r.shader, r.gx, r.gy, r.gz, p.data(), (uint32_t)p.size()) != 0) return false;
    cmd->ResourceBarrier(1, &all);
    if (timed) cmd->EndQuery(qheap, D3D12_QUERY_TYPE_TIMESTAMP, (UINT)i + 1);
    if (dump_at == (int)i) CopyOut(cmd, (int)i, p);
  }
  if (dump_at >= (int)chain.size())
    for (size_t i = 0; i < chain.size(); ++i) CopyOut(cmd, dump_at, chain[i].params);
  if (dump_at >= 0) dump_wait = 4;
  if (timed) {
    cmd->ResolveQueryData(qheap, D3D12_QUERY_TYPE_TIMESTAMP, 0, n, qread, 0);
    pending = 4;
  }
  return true;
}

inline void* __cdecl HookQuery(uint32_t id) {
  void* f = real_query(id);
  {   // every interface id asked for, once (which NVAPI paths NVIDIA's runtime uses here)
    static std::mutex seen_m;
    static uint32_t seen[512];
    static uint32_t n_seen = 0;
    std::lock_guard<std::mutex> g(seen_m);
    bool known = false;
    for (uint32_t i = 0; i < n_seen; ++i) known |= seen[i] == id;
    if (!known && n_seen < 512) {
      seen[n_seen++] = id;
      if (FILE* q = Open("interfaces.txt", "a")) {
        std::fprintf(q, "%08x %s\n", id, f != nullptr ? "yes" : "null");
        std::fclose(q);
      }
    }
  }
  if (f == nullptr) return f;
  switch (id) {
    case kCreateWithName: real_with_name = (CreateWithName)f; return (void*)&HookWithName;
    case kCreateEx: real_ex = (CreateEx)f; return (void*)&HookEx;
    case kCreateExV2: real_ex_v2 = (CreateExV2)f; return (void*)&HookExV2;
    case kLaunch: real_launch = (Launch)f; return (void*)&HookLaunch;
    default: return f;
  }
}

// At add-on load, before NVIDIA's runtime resolves its NVAPI entry points.
inline void Install(const char* game_dir) {
  HMODULE nvapi = LoadLibraryA("nvapi64.dll");
  real_query = nvapi != nullptr ? (QueryInterface)GetProcAddress(nvapi, "nvapi_QueryInterface") : nullptr;
  if (real_query == nullptr) {
    ngx_probe::Log("nv-record: nvapi_QueryInterface not found");
    return;
  }
  dir = std::string(game_dir) + "\\nv-record\\";
  CreateDirectoryA(dir.c_str(), nullptr);
  DetourTransactionBegin();
  DetourUpdateThread(GetCurrentThread());
  DetourAttach(reinterpret_cast<PVOID*>(&real_query), reinterpret_cast<PVOID>(&HookQuery));
  const LONG r = DetourTransactionCommit();
  ngx_probe::Logf("nv-record: nvapi_QueryInterface hooked (%ld); modules and functions to %s", r, dir.c_str());
}

inline void Register() {
  reshade::register_event<reshade::addon_event::init_resource>(OnInitResource);
  reshade::register_event<reshade::addon_event::destroy_resource>(OnDestroyResource);
}

inline void Unregister() {
  reshade::unregister_event<reshade::addon_event::init_resource>(OnInitResource);
  reshade::unregister_event<reshade::addon_event::destroy_resource>(OnDestroyResource);
}

// On a real FreeLibrary of the add-on (ReShade cycles it while creating the device).
inline void Uninstall() {
  if (real_query == nullptr) return;
  DetourTransactionBegin();
  DetourUpdateThread(GetCurrentThread());
  DetourDetach(reinterpret_cast<PVOID*>(&real_query), reinterpret_cast<PVOID>(&HookQuery));
  DetourTransactionCommit();
  real_query = nullptr;
}

}  // namespace nv_record
