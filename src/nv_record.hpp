#pragma once
// Recorder of NVIDIA's own NR kernels (experiments): NVIDIA's runtime runs its network as CUDA kernels on the D3D12
// command list through NVAPI (CreateCuModule -> CreateCuFunction -> LaunchCuKernelChain[Ex]); with openNR's DLL
// those are NVIDIA's kernels lowered to SM86. This hooks nvapi_QueryInterface and wraps those four entry points:
//   modules:  every cubin loaded is written to nv-record\module-<n>.bin beside the exe (once, as it is created),
//             and each function taken from it to nv-record\functions.txt ("F <handle> <module> <name>").
//   launches: RecordNV=<n> in the live ini records every launch inside the next n NR evaluates (the runner marks
//             them with Scope) to nv-record\launches.txt: "E <evaluate>" then per launch
//             "L <function> <gx> <gy> <gz> <bx> <by> <bz> <shared> <bytes> <hex of the parameter block>".
// Nothing is changed: every call goes through to dxvk-nvapi as made.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

#include <detours.h>

#include "ngx_probe.hpp"

namespace nv_record {

struct Dim3 {
  uint32_t x, y, z;
};
struct LaunchParams {   // NVAPI_CU_KERNEL_LAUNCH_PARAMS
  void* function;
  Dim3 grid, block;
  uint32_t shared;
  const void* params;
  uint32_t param_size;
};
struct LaunchParamsEx {   // NVAPI_CU_KERNEL_LAUNCH_PARAMS_EX
  void* function;
  Dim3 grid, block;
  uint32_t shared;
  const void* params;
  uint32_t param_size;
  void** kernel_params;
};

using QueryInterface = void*(__cdecl*)(uint32_t);
using CreateCuModule = int32_t(__cdecl*)(ID3D12Device*, const void*, uint32_t, void**);
using CreateCuFunction = int32_t(__cdecl*)(ID3D12Device*, void*, const char*, void**);
using LaunchChain = int32_t(__cdecl*)(ID3D12GraphicsCommandList*, const LaunchParams*, uint32_t);
using LaunchChainEx = int32_t(__cdecl*)(ID3D12GraphicsCommandList*, const LaunchParamsEx*, uint32_t);

constexpr uint32_t kCreateCuModule = 0xad1a677d;
constexpr uint32_t kCreateCuFunction = 0xe2436e22;
constexpr uint32_t kLaunchChain = 0x24973538;
constexpr uint32_t kLaunchChainEx = 0x846a9bf0;

inline QueryInterface real_query = nullptr;
inline CreateCuModule real_module = nullptr;
inline CreateCuFunction real_function = nullptr;
inline LaunchChain real_launch = nullptr;
inline LaunchChainEx real_launch_ex = nullptr;

inline std::mutex m;
inline std::string dir;            // nv-record\ beside the exe
inline uint32_t modules = 0;
inline std::atomic<int> requested{0};   // RecordNV: evaluates still to record
inline thread_local bool in_nr = false;
inline uint32_t evaluates = 0;

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
    if (FILE* f = Open("launches.txt", "a")) {
      std::fprintf(f, "E %u\n", evaluates++);
      std::fclose(f);
    }
  }
  ~Scope() {
    if (!on) return;
    in_nr = false;
    if (--requested <= 0) {
      requested = 0;
      ngx_probe::Log("nv-record: recording done");
    }
  }
};

inline void Record(void* function, const Dim3& g, const Dim3& b, uint32_t shared, const void* p, uint32_t n,
                   bool kernel_params) {
  std::lock_guard<std::mutex> lock(m);
  FILE* f = Open("launches.txt", "a");
  if (f == nullptr) return;
  std::fprintf(f, "L %p %u %u %u %u %u %u %u %u%s ", function, g.x, g.y, g.z, b.x, b.y, b.z, shared, n,
               kernel_params ? " kernelParams" : "");
  const auto* bytes = static_cast<const uint8_t*>(p);
  for (uint32_t i = 0; bytes != nullptr && i < n; ++i) std::fprintf(f, "%02x", bytes[i]);
  std::fputc('\n', f);
  std::fclose(f);
}

inline int32_t __cdecl HookModule(ID3D12Device* device, const void* blob, uint32_t size, void** module) {
  const int32_t r = real_module(device, blob, size, module);
  if (r == 0 && module != nullptr) {
    std::lock_guard<std::mutex> g(m);
    char name[64];
    std::snprintf(name, sizeof(name), "module-%u.bin", modules);
    if (FILE* f = Open(name, "wb")) {
      std::fwrite(blob, 1, size, f);
      std::fclose(f);
    }
    if (FILE* f = Open("functions.txt", "a")) {
      std::fprintf(f, "M %p %u %u\n", *module, modules, size);
      std::fclose(f);
    }
    ++modules;
  }
  return r;
}

inline int32_t __cdecl HookFunction(ID3D12Device* device, void* module, const char* name, void** function) {
  const int32_t r = real_function(device, module, name, function);
  if (r == 0 && function != nullptr) {
    std::lock_guard<std::mutex> g(m);
    if (FILE* f = Open("functions.txt", "a")) {
      std::fprintf(f, "F %p %p %s\n", *function, module, name != nullptr ? name : "?");
      std::fclose(f);
    }
  }
  return r;
}

inline int32_t __cdecl HookLaunch(ID3D12GraphicsCommandList* cmd, const LaunchParams* k, uint32_t n) {
  if (in_nr && k != nullptr)
    for (uint32_t i = 0; i < n; ++i)
      Record(k[i].function, k[i].grid, k[i].block, k[i].shared, k[i].params, k[i].param_size, false);
  return real_launch(cmd, k, n);
}

inline int32_t __cdecl HookLaunchEx(ID3D12GraphicsCommandList* cmd, const LaunchParamsEx* k, uint32_t n) {
  if (in_nr && k != nullptr)
    for (uint32_t i = 0; i < n; ++i)
      Record(k[i].function, k[i].grid, k[i].block, k[i].shared, k[i].params, k[i].param_size,
             k[i].params == nullptr && k[i].kernel_params != nullptr);
  return real_launch_ex(cmd, k, n);
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
    case kCreateCuModule: real_module = (CreateCuModule)f; return (void*)&HookModule;
    case kCreateCuFunction: real_function = (CreateCuFunction)f; return (void*)&HookFunction;
    case kLaunchChain: real_launch = (LaunchChain)f; return (void*)&HookLaunch;
    case kLaunchChainEx: real_launch_ex = (LaunchChainEx)f; return (void*)&HookLaunchEx;
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
