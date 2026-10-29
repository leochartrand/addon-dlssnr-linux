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
#include <mutex>
#include <string>

#include <detours.h>

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
