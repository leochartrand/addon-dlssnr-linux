// Milestone 5a: the visible loop. The DLSSNR model runs once per frame off the game's DLSS-SR
// output and its answer is copied back over that output, so the game's own post-processing and
// tone-mapping consume the enhanced frame. F10 toggles the pass for an instant A/B.
//
// (Milestone 4 proved the mechanism: the game-local nvngx_dlssnr.dll is loaded directly through a
// forwarder DLL whose name contains "nvngx.dll" -- the snippet's caller gate -- bypassing driver
// dispatch, which fails FAIL_OutOfDate under Proton because the NGX OTA updater is unavailable.)
//
// Recipe from OptiScaler_DLSSNR (MIT). The parameter block is the driver core's own capability
// block -- a freshly allocated one lacks the snippet/preset callbacks and create fails
// UnableToInitializeFeature. Its setters are not laid out the way the SDK header declares: the
// 64-bit setter is vtable slot 0 (resources go through it), the uint setter slot 3, and the float
// setter's slot is discovered by round-tripping a value (typed Get() works normally).
//
// This stage feeds the model the linear frame as-is. The display-referred encode with a measured
// white point -- the actual color bridge -- is stage 5b; the F10 A/B this stage provides is how
// that work gets judged.
//
// Inputs, from the fork and our WuWa capture: Color and Output at display resolution, Depth and
// MVec at render resolution (the guides the game hands DLSS), each with its own subrect; MV scale
// passed through from the game, never derived.

#pragma once

#include <windows.h>

#include <bcrypt.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <d3d12.h>
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>

#include "nr_compose.hpp"
#include "fov.hpp"
#include "diag.hpp"

#pragma comment(lib, "bcrypt")   // SHA-256 of the model DLL
#pragma comment(lib, "version")  // its file-version resource

namespace nr_runner {

// --- driver parameter block, driven through its vtable ------------------

constexpr int kVtSetUll = 0;
constexpr int kVtSetUint = 3;

using PFN_SetUll = void(__thiscall*)(void*, const char*, unsigned long long);
using PFN_SetFloat = void(__thiscall*)(void*, const char*, float);
using PFN_SetUint = void(__thiscall*)(void*, const char*, unsigned int);

inline int float_slot = -1;

inline void SetUInt(void* params, const char* name, unsigned int v) {
  void** vt = *reinterpret_cast<void***>(params);
  reinterpret_cast<PFN_SetUint>(vt[kVtSetUint])(params, name, v);
}

inline void SetFloat(void* params, const char* name, float v) {
  void** vt = *reinterpret_cast<void***>(params);
  reinterpret_cast<PFN_SetFloat>(vt[float_slot < 0 ? 1 : float_slot])(params, name, v);
}

inline void SetResource(void* params, const char* name, ID3D12Resource* v) {
  void** vt = *reinterpret_cast<void***>(params);
  reinterpret_cast<PFN_SetUll>(vt[kVtSetUll])(params, name, (unsigned long long)v);
}

// Find the float slot by writing a known value through each candidate setter and reading it back
// with the typed getter. 0.3125 is exact in binary, so the comparison is clean.
inline void DiscoverFloatSlot(NVSDK_NGX_Parameter* params) {
  if (float_slot >= 0) return;
  void** vt = *reinterpret_cast<void***>(params);
  for (int slot = 0; slot < 8; ++slot) {
    const float probe = 0.3125f;
    float read_back = 0.0f;
    reinterpret_cast<PFN_SetFloat>(vt[slot])(params, "NRProbe.Float", probe);
    if (params->Get("NRProbe.Float", &read_back) == NVSDK_NGX_Result_Success && read_back == probe) {
      float_slot = slot;
      ngx_probe::Logf("nr-fwd: float parameters go through vtable slot %d", slot);
      return;
    }
  }
  float_slot = 1;  // the header's answer, as a last resort
  ngx_probe::Log("nr-fwd: no float slot round-tripped; falling back to the header's slot 1");
}

// --- state ---------------------------------------------------------------

using PFN_FwdInit = int(__cdecl*)(const wchar_t*, const wchar_t*, ID3D12Device*, void*);
using PFN_FwdCreate = void*(__cdecl*)(ID3D12GraphicsCommandList*, void*, int*);
using PFN_FwdEvaluate = int(__cdecl*)(ID3D12GraphicsCommandList*, void*, void*);
using PFN_FwdRelease = void(__cdecl*)(void*);
using PFN_GetCapabilityParams = int(__cdecl*)(NVSDK_NGX_Parameter**);

constexpr uint32_t kFullLogs = 3;        // first evals logged in full
constexpr uint32_t kHeartbeatEvery = 600;
constexpr uint32_t kMaxFailStreak = 3;   // consecutive eval failures before the pass disables itself
constexpr int kGraveyardFrames = 240;    // frames a retired resource waits before release

struct Grave {
  ID3D12Resource* resource = nullptr;
  void* feature = nullptr;
  int frames = 0;
};

struct State {
  bool gave_up = false;
  bool enabled = true;

  HMODULE forwarder = nullptr;
  PFN_FwdInit init = nullptr;
  PFN_FwdCreate create = nullptr;
  PFN_FwdEvaluate evaluate = nullptr;
  PFN_FwdRelease release = nullptr;

  NVSDK_NGX_Parameter* caps = nullptr;
  bool snippet_inited = false;
  void* feature = nullptr;
  // Split foveation with two calls: feature runs the crop and feature2 the low-res frame, each on
  // its own region of the one atlas, so neither's grading sees the other's content.
  void* feature2 = nullptr;
  int feature_calls = 0;                 // 1 or 2, as the features were built
  ID3D12Resource* output = nullptr;      // the model's answer
  ID3D12Resource* color_copy = nullptr;  // the original, copied aside for the resolve
  ID3D12Resource* proxy = nullptr;       // the display-referred picture the model is shown
  ID3D12Resource* pre_frame = nullptr;   // before_upscale: our copy of DLSS-SR's input, processed
  DXGI_FORMAT output_format = DXGI_FORMAT_UNKNOWN;

  // Compose controls (fork semantics: transfer 0 = bypass, 1 = the model's picture; max_ratio
  // caps how far the model may move any pixel's luminance either way; colour 0 keeps the game's
  // own hue exactly -- only brightness carries the model's verdict).
  float transfer = 1.0f;
  float max_ratio = 2.0f;
  float colour_strength = 0.0f;

  // White point: the measured log-average luminance, EMA-smoothed; -1 until the first reading
  // lands. wp_scale is the user's paper-white multiplier on top.
  float wp_ema = -1.0f;
  float wp_scale = 1.0f;

  // Debug view: 0 off, 1 proxy, 2 model answer, 3 difference x20.
  int debug_view = 0;

  // Model tuning, read once when the feature is built; changing them retires the feature.
  // The three strengths mirror the RenoDX DLSS5 addon's intensities: intensity is the model's
  // overall hand (DLSSNR.Intensity), local_structure is the fine-detail synthesis
  // (LocalStructureStrength), local_tone is the broad local-contrast/tone shaping (LocalToneStrength).
  float intensity = 1.0f;
  float local_structure = 1.0f;
  float local_tone = 1.0f;
  int preset = 0;
  int style = 0;
  // Further model parameters, for experiments (-1 = leave unset, the model's own default):
  // MaskBlendMode 0-3 (NVIDIA's default 1), UICorrection and UseAutoMask 0/1 (we used to force 1),
  // SkinStructureStrength (we used to send -1), GlobalToneStrength, ScalingRatio (0 = unset).
  int mask_blend_mode = -1;
  int ui_correction = 1;
  int auto_mask = 1;
  float skin_structure = -1.0f;
  float global_tone = -1.0f;
  float scaling_ratio = 0.0f;
  // Split in two calls: the crop's network, when it should differ from `preset` (-1 = the same).
  // Lets the low-res pass, which sets the look everywhere, run a network that holds up at small
  // sizes while the full-res crop runs a cheaper one.
  int split_crop_preset = -1;

  // Geometry of the game's current DLSS-SR feature.
  unsigned render_w = 0, render_h = 0;
  unsigned out_w = 0, out_h = 0;
  // Where the pass runs: on DLSS-SR's output (the default), or with before_upscale on its
  // render-resolution input, so the model sees one uniform scale and DLSS-SR upscales its answer.
  // frame_w/h is the size of the picture the pass works on; textures are built at it.
  bool before_upscale = false;
  unsigned frame_w = 0, frame_h = 0;
  unsigned tex_w = 0, tex_h = 0;
  int create_flags = 0;
  // Geometry the NR feature was built against; a mismatch retires it.
  unsigned feature_w = 0, feature_h = 0;

  uint32_t eval_count = 0;
  uint32_t fail_streak = 0;

  // Set by the overlay thread (Apply button); consumed on the evaluate thread.
  std::atomic<bool> retire_pending{false};

  std::vector<Grave> graveyard;

  wchar_t game_dir[MAX_PATH] = {};
};

inline State s;

// The size of the picture the pass works on, for the current mode. Before the upscale only means
// something when DLSS-SR really upscales; at DLAA (input = output size) both modes are the same.
inline void UpdateFrameSize() {
  const bool before = s.before_upscale && s.render_w != 0 && s.render_h != 0 &&
                      (s.render_w != s.out_w || s.render_h != s.out_h);
  s.frame_w = before ? s.render_w : s.out_w;
  s.frame_h = before ? s.render_h : s.out_h;
}

// Model calls per frame the current settings ask for: 2 for split foveation in separate calls.
inline int WantedCalls() {
  return fov::settings.enabled && fov::settings.split && fov::settings.split_calls == 2 ? 2 : 1;
}

// The size the NR feature runs at: the display resolution, or the squeezed working size.
inline void FeatureSize(unsigned* w, unsigned* h) {
  *w = s.frame_w;
  *h = s.frame_h;
  if (fov::settings.enabled && s.frame_w != 0 && s.frame_h != 0) {
    uint32_t fw = 0, fh = 0;
    fov::WorkSize(s.frame_w, s.frame_h, &fw, &fh);
    *w = fw;
    *h = fh;
  }
}

inline const char* ResultName(int r) {
  switch ((unsigned)r) {
    case 0x1: return "Success";
    case 0xBAD00001: return "FAIL_FeatureNotSupported";
    case 0xBAD00002: return "FAIL_PlatformError";
    case 0xBAD00005: return "FAIL_InvalidParameter";
    case 0xBAD00008: return "FAIL_UnsupportedInputFormat";
    case 0xBAD00009: return "FAIL_RWFlagMissing";
    case 0xBAD0000A: return "FAIL_MissingInput";
    case 0xBAD0000B: return "FAIL_UnableToInitializeFeature";
    case 0xBAD0000C: return "FAIL_OutOfDate";
    case 0xBAD0000D: return "FAIL_OutOfGPUMemory";
    case 0xBAD0000E: return "FAIL_UnsupportedFormat";
    case 0xBAD00010: return "FAIL_UnsupportedParameter";
    case 0xBAD00011: return "FAIL_Denied";
    default: return "other";
  }
}

inline bool GiveUp(const char* why) {
  ngx_probe::Logf("nr-fwd: giving up -- %s", why);
  s.gave_up = true;
  return false;
}

// Test-only e2e levers (NR_TEST_HOOK_* macros); expands to nothing in release builds.
#include "nr_test_hooks.hpp"

// --- model identity -------------------------------------------------------

// Feature 18 only runs stably with the exact model build it was tested against. A mismatched
// nvngx_dlssnr.dll does not fail at the API -- it reports Success on every evaluate and crashes
// the game minutes later on an unrelated thread. So the model's version and SHA-256 are logged at
// init (on a background thread; the file is ~160MB), with a warning when it isn't the tested
// build, so every user report identifies the model that was actually running.
constexpr char kTestedModelSha256[] =
    "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e";

inline HANDLE model_id_thread = nullptr;

inline DWORD WINAPI ModelIdentityThread(LPVOID arg) {
  wchar_t* path = static_cast<wchar_t*>(arg);

  char version[64] = "unknown";
  DWORD handle = 0;
  if (const DWORD info_size = GetFileVersionInfoSizeW(path, &handle)) {
    void* info = std::malloc(info_size);
    VS_FIXEDFILEINFO* ffi = nullptr;
    UINT ffi_len = 0;
    if (info != nullptr && GetFileVersionInfoW(path, 0, info_size, info) &&
        VerQueryValueW(info, L"\\", reinterpret_cast<void**>(&ffi), &ffi_len) && ffi != nullptr) {
      std::snprintf(version, sizeof(version), "%u.%u.%u.%u", HIWORD(ffi->dwFileVersionMS),
                    LOWORD(ffi->dwFileVersionMS), HIWORD(ffi->dwFileVersionLS),
                    LOWORD(ffi->dwFileVersionLS));
    }
    std::free(info);
  }

  char sha_hex[65] = "unreadable";
  unsigned long long total = 0;
  bool hashed = false;
  const HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                  FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
  if (file != INVALID_HANDLE_VALUE) {
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    constexpr DWORD kChunk = 1 << 20;
    void* buf = std::malloc(kChunk);
    if (buf != nullptr &&
        BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) == 0 &&
        BCryptCreateHash(alg, &hash, nullptr, 0, nullptr, 0, 0) == 0) {
      unsigned char digest[32];
      DWORD read = 0;
      bool ok = true;
      while (ReadFile(file, buf, kChunk, &read, nullptr) && read > 0) {
        if (BCryptHashData(hash, static_cast<PUCHAR>(buf), read, 0) != 0) {
          ok = false;
          break;
        }
        total += read;
      }
      if (ok && BCryptFinishHash(hash, digest, sizeof(digest), 0) == 0) {
        for (int i = 0; i < 32; ++i) std::snprintf(sha_hex + i * 2, 3, "%02x", digest[i]);
        hashed = true;
      }
    }
    if (hash != nullptr) BCryptDestroyHash(hash);
    if (alg != nullptr) BCryptCloseAlgorithmProvider(alg, 0);
    std::free(buf);
    CloseHandle(file);
  }

  ngx_probe::Logf("nr-fwd: model nvngx_dlssnr.dll -- version %s, %llu bytes, sha256=%s", version,
                  total, sha_hex);
  if (hashed && std::strcmp(sha_hex, kTestedModelSha256) != 0) {
    ngx_probe::Warnf(
        "nr-fwd: this nvngx_dlssnr.dll is NOT the tested model build -- other versions have "
        "crashed the game mid-session with no API error; tested sha256=%s",
        kTestedModelSha256);
  }
  std::free(path);
  return 0;
}

inline void LogModelIdentityAsync(const wchar_t* model_path) {
  wchar_t* copy = _wcsdup(model_path);
  if (copy == nullptr) return;
  model_id_thread = CreateThread(nullptr, 0, ModelIdentityThread, copy, 0, nullptr);
  if (model_id_thread == nullptr) std::free(copy);
}

// --- lifetime ------------------------------------------------------------

// The GPU may still be executing command lists that reference a retired feature or texture, so
// they wait in the graveyard for a couple hundred frames before release. The graveyard grows as
// needed: releasing early instead used to free a texture the GPU was still reading -- three quick
// Apply presses overflowed the old fixed 8 slots and killed the device (issue #1).
inline void Bury(ID3D12Resource* resource, void* feature) {
  if (resource == nullptr && feature == nullptr) return;
  for (auto& g : s.graveyard) {
    if (g.resource == nullptr && g.feature == nullptr) {
      g.resource = resource;
      g.feature = feature;
      g.frames = kGraveyardFrames;
      return;
    }
  }
  s.graveyard.push_back({resource, feature, kGraveyardFrames});
}

inline void TickGraveyard() {
  for (auto& g : s.graveyard) {
    if (g.resource == nullptr && g.feature == nullptr) continue;
    if (--g.frames > 0) continue;
    if (g.resource != nullptr) g.resource->Release();
    if (g.feature != nullptr && s.release != nullptr) s.release(g.feature);
    g = {};
  }
}

inline void RetireFeature(const char* why) {
  if (s.feature == nullptr && s.output == nullptr) return;
  ngx_probe::Logf("nr-fwd: retiring NR feature (%s)", why);
  Bury(s.output, s.feature);
  Bury(nullptr, s.feature2);
  s.feature2 = nullptr;
  s.feature_calls = 0;
  Bury(s.color_copy, nullptr);
  Bury(s.proxy, nullptr);
  s.feature = nullptr;
  s.output = nullptr;
  s.color_copy = nullptr;
  s.proxy = nullptr;
  s.output_format = DXGI_FORMAT_UNKNOWN;
  s.feature_w = s.feature_h = 0;
}

// Remember the game's DLSS-SR geometry; the NR feature is built against the display resolution and
// its guides against the render resolution.
inline void OnDlssCreate(unsigned w, unsigned h, unsigned ow, unsigned oh, int flags) {
  s.render_w = w;
  s.render_h = h;
  s.out_w = ow;
  s.out_h = oh;
  s.create_flags = flags;
  UpdateFrameSize();
  unsigned fw = 0, fh = 0;
  FeatureSize(&fw, &fh);
  if (s.feature != nullptr && (s.feature_w != fw || s.feature_h != fh))
    RetireFeature("geometry changed");
}

// Process teardown: pointers in the shared block must not outlive us.
inline void Shutdown() {
  if (model_id_thread != nullptr) {
    WaitForSingleObject(model_id_thread, 3000);  // the hash thread must not outlive the DLL
    CloseHandle(model_id_thread);
    model_id_thread = nullptr;
  }
  if (s.caps != nullptr) {
    void* p = s.caps;
    SetUInt(p, "DLSSNR.Enabled", 0u);
    SetResource(p, "DLSSNR.Color", nullptr);
    SetResource(p, "DLSSNR.Depth", nullptr);
    SetResource(p, "DLSSNR.MVec", nullptr);
    SetResource(p, "DLSSNR.Output", nullptr);
  }
  if (s.feature != nullptr && s.release != nullptr) s.release(s.feature);
  s.feature = nullptr;
  if (s.feature2 != nullptr && s.release != nullptr) s.release(s.feature2);
  s.feature2 = nullptr;
  if (s.output != nullptr) s.output->Release();
  s.output = nullptr;
  if (s.color_copy != nullptr) s.color_copy->Release();
  s.color_copy = nullptr;
  if (s.proxy != nullptr) s.proxy->Release();
  s.proxy = nullptr;
  if (s.pre_frame != nullptr) s.pre_frame->Release();
  s.pre_frame = nullptr;
  for (auto& g : s.graveyard) {
    if (g.resource != nullptr) g.resource->Release();
    if (g.feature != nullptr && s.release != nullptr) s.release(g.feature);
    g = {};
  }
  nr_compose::Release();
  fov::Release();
}

inline void SetEnabled(bool on) {
  if (s.enabled == on) return;
  s.enabled = on;
  s.fail_streak = 0;
  ngx_probe::Logf("nr-fwd: pass %s", on ? "ENABLED" : "DISABLED");
}

inline bool IsEnabled() { return s.enabled; }

// --- setup ---------------------------------------------------------------

inline bool EnsureSetup(ID3D12GraphicsCommandList* cmd) {
  if (s.caps != nullptr && s.forwarder != nullptr) return true;
  NR_TEST_HOOKS_INIT();

  // Everything lives beside the game exe: the forwarder, the model, and our data path.
  if (s.game_dir[0] == 0) {
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    wchar_t* slash = wcsrchr(exe, L'\\');
    if (slash == nullptr) return GiveUp("could not resolve the game directory");
    *slash = 0;
    wcscpy_s(s.game_dir, exe);
  }

  if (s.forwarder == nullptr) {
    wchar_t path[MAX_PATH] = {};
    swprintf_s(path, L"%s\\nvngx.dll_nrfwd.dll", s.game_dir);
    s.forwarder = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (s.forwarder == nullptr) return GiveUp("nvngx.dll_nrfwd.dll not found beside the game exe");
    s.init = (PFN_FwdInit)GetProcAddress(s.forwarder, "nrfwd_init");
    s.create = (PFN_FwdCreate)GetProcAddress(s.forwarder, "nrfwd_create");
    s.evaluate = (PFN_FwdEvaluate)GetProcAddress(s.forwarder, "nrfwd_evaluate");
    s.release = (PFN_FwdRelease)GetProcAddress(s.forwarder, "nrfwd_release");
    if (s.init == nullptr || s.create == nullptr || s.evaluate == nullptr || s.release == nullptr)
      return GiveUp("forwarder exports did not resolve");
    ngx_probe::Log("nr-fwd: forwarder loaded");
  }

  if (s.caps == nullptr) {
    // The driver core's capability block. The game's own DLSS already initialised the core, so this
    // is a read of existing state, not an init.
    HMODULE nvngx = GetModuleHandleA("_nvngx.dll");
    if (nvngx == nullptr) return GiveUp("_nvngx.dll not resident");
    auto get_caps =
        (PFN_GetCapabilityParams)GetProcAddress(nvngx, "NVSDK_NGX_D3D12_GetCapabilityParameters");
    if (get_caps == nullptr) return GiveUp("GetCapabilityParameters not exported by _nvngx.dll");
    if (get_caps(&s.caps) != NVSDK_NGX_Result_Success || s.caps == nullptr) {
      s.caps = nullptr;
      return GiveUp("the NGX core refused its capability parameters");
    }
    DiscoverFloatSlot(s.caps);
  }

  return true;
}

// Three display-resolution textures in the game's own DLSS output format (accepted by the model
// in milestone 5a): the model's answer, the copied-aside original, and the encoded proxy.
inline bool CreateTextures(ID3D12GraphicsCommandList* cmd, ID3D12Resource* game_output) {
  if (s.output != nullptr) return true;
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
    return GiveUp("could not reach the device from the command list");

  const D3D12_RESOURCE_DESC game_desc = game_output->GetDesc();

  D3D12_HEAP_PROPERTIES heap = {};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC desc = {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  desc.Width = s.frame_w;
  desc.Height = s.frame_h;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.Format = game_desc.Format;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  // colorCopy must match the game's output format exactly (CopyResource); the proxy and the
  // model's answer go FP16 -- R11G11B10's 5-bit blue mantissa rounds the encode toward green.
  struct {
    ID3D12Resource** texture;
    DXGI_FORMAT format;
  } textures[3] = {
      {&s.output, DXGI_FORMAT_R16G16B16A16_FLOAT},
      {&s.color_copy, game_desc.Format},
      {&s.proxy, DXGI_FORMAT_R16G16B16A16_FLOAT},
  };
  bool ok = true;
  for (auto& t : textures) {
    desc.Format = t.format;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                               IID_PPV_ARGS(t.texture)))) {
      ok = false;
      break;
    }
  }
  device->Release();
  if (!ok) {
    if (s.output != nullptr) s.output->Release();
    if (s.color_copy != nullptr) s.color_copy->Release();
    s.output = s.color_copy = s.proxy = nullptr;
    return GiveUp("texture creation failed");
  }
  s.output_format = game_desc.Format;
  s.tex_w = s.frame_w;
  s.tex_h = s.frame_h;
  ngx_probe::Logf("nr-fwd: textures %ux%u format=%u (answer, original, proxy)", s.frame_w, s.frame_h,
                  (unsigned)s.output_format);
  return true;
}

// Model tuning, all defaults. Read at create, re-written at evaluate because the block is shared
// with the game's own DLSS, which overwrites values between frames.
inline void WriteTuning(void* p) {
  SetFloat(p, "DLSSNR.Intensity", s.intensity);
  SetUInt(p, "DLSSNR.Style", (unsigned)s.style);
  SetFloat(p, "DLSSNR.LocalStructureStrength", s.local_structure);
  SetFloat(p, "DLSSNR.LocalToneStrength", s.local_tone);
  SetFloat(p, "DLSSNR.SkinStructureStrength", s.skin_structure);
  if (s.auto_mask >= 0) SetUInt(p, "DLSSNR.UseAutoMask", (unsigned)s.auto_mask);
  if (s.ui_correction >= 0) SetUInt(p, "DLSSNR.UICorrection", (unsigned)s.ui_correction);
  if (s.mask_blend_mode >= 0) SetUInt(p, "DLSSNR.MaskBlendMode", (unsigned)s.mask_blend_mode);
  if (s.global_tone >= 0.0f) SetFloat(p, "DLSSNR.GlobalToneStrength", s.global_tone);
  if (s.scaling_ratio > 0.0f) SetFloat(p, "DLSSNR.ScalingRatio", s.scaling_ratio);
}

// Called by the overlay after the create-time model settings changed: the model reads them only
// while building the feature, so it is retired and rebuilt. The overlay runs on the present
// thread while the evaluate thread owns the feature and textures, so only a flag is set here;
// the retire itself executes at the top of the next OnDlssEvaluated, on the owning thread.
inline void ApplyModelSettings() { s.retire_pending.store(true, std::memory_order_relaxed); }

inline bool EnsureFeature(ID3D12GraphicsCommandList* cmd) {
  if (s.feature != nullptr) return true;

  if (!s.snippet_inited) {
    wchar_t snippet[MAX_PATH] = {};
    swprintf_s(snippet, L"%s\\nvngx_dlssnr.dll", s.game_dir);
    // Before init, not after: a mismatched model can crash without ever returning an error, and
    // the identity line must already be in the log when it does.
    LogModelIdentityAsync(snippet);
    ID3D12Device* device = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr)
      return GiveUp("could not reach the device for init");
    const int init_result = s.init(snippet, s.game_dir, device, s.caps);
    device->Release();
    ngx_probe::Logf("nr-fwd: snippet init => 0x%x (%s)", (unsigned)init_result,
                    ResultName(init_result));
    if (init_result != 1) return GiveUp("snippet init failed");
    s.snippet_inited = true;
  }

  unsigned fw = 0, fh = 0;
  FeatureSize(&fw, &fh);
  void* p = s.caps;
  auto create = [&](unsigned w, unsigned h, int preset) {
    SetUInt(p, "DLSSNR.Enabled", 1u);
    SetUInt(p, "DLSSNR.Width", w);
    SetUInt(p, "DLSSNR.Height", h);
    SetUInt(p, "CreationNodeMask", 1u);
    SetUInt(p, "VisibilityNodeMask", 1u);
    SetUInt(p, "DLSSNR.Hint.Render.Preset", (unsigned)preset);
    WriteTuning(p);
    int create_result = 0;
    void* f = s.create(cmd, s.caps, &create_result);
    ngx_probe::Logf("nr-fwd: CreateFeature(18) %ux%u => 0x%x (%s) handle=%p", w, h,
                    (unsigned)create_result, ResultName(create_result), f);
    return f;
  };
  const int calls = WantedCalls();
  if (calls == 2) {
    const fov::SplitLayout l = fov::SplitSize(s.frame_w, s.frame_h);
    s.feature = create(l.crop_w, l.crop_h, s.split_crop_preset >= 0 ? s.split_crop_preset : s.preset);
    s.feature2 = s.feature != nullptr ? create(l.low_w, l.low_h, s.preset) : nullptr;
    if (s.feature2 == nullptr) {
      if (s.feature != nullptr) Bury(nullptr, s.feature);
      s.feature = nullptr;
    }
  } else {
    s.feature = create(fw, fh, s.preset);
  }
  if (s.feature == nullptr) return GiveUp("feature creation failed");
  s.feature_calls = calls;
  s.feature_w = fw;
  s.feature_h = fh;
  if (fw != s.frame_w || fh != s.frame_h)
    ngx_probe::Logf("nr-fwd: foveated -- the model runs at %ux%u for a %ux%u frame (%.0f%% of the pixels)",
                    fw, fh, s.frame_w, s.frame_h, 100.0 * fw * fh / ((double)s.frame_w * s.frame_h));
  if (s.frame_w != s.out_w || s.frame_h != s.out_h)
    ngx_probe::Logf("nr-fwd: before the upscale -- the pass runs on DLSS-SR's %ux%u input, which it "
                    "then upscales to %ux%u", s.frame_w, s.frame_h, s.out_w, s.out_h);
  // Evaluate from the next frame on, so the init work this create recorded executes first.
  return false;
}

// --- the per-frame pass ---------------------------------------------------

// If the model fails after RecordPre already ran, put every state back so the frame proceeds
// with the original picture untouched.
inline void AbortAfterPre(ID3D12GraphicsCommandList* cmd, ID3D12Resource* game_out) {
  if (s.feature_w != s.frame_w || s.feature_h != s.frame_h) fov::Abort(cmd);
  nr_compose::Barrier(cmd, game_out, D3D12_RESOURCE_STATE_COPY_SOURCE,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  nr_compose::Barrier(cmd, s.color_copy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  nr_compose::Barrier(cmd, s.proxy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

// `color` is the picture the pass rewrites in place (UNORDERED_ACCESS on entry and exit, frame_w x
// frame_h): the game's DLSS output, or our copy of its input. True when the model's answer landed.
inline bool Evaluate(ID3D12GraphicsCommandList* cmd, const NVSDK_NGX_Parameter* game_params,
                     ID3D12Resource* color) {
  // The model sees the frame's encoded proxy, and the game's DLSS guides are the model's guides.
  ID3D12Resource* depth = nullptr;
  ID3D12Resource* motion = nullptr;
  game_params->Get(NVSDK_NGX_Parameter_Depth, &depth);
  game_params->Get(NVSDK_NGX_Parameter_MotionVectors, &motion);
  if (color == nullptr || depth == nullptr || motion == nullptr) return false;

  if (!CreateTextures(cmd, color)) return false;

  // Fold the latest luminance reading into the smoothed white point. The reading lags a few
  // frames, which the EMA absorbs; until the first one lands, 1.0 preserves 5b behaviour.
  const float raw_wp = nr_compose::ReadWhitePoint(s.eval_count);
  if (raw_wp > 0.0f && raw_wp == raw_wp) {
    s.wp_ema = (s.wp_ema <= 0.0f) ? raw_wp : s.wp_ema + 0.05f * (raw_wp - s.wp_ema);
  }
  float wp = (s.wp_ema > 0.0f) ? s.wp_ema : 1.0f;
  wp = (wp < 0.01f ? 0.01f : (wp > 1000.0f ? 1000.0f : wp)) * s.wp_scale;

  diag::Poll(s.eval_count, s.game_dir, ngx_probe::Log);
  diag::Begin(cmd, s.eval_count, ngx_probe::Log);

  // Copy the original aside, encode the display-referred proxy, record the measurement.
  if (!nr_compose::RecordPre(cmd, color, s.color_copy, s.proxy, s.frame_w, s.frame_h, s.eval_count,
                             wp)) {
    GiveUp("compose pipeline initialisation failed");
    return false;
  }

  diag::Mark(cmd, diag::kEncode);

  float mv_scale_x = 1.0f, mv_scale_y = 1.0f;
  game_params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &mv_scale_x);
  game_params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &mv_scale_y);
  int reset = 0;
  game_params->Get(NVSDK_NGX_Parameter_Reset, &reset);
  const bool depth_inverted =
      (s.create_flags & NVSDK_NGX_DLSS_Feature_Flags_DepthInverted) != 0;

  // Foveated: the model gets working-size copies (motion vectors already in working pixels) and
  // writes a working-size answer that Expand stretches back below. Built at the feature's size.
  const bool foveated = s.feature_w != s.frame_w || s.feature_h != s.frame_h;
  ID3D12Resource* in_color = s.proxy;
  ID3D12Resource* in_depth = depth;
  ID3D12Resource* in_motion = motion;
  ID3D12Resource* answer = s.output;
  unsigned guide_w = s.render_w, guide_h = s.render_h;
  // The mode is read once, so the expand below always matches the squeeze it follows.
  const bool split = fov::settings.split;
  if (foveated) {
    const bool squeezed = (split ? fov::SplitSqueeze : fov::Squeeze)(
        cmd, s.proxy, depth, motion, s.frame_w, s.frame_h, s.render_w, s.render_h, mv_scale_x,
        mv_scale_y, s.eval_count);
    if (!squeezed || fov::g.x.work != s.feature_w || fov::g.y.work != s.feature_h) {
      // A failed Squeeze transitioned nothing; a size that no longer matches the feature means
      // the settings moved under us -- undo the squeeze and rebuild next frame.
      if (squeezed) fov::Abort(cmd);
      nr_compose::Barrier(cmd, color, D3D12_RESOURCE_STATE_COPY_SOURCE,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      nr_compose::Barrier(cmd, s.color_copy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      nr_compose::Barrier(cmd, s.proxy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      s.retire_pending = true;
      return false;
    }
    in_color = fov::g.proxy_s;
    in_depth = fov::g.depth_s;
    in_motion = fov::g.motion_s;
    answer = fov::g.answer_s;
    guide_w = s.feature_w;
    guide_h = s.feature_h;
    mv_scale_x = mv_scale_y = 1.0f;
    if (fov::g.reset) reset = 1;
    fov::g.reset = false;
  }

  diag::Mark(cmd, diag::kSqueeze);

  void* p = s.caps;
  const uint32_t frame = s.eval_count;   // the ring index RecordPre and Squeeze used
  const uint32_t n = ++s.eval_count;
  // One model call on a region of the inputs (the guides' region scales with them).
  auto call = [&](void* feature, unsigned x, unsigned y, unsigned w, unsigned h) {
    const unsigned gx = x * guide_w / s.feature_w, gy = y * guide_h / s.feature_h;
    const unsigned gw = w * guide_w / s.feature_w, gh = h * guide_h / s.feature_h;
    SetResource(p, "DLSSNR.Color", in_color);
    SetResource(p, "DLSSNR.Depth", in_depth);
    SetResource(p, "DLSSNR.MVec", in_motion);
    SetResource(p, "DLSSNR.Output", answer);

    SetUInt(p, "DLSSNR.Enabled", 1u);
    SetUInt(p, "DLSSNR.Width", w);
    SetUInt(p, "DLSSNR.Height", h);
    SetUInt(p, "DLSSNR.DepthInverted", depth_inverted ? 1u : 0u);
    SetUInt(p, "DLSSNR.Reset", reset != 0 ? 1u : 0u);

    SetUInt(p, "DLSSNR.ColorSubrectBaseX", x);
    SetUInt(p, "DLSSNR.ColorSubrectBaseY", y);
    SetUInt(p, "DLSSNR.ColorSubrectWidth", w);
    SetUInt(p, "DLSSNR.ColorSubrectHeight", h);
    SetUInt(p, "DLSSNR.OutputSubrectBaseX", x);
    SetUInt(p, "DLSSNR.OutputSubrectBaseY", y);
    SetUInt(p, "DLSSNR.OutputSubrectWidth", w);
    SetUInt(p, "DLSSNR.OutputSubrectHeight", h);
    SetUInt(p, "DLSSNR.DepthSubrectBaseX", gx);
    SetUInt(p, "DLSSNR.DepthSubrectBaseY", gy);
    SetUInt(p, "DLSSNR.DepthSubrectWidth", gw);
    SetUInt(p, "DLSSNR.DepthSubrectHeight", gh);
    SetUInt(p, "DLSSNR.MVecSubrectBaseX", gx);
    SetUInt(p, "DLSSNR.MVecSubrectBaseY", gy);
    SetUInt(p, "DLSSNR.MVecSubrectWidth", gw);
    SetUInt(p, "DLSSNR.MVecSubrectHeight", gh);

    SetFloat(p, "DLSSNR.MVecScaleX", mv_scale_x);
    SetFloat(p, "DLSSNR.MVecScaleY", mv_scale_y);
    WriteTuning(p);
    return s.evaluate(cmd, feature, s.caps);
  };
  int result;
  if (foveated && split && s.feature_calls == 2) {
    const fov::SplitLayout l = fov::SplitSize(s.frame_w, s.frame_h);
    result = call(s.feature, 0, 0, l.crop_w, l.crop_h);
    if (result == 1) result = call(s.feature2, l.low_x, 0, l.low_w, l.low_h);
  } else {
    result = call(s.feature, 0, 0, s.feature_w, s.feature_h);
  }

  if (n <= kFullLogs || n % kHeartbeatEvery == 0) {
    ngx_probe::Logf("nr-fwd: EvaluateFeature #%u => 0x%x (%s)", n, (unsigned)result,
                    ResultName(result));
  }

  if (result != 1) {
    AbortAfterPre(cmd, color);
    if (++s.fail_streak >= kMaxFailStreak) {
      ngx_probe::Logf("nr-fwd: %u consecutive failures (last: %s) -- pass disabled",
                      s.fail_streak, ResultName(result));
      s.enabled = false;
    }
    return false;
  }
  s.fail_streak = 0;
  diag::Mark(cmd, diag::kModel);
  if (foveated && split) fov::SplitExpand(cmd, s.output, s.proxy, frame);
  else if (foveated) fov::Expand(cmd, s.output, frame);
  diag::Mark(cmd, diag::kExpand);

  // Anchor the model's answer to the original and write the blend into the game's output, which
  // its post-processing reads next.
  nr_compose::RecordPost(cmd, color, s.color_copy, s.proxy, s.output, s.frame_w, s.frame_h,
                         s.eval_count, s.transfer, s.max_ratio, s.colour_strength, wp,
                         (uint32_t)s.debug_view);
  if (foveated && !split && s.debug_view == 4) fov::DebugMap(cmd, color, frame);
  diag::Mark(cmd, diag::kResolve);
  diag::End(cmd);

  char info[384];
  std::snprintf(info, sizeof(info),
                "foveated=%d\nsplit=%d\ncrop_x=%u\ncrop_y=%u\nwork=%ux%u\nfovea_w=%.3f\nfovea_h=%.3f\nfalloff=%.3f\nedge=%.3f\n"
                "centre_x=%.1f\ncentre_y=%.1f\ncolour=%.2f\nguard=%.2f\nwhite_point=%.4f\n",
                foveated ? 1 : 0, foveated && split ? 1 : 0, fov::g.crop_x, fov::g.crop_y,
                s.feature_w, s.feature_h, fov::settings.width,
                fov::settings.height, fov::settings.falloff, fov::settings.edge,
                foveated ? fov::g.centre_x : s.frame_w * 0.5f, foveated ? fov::g.centre_y : s.frame_h * 0.5f,
                s.colour_strength, s.max_ratio, wp);
  diag::Capture(cmd, s.color_copy, color, s.eval_count, info, ngx_probe::Log);
  return true;
}

// Shared by both entry points: housekeeping, then whether the pass may run this frame.
inline bool Prepare(ID3D12GraphicsCommandList* cmd) {
  TickGraveyard();
  if (s.retire_pending.exchange(false, std::memory_order_relaxed))
    RetireFeature("model settings changed");
  if (!s.enabled) return false;
  if (s.output != nullptr && (s.tex_w != s.frame_w || s.tex_h != s.frame_h))
    RetireFeature("the pass moved to the other side of the upscale");
  {
    unsigned fw = 0, fh = 0;
    FeatureSize(&fw, &fh);
    if (s.feature != nullptr && (s.feature_w != fw || s.feature_h != fh))
      RetireFeature("foveation geometry changed");
    if (s.feature != nullptr && s.feature_calls != WantedCalls())
      RetireFeature("split calls changed");
  }
  if (!EnsureSetup(cmd)) return false;
  return EnsureFeature(cmd);
}

inline void Transition(ID3D12GraphicsCommandList* cmd, ID3D12Resource* res,
                       D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
  nr_compose::Barrier(cmd, res, from, to);
}

// Before the game's DLSS-SR evaluate, in before_upscale mode: copy its Color input aside, run the
// pass on the copy, and point the evaluate's Color at the result. The input is in
// NON_PIXEL_SHADER_RESOURCE here, as DLSS requires (the Feeder host's contract). Returns the
// original Color when it was swapped, for AfterDlss to put back; nullptr when nothing changed.
inline ID3D12Resource* BeforeDlss(ID3D12GraphicsCommandList* cmd,
                                  const NVSDK_NGX_Parameter* game_params) {
  if (s.gave_up || cmd == nullptr || game_params == nullptr) return nullptr;
  if (s.out_w == 0 || s.out_h == 0) return nullptr;  // no DLSS-SR create seen yet
  UpdateFrameSize();
  if (s.frame_w == s.out_w && s.frame_h == s.out_h) return nullptr;  // the pass runs after
  if (!Prepare(cmd)) return nullptr;

  ID3D12Resource* input = nullptr;
  game_params->Get(NVSDK_NGX_Parameter_Color, &input);
  if (input == nullptr) return nullptr;
  const D3D12_RESOURCE_DESC in_desc = input->GetDesc();
  if (in_desc.Width != s.frame_w || in_desc.Height != s.frame_h) return nullptr;

  if (s.pre_frame != nullptr && s.pre_frame->GetDesc().Format != in_desc.Format) {
    Bury(s.pre_frame, nullptr);
    s.pre_frame = nullptr;
  }
  if (s.pre_frame != nullptr &&
      (s.pre_frame->GetDesc().Width != s.frame_w || s.pre_frame->GetDesc().Height != s.frame_h)) {
    Bury(s.pre_frame, nullptr);
    s.pre_frame = nullptr;
  }
  if (s.pre_frame == nullptr) {
    ID3D12Device* device = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return nullptr;
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc = in_desc;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    desc.MipLevels = 1;
    desc.DepthOrArraySize = 1;
    const HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                       nullptr, IID_PPV_ARGS(&s.pre_frame));
    device->Release();
    if (FAILED(hr)) {
      s.pre_frame = nullptr;
      ngx_probe::Logf("nr-fwd: before-upscale frame creation failed 0x%08x", (unsigned)hr);
      return nullptr;
    }
  }

  Transition(cmd, input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
  Transition(cmd, s.pre_frame, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
  cmd->CopyResource(s.pre_frame, input);
  Transition(cmd, input, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  Transition(cmd, s.pre_frame, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

  if (!Evaluate(cmd, game_params, s.pre_frame)) return nullptr;  // pre_frame is UAV again

  Transition(cmd, s.pre_frame, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  const_cast<NVSDK_NGX_Parameter*>(game_params)->Set(NVSDK_NGX_Parameter_Color, s.pre_frame);
  NR_TEST_HOOK_AFTER_EVALUATE();
  return input;
}

// After the DLSS-SR evaluate that BeforeDlss redirected: restore the game's Color and our state.
inline void AfterDlss(ID3D12GraphicsCommandList* cmd, const NVSDK_NGX_Parameter* game_params,
                      ID3D12Resource* original) {
  const_cast<NVSDK_NGX_Parameter*>(game_params)->Set(NVSDK_NGX_Parameter_Color, original);
  Transition(cmd, s.pre_frame, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

// Called after every successful game DLSS-SR evaluate, on the game's own command list.
inline void OnDlssEvaluated(ID3D12GraphicsCommandList* cmd, const NVSDK_NGX_Parameter* game_params) {
  if (s.gave_up || cmd == nullptr || game_params == nullptr) return;
  if (s.out_w == 0 || s.out_h == 0) return;  // no DLSS-SR create seen yet
  UpdateFrameSize();
  if (s.frame_w != s.out_w || s.frame_h != s.out_h) return;  // the pass ran before the upscale
  if (!Prepare(cmd)) return;

  ID3D12Resource* color = nullptr;
  game_params->Get(NVSDK_NGX_Parameter_Output, &color);
  Evaluate(cmd, game_params, color);

  NR_TEST_HOOK_AFTER_EVALUATE();
}

}  // namespace nr_runner
