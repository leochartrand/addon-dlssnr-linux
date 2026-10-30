// Foveated NR: the model gets one atlas -- a full-resolution crop around the gaze beside the whole
// frame at low resolution -- in a model call per region (or one call on the atlas). The low-resolution
// answer sets the broad brightness and colour everywhere; the crop only adds its detail, so looking
// around cannot change the grading:
//
//   split squeeze: proxy -> atlas (crop | low-res frame), depth and motion likewise
//   model:         the atlas
//   split low:     each answer's coarse part as an offset from its input, at the low resolution
//   split blur:    a light Gaussian (H, then V forming the offsets: global; global - crop)
//   split expand:  periphery = input + its offset (the low-res answer, plus the input's finer
//                  detail); fovea = crop answer + its offset, feathered in
//
// The fovea follows a gaze point when one is streamed in over UDP. (The "split" names date from when a
// squeezed-frame mode existed beside this one.)

#pragma once

#include <winsock2.h>
#include <windows.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <d3d12.h>

#pragma comment(lib, "ws2_32")

#include "../build/shaders/fov_debug_dxil.h"
#include "../build/shaders/fov_split_squeeze_dxil.h"
#include "../build/shaders/fov_split_low_dxil.h"
#include "../build/shaders/fov_split_blur_dxil.h"
#include "../build/shaders/fov_split_expand_dxil.h"
#include "../build/shaders/fov_split_precrop_dxil.h"
#include "../build/shaders/fov_split_crop_dxil.h"

namespace fov {

constexpr uint32_t kRingSize = 8;         // frames of descriptors in flight
enum Pass : uint32_t { kDebug, kSplitSqueeze, kSplitLow, kSplitBlurH, kSplitBlurV,
                       kSplitExpand, kSplitPrecrop, kPasses };
constexpr uint32_t kSlotsPerPass = 9;     // t0-t5 (t3 the crop state), u0-u2
constexpr uint32_t kConstants = 31;       // FovConstants in fov_common.hlsli
constexpr uint32_t kSplitGap = 16;        // atlas pixels between the crop and the low-res frame

// --- settings (persisted by addon.cpp) -----------------------------------

struct Settings {
  bool enabled = false;
  float width = 0.40f;     // fovea, as a fraction of the frame width
  float height = 0.50f;    // and of its height
  float falloff = 0.10f;   // the crop's edge blend, as a fraction of the frame height
  int gaze_source = 0;     // 0 fixed centre, 1 UDP
  int gaze_port = 47620;
  float gaze_smooth_ms = 30.0f;
  float jump_reset = 0.20f;  // a move larger than this fraction of the frame resets the history
  float global_scale = 0.25f;  // split: the low-res frame's size, per axis
  float grade_blur = 1.0f;     // split: softening of the low-res offsets, sigma in low-res pixels
  int split_calls = 2;         // split: 2 = a model call per region (independent grading), 1 = one
  float tone_smoothing = 0.95f; // split: weight of the carried tone history where the scene is still
  float tone_deadband = 0.01f;  // split: input change (display units) below which the tone holds
  // Inside the crop, how much of its broad tone is the periphery's (1: one grading everywhere, the crop only adds
  // detail) or the crop's own (0: closer to the full frame on what you look at, but the grading of the crop can
  // differ from its surroundings). Measured 2026-10-29: the periphery's low-res tone is not the full frame's
  // either, and the face came out further from the full frame the bigger the periphery.
  float crop_tone = 1.0f;
  int global_every = 1;        // split: run the whole-frame pass every Nth frame (history between)
  // Split with two calls: a debug view of one pass alone (0 off, 1 what it sees, 2 its answer,
  // 3 what it changed x20). The fovea pass wins when both are set.
  int fovea_debug = 0;
  int periphery_debug = 0;
  // Split with two calls: false skips the periphery model call -- outside the crop the picture is the
  // game's own frame (with whatever lighting shaders run before the feed), inside it the crop's answer
  // with its own coarse part.
  bool periphery_pass = true;
  bool crop_pass = true;   // off: the crop's answer is its input (the model still runs, at intensity 0)
  bool crop_on_periphery = false;   // the crop's input is the frame plus the periphery's change (fov_split_precrop)
  // Split with two calls: the engine programs' fields (nr_engine), which the regions snap to for both
  // backends unless free_sizes (0 = no programs, or free sizes: the regions follow the settings above).
  // Set by the runner each frame.
  uint32_t engine_crop_w = 0, engine_crop_h = 0, engine_low_w = 0, engine_low_h = 0;
  // Split: the crop's position grid in full pixels, with a hold of 3/4 of it (0 or 1 = follow the
  // gaze pixel by pixel).
  int crop_snap = 32;
  // NGX only: the regions follow the size settings above even when engine programs exist (to try other
  // sizes); the engine then cannot run them.
  bool free_sizes = false;
};

inline Settings settings;

// --- gaze over UDP -------------------------------------------------------
//
// Two little-endian float32 (x, y), normalised 0..1 from the top-left, to 127.0.0.1:port. Wine's
// sockets are the host's, so a native Linux tracker reaches this directly.

struct Gaze {
  std::atomic<uint64_t> packed{0};       // x and y as two floats
  std::atomic<ULONGLONG> last_tick{0};   // GetTickCount64 of the last packet
  std::atomic<bool> stop{false};
  HANDLE thread = nullptr;
  SOCKET sock = INVALID_SOCKET;
  int port = 0;
};

inline Gaze gaze;

// The newest gaze point for the GPU: the crop pass (fov_split_crop) reads it from this mapped upload
// buffer when it runs, so the crop follows the eye as of execution, not as of recording.
inline std::atomic<volatile uint64_t*> gaze_gpu{nullptr};

inline DWORD WINAPI GazeThread(LPVOID) {
  char buf[64];
  while (!gaze.stop.load(std::memory_order_relaxed)) {
    const int n = recv(gaze.sock, buf, sizeof(buf), 0);
    if (n == 8) {
      float xy[2];
      std::memcpy(xy, buf, 8);
      if (xy[0] == xy[0] && xy[1] == xy[1]) {
        uint64_t v;
        std::memcpy(&v, xy, 8);
        gaze.packed.store(v, std::memory_order_relaxed);
        gaze.last_tick.store(GetTickCount64(), std::memory_order_relaxed);
        if (volatile uint64_t* m = gaze_gpu.load(std::memory_order_acquire)) *m = v;   // one aligned store
      }
    }
    // n <= 0: the 200 ms receive timeout, so the stop flag is seen promptly.
  }
  return 0;
}

inline void GazeStop() {
  if (gaze.thread == nullptr) return;
  gaze.stop = true;
  WaitForSingleObject(gaze.thread, 1000);
  CloseHandle(gaze.thread);
  gaze.thread = nullptr;
  closesocket(gaze.sock);
  gaze.sock = INVALID_SOCKET;
  WSACleanup();
}

inline void GazeStart(int port) {
  if (gaze.thread != nullptr && gaze.port == port) return;
  GazeStop();
  WSADATA wsa;
  if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return;
  gaze.sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (gaze.sock == INVALID_SOCKET) { WSACleanup(); return; }
  sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_port = htons((u_short)port);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  DWORD timeout_ms = 200;
  setsockopt(gaze.sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));
  if (bind(gaze.sock, (sockaddr*)&addr, sizeof(addr)) != 0) {
    ngx_probe::Logf("fov: gaze port %d could not be bound (%d)", port, WSAGetLastError());
    closesocket(gaze.sock);
    gaze.sock = INVALID_SOCKET;
    WSACleanup();
    return;
  }
  gaze.stop = false;
  gaze.port = port;
  gaze.thread = CreateThread(nullptr, 0, GazeThread, nullptr, 0, nullptr);
  ngx_probe::Logf("fov: listening for gaze on 127.0.0.1:%d", port);
}

// True with a fresh point (normalised) when the tracker has sent one in the last half second.
inline bool GazeLatest(float* x, float* y) {
  if (gaze.thread == nullptr) return false;
  if (GetTickCount64() - gaze.last_tick.load(std::memory_order_relaxed) > 500) return false;
  const uint64_t v = gaze.packed.load(std::memory_order_relaxed);
  float xy[2];
  std::memcpy(xy, &v, 8);
  *x = xy[0] < 0.0f ? 0.0f : (xy[0] > 1.0f ? 1.0f : xy[0]);
  *y = xy[1] < 0.0f ? 0.0f : (xy[1] > 1.0f ? 1.0f : xy[1]);
  return true;
}

// --- GPU state -----------------------------------------------------------

struct State {
  ID3D12RootSignature* root_sig = nullptr;
  ID3D12PipelineState* debug_pso = nullptr;
  ID3D12PipelineState* split_pso[5] = {};   // squeeze, low, blur, expand, precrop
  ID3D12DescriptorHeap* heap = nullptr;
  uint32_t descriptor_size = 0;
  bool failed = false;

  // Working-size textures, all UNORDERED_ACCESS between frames.
  ID3D12Resource* proxy_s = nullptr;
  ID3D12Resource* depth_s = nullptr;
  ID3D12Resource* motion_s = nullptr;
  ID3D12Resource* answer_s = nullptr;
  uint32_t tex_w = 0, tex_h = 0;

  // Split mode: the low-resolution pictures (UNORDERED_ACCESS between frames) and the crop.
  ID3D12Resource* low[10] = {};  // global, input, crop; their H blur; the two offsets; tone history x2
  uint32_t low_w = 0, low_h = 0;
  uint32_t hist_flip = 0;        // which of low[8], low[9] holds last frame's tone
  bool hist_valid = false;
  uint32_t render_w = 0, render_h = 0;   // the motion vectors' size and scale, for the history
  float mv_x = 1.0f, mv_y = 1.0f;
  uint32_t crop_x = 0, crop_y = 0;   // the CPU's estimate (logs, dumps); the GPU's is crop_state
  int prev_crop_x = -1, prev_crop_y = -1;
  // Split: the crop pass -- its root signature (root SRV gaze, root UAV state, constants), the mapped
  // gaze buffer, and the crop state (4 floats: origin, previous origin), which stays
  // NON_PIXEL_SHADER_RESOURCE between frames.
  ID3D12RootSignature* crop_root = nullptr;
  ID3D12PipelineState* crop_pso = nullptr;
  ID3D12Resource* gaze_buf = nullptr;
  ID3D12Resource* crop_state = nullptr;

  // The frame and atlas sizes this layout was built for.
  struct { uint32_t full = 0, work = 0; } x, y;
  float centre_x = -1.0f, centre_y = -1.0f;   // smoothed, in full pixels
  // Replaced textures and LUT buffers may still be read by frames in flight (nr_runner's issue #1).
  struct Grave { ID3D12Resource* r; int frames; };
  std::vector<Grave> graves;
  ULONGLONG last_tick = 0;
  bool reset = false;   // the fovea jumped: tell the model to drop its history
};

inline State g;

inline void Bury(ID3D12Resource*& r) {
  if (r != nullptr) g.graves.push_back({r, 240});
  r = nullptr;
}

inline void TickGraves() {
  for (auto& gr : g.graves)
    if (gr.r != nullptr && --gr.frames <= 0) { gr.r->Release(); gr.r = nullptr; }
  while (!g.graves.empty() && g.graves.back().r == nullptr) g.graves.pop_back();
}

inline void Release() {
  GazeStop();
  for (ID3D12Resource** t : {&g.proxy_s, &g.depth_s, &g.motion_s, &g.answer_s})
    if (*t != nullptr) (*t)->Release();
  for (auto* r : g.low)
    if (r != nullptr) r->Release();
  for (auto* p : g.split_pso)
    if (p != nullptr) p->Release();
  gaze_gpu.store(nullptr);
  for (IUnknown* u : {(IUnknown*)g.crop_pso, (IUnknown*)g.crop_root, (IUnknown*)g.gaze_buf, (IUnknown*)g.crop_state})
    if (u != nullptr) u->Release();
  for (auto& gr : g.graves)
    if (gr.r != nullptr) gr.r->Release();
  if (g.heap != nullptr) g.heap->Release();
  if (g.debug_pso != nullptr) g.debug_pso->Release();
  if (g.root_sig != nullptr) g.root_sig->Release();
  g = {};
}

// Split mode's layout for a frame size: the crop, the low-res frame and the atlas holding both.
struct SplitLayout {
  uint32_t crop_w, crop_h, low_w, low_h, low_x, work_w, work_h;
};

inline SplitLayout SplitSize(uint32_t full_w, uint32_t full_h) {
  const auto& st = settings;
  auto len = [](uint32_t full, float f) {
    uint32_t v = (uint32_t)std::lround(f * full) & ~1u;
    return v < 16 ? 16 : (v > full ? full : v);
  };
  SplitLayout l;
  l.crop_w = len(full_w, st.width);
  l.crop_h = len(full_h, st.height);
  l.low_w = len(full_w, st.global_scale);
  l.low_h = len(full_h, st.global_scale);
  if (st.engine_crop_w != 0 && st.engine_crop_w <= full_w && st.engine_crop_h <= full_h &&
      st.engine_low_w <= full_w && st.engine_low_h <= full_h) {
    // The engine runs each region at its program's field, which is its own size (no padding). A
    // low-res aspect slightly off the frame's is undone by the expand, which maps it back over it.
    l.crop_w = st.engine_crop_w;
    l.crop_h = st.engine_crop_h;
    l.low_w = st.engine_low_w;
    l.low_h = st.engine_low_h;
  }
  l.low_x = l.crop_w + kSplitGap;
  l.work_w = l.low_x + l.low_w;
  l.work_h = l.crop_h > l.low_h ? l.crop_h : l.low_h;
  return l;
}

// The working size the current settings give, for the NR feature and its textures.
inline void WorkSize(uint32_t full_w, uint32_t full_h, uint32_t* w, uint32_t* h) {
  const SplitLayout l = SplitSize(full_w, full_h);
  *w = l.work_w;
  *h = l.work_h;
}

// Root signature: [0] SRV table t0-t5, [1] UAV table u0-u2, [2] constants b0, static linear s0.
inline bool Init(ID3D12Device* device) {
  if (g.root_sig != nullptr) return true;
  if (g.failed) return false;

  D3D12_DESCRIPTOR_RANGE srv = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 6, 0, 0, 0};
  D3D12_DESCRIPTOR_RANGE uav = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 3, 0, 0, 0};
  D3D12_ROOT_PARAMETER params[3] = {};
  params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  params[0].DescriptorTable = {1, &srv};
  params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  params[1].DescriptorTable = {1, &uav};
  params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  params[2].Constants = {0, 0, kConstants};
  D3D12_STATIC_SAMPLER_DESC sampler = {};
  sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
  sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
  sampler.MaxLOD = D3D12_FLOAT32_MAX;
  sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

  D3D12_ROOT_SIGNATURE_DESC desc = {3, params, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE};
  ID3DBlob* blob = nullptr;
  ID3DBlob* error = nullptr;
  if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error))) {
    if (error != nullptr) error->Release();
    g.failed = true;
    return false;
  }
  const HRESULT hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                                 IID_PPV_ARGS(&g.root_sig));
  blob->Release();
  if (FAILED(hr)) { g.failed = true; return false; }

  D3D12_COMPUTE_PIPELINE_STATE_DESC pso = {};
  pso.pRootSignature = g.root_sig;
  pso.CS = {fov_debug_dxil, sizeof(fov_debug_dxil)};
  if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&g.debug_pso)))) { g.failed = true; return false; }
  const D3D12_SHADER_BYTECODE split[5] = {{fov_split_squeeze_dxil, sizeof(fov_split_squeeze_dxil)},
                                          {fov_split_low_dxil, sizeof(fov_split_low_dxil)},
                                          {fov_split_blur_dxil, sizeof(fov_split_blur_dxil)},
                                          {fov_split_expand_dxil, sizeof(fov_split_expand_dxil)},
                                          {fov_split_precrop_dxil, sizeof(fov_split_precrop_dxil)}};
  for (int i = 0; i < 5; ++i) {
    pso.CS = split[i];
    if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&g.split_pso[i])))) { g.failed = true; return false; }
  }

  D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kRingSize * kPasses * kSlotsPerPass,
                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
  if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g.heap)))) { g.failed = true; return false; }
  g.descriptor_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  return true;
}

inline bool EnsureTextures(ID3D12Device* device, uint32_t w, uint32_t h) {
  if (g.proxy_s != nullptr && g.tex_w == w && g.tex_h == h) return true;
  for (ID3D12Resource** t : {&g.proxy_s, &g.depth_s, &g.motion_s, &g.answer_s}) Bury(*t);
  D3D12_HEAP_PROPERTIES heap = {D3D12_HEAP_TYPE_DEFAULT};
  D3D12_RESOURCE_DESC desc = {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  desc.Width = w;
  desc.Height = h;
  desc.DepthOrArraySize = desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  struct { ID3D12Resource** t; DXGI_FORMAT f; } list[4] = {
      {&g.proxy_s, DXGI_FORMAT_R16G16B16A16_FLOAT},
      {&g.depth_s, DXGI_FORMAT_R32_FLOAT},
      {&g.motion_s, DXGI_FORMAT_R16G16_FLOAT},
      {&g.answer_s, DXGI_FORMAT_R16G16B16A16_FLOAT},
  };
  for (auto& e : list) {
    desc.Format = e.f;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                               IID_PPV_ARGS(e.t))))
      return false;
  }
  g.tex_w = w;
  g.tex_h = h;
  ngx_probe::Logf("fov: working textures %ux%u", w, h);
  return true;
}

// A readable view format for whatever the game handed over (depth often arrives typeless).
inline DXGI_FORMAT ReadableFormat(DXGI_FORMAT f) {
  switch (f) {
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT: return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    default: return f;
  }
}

inline D3D12_CPU_DESCRIPTOR_HANDLE Cpu(uint32_t i) {
  D3D12_CPU_DESCRIPTOR_HANDLE h = g.heap->GetCPUDescriptorHandleForHeapStart();
  h.ptr += (SIZE_T)i * g.descriptor_size;
  return h;
}

inline D3D12_GPU_DESCRIPTOR_HANDLE Gpu(uint32_t i) {
  D3D12_GPU_DESCRIPTOR_HANDLE h = g.heap->GetGPUDescriptorHandleForHeapStart();
  h.ptr += (UINT64)i * g.descriptor_size;
  return h;
}

// A null resource with a full description is a valid "nothing here" descriptor.
inline void Srv(ID3D12Device* d, uint32_t i, ID3D12Resource* r, DXGI_FORMAT fallback) {
  D3D12_SHADER_RESOURCE_VIEW_DESC v = {};
  v.Format = r != nullptr ? ReadableFormat(r->GetDesc().Format) : fallback;
  v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  v.Texture2D.MipLevels = 1;
  d->CreateShaderResourceView(r, &v, Cpu(i));
}

inline void Uav(ID3D12Device* d, uint32_t i, ID3D12Resource* r, DXGI_FORMAT fallback) {
  D3D12_UNORDERED_ACCESS_VIEW_DESC v = {};
  v.Format = r != nullptr ? r->GetDesc().Format : fallback;
  v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  d->CreateUnorderedAccessView(r, nullptr, &v, Cpu(i));
}

inline void Barrier(ID3D12GraphicsCommandList* cmd, ID3D12Resource* r, D3D12_RESOURCE_STATES a,
                    D3D12_RESOURCE_STATES b) {
  D3D12_RESOURCE_BARRIER x = {};
  x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  x.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, a, b};
  cmd->ResourceBarrier(1, &x);
}

// One pass's descriptors in its own slots of this frame's block -- t0-t2, t4, t5 from `srv`, t3 the crop
// state, u0-u2 from `uav` -- then its pipeline, tables and constants.
struct View {
  ID3D12Resource* r;
  DXGI_FORMAT fallback;   // the view format when r is null
};

// The crop state where the passes read it (t3: CropOrigin() in fov_common.hlsli); a null view before it exists.
inline void CropSrv(ID3D12Device* d, uint32_t i) {
  D3D12_SHADER_RESOURCE_VIEW_DESC v = {};
  v.Format = DXGI_FORMAT_UNKNOWN;
  v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  v.Buffer.NumElements = 4;
  v.Buffer.StructureByteStride = 4;
  d->CreateShaderResourceView(g.crop_state, &v, Cpu(i));
}

inline void Bind(ID3D12Device* device, ID3D12GraphicsCommandList* cmd, ID3D12PipelineState* pso,
                 uint32_t frame, Pass pass, const View (&srv)[5], const View (&uav)[3],
                 const uint32_t* constants) {
  const uint32_t base = ((frame % kRingSize) * kPasses + pass) * kSlotsPerPass;
  const uint32_t reg[5] = {0, 1, 2, 4, 5};
  for (int i = 0; i < 5; ++i) Srv(device, base + reg[i], srv[i].r, srv[i].fallback);
  CropSrv(device, base + 3);
  for (int i = 0; i < 3; ++i) Uav(device, base + 6 + i, uav[i].r, uav[i].fallback);
  cmd->SetDescriptorHeaps(1, &g.heap);
  cmd->SetComputeRootSignature(g.root_sig);
  cmd->SetPipelineState(pso);
  cmd->SetComputeRootDescriptorTable(0, Gpu(base));
  cmd->SetComputeRootDescriptorTable(1, Gpu(base + 6));
  cmd->SetComputeRoot32BitConstants(2, kConstants, constants, 0);
}

constexpr DXGI_FORMAT kRgba = DXGI_FORMAT_R16G16B16A16_FLOAT;
constexpr DXGI_FORMAT kR32 = DXGI_FORMAT_R32_FLOAT;
constexpr DXGI_FORMAT kRg16 = DXGI_FORMAT_R16G16_FLOAT;

// Advance the fovea centre for this frame: the gaze point when one is streaming, the frame centre
// otherwise, smoothed over gaze_smooth_ms. Returns the centre in full pixels.
inline void UpdateCentre(uint32_t full_w, uint32_t full_h) {
  const auto& st = settings;
  float tx = 0.5f, ty = 0.5f;
  if (st.gaze_source == 1) {
    GazeStart(st.gaze_port);
    float gx, gy;
    if (GazeLatest(&gx, &gy)) { tx = gx; ty = gy; }
  } else {
    GazeStop();
  }
  tx *= (float)full_w;
  ty *= (float)full_h;

  const ULONGLONG now = GetTickCount64();
  if (g.centre_x < 0.0f) {
    g.centre_x = tx;
    g.centre_y = ty;
  } else {
    const float jump = std::fabs(tx - g.centre_x) / full_w + std::fabs(ty - g.centre_y) / full_h;
    if (jump > st.jump_reset) {
      g.centre_x = tx;   // a saccade: land there now and start the history over
      g.centre_y = ty;
      g.reset = true;
    } else {
      const float dt = (float)(now - g.last_tick);
      const float a = st.gaze_smooth_ms > 0.0f ? 1.0f - std::exp(-dt / st.gaze_smooth_ms) : 1.0f;
      g.centre_x += a * (tx - g.centre_x);
      g.centre_y += a * (ty - g.centre_y);
    }
  }
  g.last_tick = now;
}

inline void SplitConstants(uint32_t* c, const SplitLayout& l, uint32_t full_w, uint32_t full_h,
                           uint32_t render_w, uint32_t render_h, float mv_x, float mv_y);

// Debug view 4: tint the finished frame by where the model saw full resolution (the crop) and mark the
// fovea centre and the raw gaze point. The frame is UNORDERED_ACCESS after the resolve.
inline void DebugMap(ID3D12GraphicsCommandList* cmd, ID3D12Resource* frame_res, uint32_t frame) {
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return;
  const uint32_t full_w = g.x.full, full_h = g.y.full;
  float gx = -1.0f, gy = -1.0f;
  if (GazeLatest(&gx, &gy)) { gx *= (float)full_w; gy *= (float)full_h; }
  else gx = gy = -1.0f;
  const float floats[4] = {g.centre_x, g.centre_y, gx, gy};
  uint32_t c[kConstants] = {};
  SplitConstants(c, SplitSize(full_w, full_h), full_w, full_h, 1, 1, 0.0f, 0.0f);
  std::memcpy(&c[11], floats, sizeof(floats));

  D3D12_RESOURCE_BARRIER uav = {};
  uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  uav.UAV.pResource = frame_res;
  cmd->ResourceBarrier(1, &uav);   // after the resolve's writes
  Bind(device, cmd, g.debug_pso, frame, kDebug,
       {{nullptr, kRgba}, {nullptr, kR32}, {nullptr, kRg16}, {nullptr, kRgba}, {nullptr, kRgba}},
       {{frame_res, DXGI_FORMAT_UNKNOWN}, {nullptr, kR32}, {nullptr, kRg16}}, c);
  cmd->Dispatch((full_w + 7) / 8, (full_h + 7) / 8, 1);
  device->Release();
}

// --- split mode ----------------------------------------------------------

inline bool EnsureLowTextures(ID3D12Device* device, uint32_t w, uint32_t h) {
  if (g.low[0] != nullptr && g.low_w == w && g.low_h == h) return true;
  for (auto*& r : g.low) Bury(r);
  g.hist_valid = false;
  D3D12_HEAP_PROPERTIES heap = {D3D12_HEAP_TYPE_DEFAULT};
  D3D12_RESOURCE_DESC desc = {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  desc.Width = w;
  desc.Height = h;
  desc.DepthOrArraySize = desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Format = kRgba;
  desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  for (auto*& r : g.low)
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                               IID_PPV_ARGS(&r))))
      return false;
  g.low_w = w;
  g.low_h = h;
  return true;
}

// The split constants: the usual header, then the crop, the low-res frame and the blur.
inline void SplitConstants(uint32_t* c, const SplitLayout& l, uint32_t full_w, uint32_t full_h,
                           uint32_t render_w, uint32_t render_h, float mv_x, float mv_y) {
  const float feather_cap = 0.5f * (float)(l.crop_w < l.crop_h ? l.crop_w : l.crop_h);
  float feather = settings.falloff * (float)full_h;
  feather = feather > feather_cap ? feather_cap : feather;
  c[0] = full_w; c[1] = full_h; c[2] = l.work_w; c[3] = l.work_h;
  c[4] = render_w; c[5] = render_h;
  std::memcpy(&c[6], &mv_x, 4);
  std::memcpy(&c[7], &mv_y, 4);
  c[15] = g.crop_x; c[16] = g.crop_y;
  const int prev[2] = {g.prev_crop_x, g.prev_crop_y};
  std::memcpy(&c[17], prev, 8);
  c[19] = l.crop_w; c[20] = l.crop_h;
  c[21] = l.low_x; c[22] = l.low_w; c[23] = l.low_h;
  std::memcpy(&c[24], &feather, 4);
  c[25] = 0;
  std::memcpy(&c[26], &settings.grade_blur, 4);
  std::memcpy(&c[30], &settings.crop_tone, 4);
}

// The crop pass's objects: its root signature ([0] root SRV t0 gaze, [1] root UAV u0 state,
// [2] constants), pipeline, the mapped gaze buffer and the crop state.
inline bool EnsureCrop(ID3D12Device* device) {
  if (g.crop_pso != nullptr) return true;
  D3D12_ROOT_PARAMETER params[3] = {};
  params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
  params[0].Descriptor = {0, 0};
  params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
  params[1].Descriptor = {0, 0};
  params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  params[2].Constants = {0, 0, 9};
  D3D12_ROOT_SIGNATURE_DESC desc = {3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
  ID3DBlob* blob = nullptr;
  ID3DBlob* error = nullptr;
  if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error))) {
    if (error != nullptr) error->Release();
    return false;
  }
  const HRESULT hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                                 IID_PPV_ARGS(&g.crop_root));
  blob->Release();
  if (FAILED(hr)) return false;
  D3D12_COMPUTE_PIPELINE_STATE_DESC pso = {};
  pso.pRootSignature = g.crop_root;
  pso.CS = {fov_split_crop_dxil, sizeof(fov_split_crop_dxil)};
  if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&g.crop_pso)))) return false;

  D3D12_RESOURCE_DESC b = {};
  b.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  b.Width = 256;
  b.Height = b.DepthOrArraySize = b.MipLevels = 1;
  b.SampleDesc.Count = 1;
  b.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  D3D12_HEAP_PROPERTIES up = {D3D12_HEAP_TYPE_UPLOAD}, def = {D3D12_HEAP_TYPE_DEFAULT};
  if (FAILED(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &b, D3D12_RESOURCE_STATE_GENERIC_READ,
                                             nullptr, IID_PPV_ARGS(&g.gaze_buf))))
    return false;
  b.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  if (FAILED(device->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &b,
                                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
                                             IID_PPV_ARGS(&g.crop_state))))
    return false;
  const D3D12_RANGE none = {0, 0};
  void* m = nullptr;
  if (FAILED(g.gaze_buf->Map(0, &none, &m))) return false;
  const uint64_t v = gaze.packed.load(std::memory_order_relaxed);
  *static_cast<volatile uint64_t*>(m) = v;
  gaze_gpu.store(static_cast<volatile uint64_t*>(m), std::memory_order_release);
  return true;
}

// Record the crop pass: the crop state from the newest gaze when it runs (NON_PIXEL_SHADER_RESOURCE
// before and after). `reset`: no hold and no previous origin (first frame, a new layout).
inline void CropPass(ID3D12GraphicsCommandList* cmd, const SplitLayout& l, uint32_t full_w, uint32_t full_h,
                     bool reset) {
  float gx, gy;
  const bool use_gaze = settings.gaze_source == 1 && GazeLatest(&gx, &gy);
  uint32_t c[9] = {full_w, full_h, l.crop_w, l.crop_h, (uint32_t)settings.crop_snap, reset ? 1u : 0u,
                   use_gaze ? 1u : 0u, 0, 0};
  std::memcpy(&c[7], &g.centre_x, 4);
  std::memcpy(&c[8], &g.centre_y, 4);
  Barrier(cmd, g.crop_state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  cmd->SetComputeRootSignature(g.crop_root);
  cmd->SetPipelineState(g.crop_pso);
  cmd->SetComputeRootShaderResourceView(0, g.gaze_buf->GetGPUVirtualAddress());
  cmd->SetComputeRootUnorderedAccessView(1, g.crop_state->GetGPUVirtualAddress());
  cmd->SetComputeRoot32BitConstants(2, 9, c, 0);
  cmd->Dispatch(1, 1, 1);
  Barrier(cmd, g.crop_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

// Place the crop at the gaze and fill the atlas. Leaves proxy_s, depth_s and motion_s
// NON_PIXEL_SHADER_RESOURCE for the model (g.x.work/g.y.work = the atlas); expects proxy in that state
// already and the game's depth and motion in a shader-readable state (as DLSS requires of its inputs).
inline bool SplitSqueeze(ID3D12GraphicsCommandList* cmd, ID3D12Resource* proxy, ID3D12Resource* depth,
                         ID3D12Resource* motion, uint32_t full_w, uint32_t full_h, uint32_t render_w,
                         uint32_t render_h, float mv_scale_x, float mv_scale_y, uint32_t frame) {
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return false;
  const SplitLayout l = SplitSize(full_w, full_h);

  TickGraves();
  UpdateCentre(full_w, full_h);
  // The crop's motion vectors carry its move, so a jump needs no reset: what enters the crop is
  // simply new to it.
  g.reset = false;
  // The crop moves on a grid, and only once the gaze has left the current position by more than
  // the hold distance: a crop that follows the tracker's jitter pixel by pixel shifts the model's
  // windows over the scene every frame, and its answer shimmers with them.
  const int snap = settings.crop_snap;
  auto origin = [snap](float centre, uint32_t size, uint32_t full, int prev) {
    long o = std::lround(centre - 0.5f * (float)size);
    if (snap > 1) {
      if (prev >= 0 && std::labs(o - prev) <= (long)snap * 3 / 4) o = prev;
      else o = (o + snap / 2) / snap * snap;
    }
    return (uint32_t)(o < 0 ? 0 : (o > (long)(full - size) ? (long)(full - size) : o));
  };
  const bool same_layout = g.x.full == full_w && g.y.full == full_h && g.x.work == l.work_w && g.y.work == l.work_h;
  g.crop_x = origin(g.centre_x, l.crop_w, full_w, same_layout ? g.prev_crop_x : -1);
  g.crop_y = origin(g.centre_y, l.crop_h, full_h, same_layout ? g.prev_crop_y : -1);
  const bool layout_reset = g.prev_crop_x < 0 || g.x.full != full_w || g.y.full != full_h ||
                            g.x.work != l.work_w || g.y.work != l.work_h;
  if (layout_reset) {
    g.prev_crop_x = (int)g.crop_x;   // first frame, or a new layout: start the history over
    g.prev_crop_y = (int)g.crop_y;
    g.reset = true;
    g.hist_valid = false;
  }
  g.render_w = render_w;
  g.render_h = render_h;
  g.mv_x = mv_scale_x;
  g.mv_y = mv_scale_y;
  g.x.full = full_w;
  g.y.full = full_h;
  g.x.work = l.work_w;
  g.y.work = l.work_h;

  if (!Init(device) || !EnsureTextures(device, l.work_w, l.work_h) ||
      !EnsureLowTextures(device, l.low_w, l.low_h)) {
    device->Release();
    return false;
  }

  if (!EnsureCrop(device)) {
    device->Release();
    return false;
  }
  CropPass(cmd, l, full_w, full_h, layout_reset);

  uint32_t c[kConstants] = {};
  SplitConstants(c, l, full_w, full_h, render_w, render_h, mv_scale_x, mv_scale_y);
  Bind(device, cmd, g.split_pso[0], frame, kSplitSqueeze,
       {{proxy, kRgba}, {depth, kR32}, {motion, kRg16}, {nullptr, kRgba}, {nullptr, kRgba}},
       {{g.proxy_s, kRgba}, {g.depth_s, kR32}, {g.motion_s, kRg16}}, c);
  cmd->Dispatch((l.work_w + 7) / 8, (l.work_h + 7) / 8, 1);

  for (ID3D12Resource* t : {g.proxy_s, g.depth_s, g.motion_s})
    Barrier(cmd, t, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  g.prev_crop_x = (int)g.crop_x;   // for the next frame's motion vectors
  g.prev_crop_y = (int)g.crop_y;
  device->Release();
  return true;
}

// Between the periphery's model call and the crop's: the crop's input in the atlas (`atlas`, NON_PIXEL_SHADER_RESOURCE,
// left so) becomes the frame plus the periphery's change, read from its answer (`answer`, UNORDERED_ACCESS, left so).
inline void SplitPrecrop(ID3D12GraphicsCommandList* cmd, ID3D12Resource* atlas, ID3D12Resource* answer,
                         uint32_t frame) {
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return;
  const uint32_t full_w = g.x.full, full_h = g.y.full;
  const SplitLayout l = SplitSize(full_w, full_h);
  constexpr auto kUav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  constexpr auto kSrv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  uint32_t c[kConstants] = {};
  SplitConstants(c, l, full_w, full_h, g.render_w, g.render_h, g.mv_x, g.mv_y);
  D3D12_RESOURCE_BARRIER written = {};
  written.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  written.UAV.pResource = answer;
  cmd->ResourceBarrier(1, &written);
  Barrier(cmd, atlas, kSrv, kUav);
  Bind(device, cmd, g.split_pso[4], frame, kSplitPrecrop,
       {{nullptr, kRgba}, {nullptr, kRgba}, {nullptr, kRgba}, {nullptr, kRgba}, {nullptr, kRgba}},
       {{atlas, kRgba}, {answer, kRgba}, {nullptr, kRg16}}, c);
  cmd->Dispatch((l.crop_w + 7) / 8, (l.crop_h + 7) / 8, 1);
  Barrier(cmd, atlas, kUav, kSrv);
  device->Release();
}

// Whether the whole-frame pass must run this frame: there is no tone history to carry yet.
inline bool NeedsGlobal() { return !g.hist_valid; }

// Split mode's Expand: the tone transfer and the blend into the full-size answer (UNORDERED_ACCESS,
// as the resolve expects it). `proxy` is the model's full-size input, NON_PIXEL_SHADER_RESOURCE;
// `motion` the game's motion vectors; `global_fresh` whether the whole-frame pass ran this frame.
// The expand shader's pass debug code: 0 off, 1-3 the fovea pass's view, 4-6 the periphery's.
inline uint32_t PassDebug() {
  if (settings.fovea_debug > 0) return (uint32_t)settings.fovea_debug;
  if (settings.periphery_debug > 0) return 3u + (uint32_t)settings.periphery_debug;
  return 0u;
}

inline void SplitExpand(ID3D12GraphicsCommandList* cmd, ID3D12Resource* answer, ID3D12Resource* proxy,
                        ID3D12Resource* motion, uint32_t frame, bool global_fresh) {
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return;
  const uint32_t full_w = g.x.full, full_h = g.y.full;
  const SplitLayout l = SplitSize(full_w, full_h);
  constexpr auto kUav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  constexpr auto kSrv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

  // SplitSqueeze already advanced prev_crop to this frame's crop; the passes below only need
  // the current one.
  uint32_t c[kConstants] = {};
  SplitConstants(c, l, full_w, full_h, g.render_w, g.render_h, g.mv_x, g.mv_y);
  const uint32_t gx = (l.low_w + 7) / 8, gy = (l.low_h + 7) / 8;
  ID3D12Resource* const* low = g.low;
  ID3D12Resource* const hist_prev = low[8 + g.hist_flip];
  ID3D12Resource* const hist_next = low[9 - g.hist_flip];
  float smoothing = settings.tone_smoothing;
  smoothing = smoothing < 0.0f ? 0.0f : (smoothing > 0.95f ? 0.95f : smoothing);
  const float alpha = !g.hist_valid ? 1.0f : (global_fresh ? 1.0f - smoothing : 0.0f);
  std::memcpy(&c[27], &alpha, 4);
  std::memcpy(&c[28], &settings.tone_deadband, 4);

  Barrier(cmd, g.answer_s, kUav, kSrv);
  if (!settings.periphery_pass) {
    // No whole-frame answer: no offsets. Null views read as zero, so the expand passes the input
    // through outside the crop and keeps the crop's own coarse part inside it.
    c[29] = PassDebug();
    Bind(device, cmd, g.split_pso[3], frame, kSplitExpand,
         {{g.answer_s, kRgba}, {proxy, kRgba}, {nullptr, kRgba}, {nullptr, kRgba}, {g.proxy_s, kRgba}},
         {{answer, kRgba}, {nullptr, kR32}, {nullptr, kRg16}}, c);
    cmd->Dispatch((full_w + 7) / 8, (full_h + 7) / 8, 1);
    Barrier(cmd, g.answer_s, kSrv, kUav);
    g.hist_valid = false;   // the tone history is stale once the pass comes back
    for (ID3D12Resource* t : {g.proxy_s, g.depth_s, g.motion_s}) Barrier(cmd, t, kSrv, kUav);
    device->Release();
    return;
  }
  Bind(device, cmd, g.split_pso[1], frame, kSplitLow,
       {{g.answer_s, kRgba}, {proxy, kRgba}, {nullptr, kRgba}, {nullptr, kRgba}, {nullptr, kRgba}},
       {{low[0], kRgba}, {low[1], kRgba}, {low[2], kRgba}}, c);
  cmd->Dispatch(gx, gy, 1);

  for (int i = 0; i < 3; ++i) Barrier(cmd, low[i], kUav, kSrv);
  c[25] = 0;
  Bind(device, cmd, g.split_pso[2], frame, kSplitBlurH,
       {{low[0], kRgba}, {low[1], kRgba}, {low[2], kRgba}, {nullptr, kRgba}, {nullptr, kRgba}},
       {{low[3], kRgba}, {low[4], kRgba}, {low[5], kRgba}}, c);
  cmd->Dispatch(gx, gy, 1);

  for (int i = 3; i < 6; ++i) Barrier(cmd, low[i], kUav, kSrv);
  Barrier(cmd, hist_prev, kUav, kSrv);
  c[25] = 1;
  Bind(device, cmd, g.split_pso[2], frame, kSplitBlurV,
       {{low[3], kRgba}, {low[4], kRgba}, {low[5], kRgba}, {hist_prev, kRgba}, {motion, kRg16}},
       {{low[6], kRgba}, {low[7], kRgba}, {hist_next, kRgba}}, c);
  cmd->Dispatch(gx, gy, 1);

  for (int i = 6; i < 8; ++i) Barrier(cmd, low[i], kUav, kSrv);
  c[29] = PassDebug();
  Bind(device, cmd, g.split_pso[3], frame, kSplitExpand,
       {{g.answer_s, kRgba}, {proxy, kRgba}, {low[6], kRgba}, {low[7], kRgba}, {g.proxy_s, kRgba}},
       {{answer, kRgba}, {nullptr, kR32}, {nullptr, kRg16}}, c);
  cmd->Dispatch((full_w + 7) / 8, (full_h + 7) / 8, 1);

  for (int i = 0; i < 8; ++i) Barrier(cmd, low[i], kSrv, kUav);
  Barrier(cmd, hist_prev, kSrv, kUav);
  Barrier(cmd, g.answer_s, kSrv, kUav);
  g.hist_flip ^= 1u;
  g.hist_valid = true;
  for (ID3D12Resource* t : {g.proxy_s, g.depth_s, g.motion_s}) Barrier(cmd, t, kSrv, kUav);
  device->Release();
}

// Undo SplitSqueeze's transitions when the model fails after it ran.
inline void Abort(ID3D12GraphicsCommandList* cmd) {
  for (ID3D12Resource* t : {g.proxy_s, g.depth_s, g.motion_s})
    Barrier(cmd, t, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

}  // namespace fov
