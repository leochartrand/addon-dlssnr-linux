// Foveated NR: run the model on a squeezed frame -- full resolution inside a fovea, smoothly
// lower beyond it -- and stretch its answer back before the resolve. The model's cost follows its
// pixel count, so a 4K frame squeezed to ~70% of its pixels costs ~70% as much.
//
//   squeeze: proxy, depth, motion (full / render res) -> working-size copies   (fov_squeeze)
//   model:   Color/Depth/MVec = the copies, Output = answer_s                  (working size)
//   expand:  answer_s -> the usual full-size answer, then the normal resolve   (fov_expand)
//
// The mapping is separable: each axis has a density that is 1 across the fovea and falls with a
// smoothstep to the edge scale over the falloff, and the working coordinate is its running sum
// (the FoveatedDLSS5 geometry). The fovea follows a gaze point when one is streamed in over UDP;
// motion vectors cross the previous and current mappings so the model's history survives the
// move, and only a jump resets it.

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

#include "../build/shaders/fov_squeeze_dxil.h"
#include "../build/shaders/fov_expand_dxil.h"
#include "../build/shaders/fov_debug_dxil.h"

namespace fov {

constexpr uint32_t kRingSize = 8;         // frames of descriptors and LUT uploads in flight
constexpr uint32_t kSlotsPerFrame = 21;   // squeeze, expand, debug: 4 SRV + 3 UAV each
constexpr uint32_t kConstants = 15;       // FovConstants in fov_common.hlsli

// --- settings (persisted by addon.cpp) -----------------------------------

struct Settings {
  bool enabled = false;
  float width = 0.40f;     // fovea, as a fraction of the frame width
  float height = 0.50f;    // and of its height
  float falloff = 0.10f;   // blend length on each side, as a fraction of the frame height
  float edge = 0.50f;      // resolution beyond the falloff, relative to the fovea, per axis
  int gaze_source = 0;     // 0 fixed centre, 1 UDP
  int gaze_port = 47620;
  float gaze_smooth_ms = 30.0f;
  float jump_reset = 0.20f;  // a move larger than this fraction of the frame resets the history
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

// --- the mapping ---------------------------------------------------------

struct Axis {
  uint32_t full = 0, work = 0;
  std::vector<float> fwd, inv;  // per full pixel centre / per working pixel centre
};

// Density across one axis: 1 within half_c of the centre, smoothstep down to edge over falloff.
inline float Density(float x, float centre, float half_c, float falloff, float edge) {
  const float d = std::fabs(x - centre) - half_c;
  if (d <= 0.0f) return 1.0f;
  float t = falloff > 0.0f ? d / falloff : 1.0f;
  t = t > 1.0f ? 1.0f : t;
  return 1.0f - (1.0f - edge) * t * t * (3.0f - 2.0f * t);
}

// The working length is fixed by the geometry alone (the centre is clamped so the whole fovea
// and falloff stay on screen), so the model's feature never has to be rebuilt as the eye moves.
inline uint32_t WorkLength(uint32_t full, float c, float f, float e) {
  if (c + 2.0f * f >= (float)full) return full;
  const float len = c + f * (1.0f + e) + e * ((float)full - c - 2.0f * f);
  const uint32_t w = (uint32_t)std::lround(len);
  return w < 16 ? 16 : (w > full ? full : w);
}

inline void BuildAxis(Axis& a, uint32_t full, float centre, float c, float f, float e) {
  a.full = full;
  a.work = WorkLength(full, c, f, e);
  const float lo = c * 0.5f + f, hi = (float)full - c * 0.5f - f;
  if (lo <= hi) centre = centre < lo ? lo : (centre > hi ? hi : centre);

  // Running sum at pixel boundaries, normalised so it ends exactly at the working length.
  std::vector<double> edge_at(full + 1);
  edge_at[0] = 0.0;
  for (uint32_t i = 0; i < full; ++i)
    edge_at[i + 1] = edge_at[i] + (a.work == full ? 1.0 : Density(i + 0.5f, centre, c * 0.5f, f, e));
  const double k = (double)a.work / edge_at[full];
  for (auto& v : edge_at) v *= k;

  a.fwd.resize(full);
  for (uint32_t i = 0; i < full; ++i) a.fwd[i] = (float)(0.5 * (edge_at[i] + edge_at[i + 1]));

  a.inv.resize(a.work);
  uint32_t i = 0;
  for (uint32_t u = 0; u < a.work; ++u) {
    const double target = u + 0.5;
    while (i + 1 < full && edge_at[i + 1] < target) ++i;
    const double span = edge_at[i + 1] - edge_at[i];
    a.inv[u] = (float)(i + (span > 0.0 ? (target - edge_at[i]) / span : 0.5));
  }
}

// --- GPU state -----------------------------------------------------------

struct State {
  ID3D12RootSignature* root_sig = nullptr;
  ID3D12PipelineState* squeeze_pso = nullptr;
  ID3D12PipelineState* expand_pso = nullptr;
  ID3D12PipelineState* debug_pso = nullptr;
  ID3D12DescriptorHeap* heap = nullptr;
  uint32_t descriptor_size = 0;
  ID3D12Resource* lut[kRingSize] = {};   // upload heap, persistently mapped
  float* lut_map[kRingSize] = {};
  uint32_t lut_floats = 0;
  bool failed = false;

  // Working-size textures, all UNORDERED_ACCESS between frames.
  ID3D12Resource* proxy_s = nullptr;
  ID3D12Resource* depth_s = nullptr;
  ID3D12Resource* motion_s = nullptr;
  ID3D12Resource* answer_s = nullptr;
  uint32_t tex_w = 0, tex_h = 0;

  // This frame's mapping, and the one the previous frame used.
  Axis x, y;
  std::vector<float> prev_x, prev_y;
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
  for (auto*& r : g.lut)
    if (r != nullptr) r->Release();
  for (auto& gr : g.graves)
    if (gr.r != nullptr) gr.r->Release();
  if (g.heap != nullptr) g.heap->Release();
  if (g.squeeze_pso != nullptr) g.squeeze_pso->Release();
  if (g.expand_pso != nullptr) g.expand_pso->Release();
  if (g.debug_pso != nullptr) g.debug_pso->Release();
  if (g.root_sig != nullptr) g.root_sig->Release();
  g = {};
}

// The working size the current settings give, for the NR feature and its textures.
inline void WorkSize(uint32_t full_w, uint32_t full_h, uint32_t* w, uint32_t* h) {
  const auto& st = settings;
  *w = WorkLength(full_w, st.width * full_w, st.falloff * full_h, st.edge);
  *h = WorkLength(full_h, st.height * full_h, st.falloff * full_h, st.edge);
}

// Root signature: [0] SRV table t0-t3, [1] UAV table u0-u2, [2] constants b0, static linear s0.
inline bool Init(ID3D12Device* device) {
  if (g.root_sig != nullptr) return true;
  if (g.failed) return false;

  D3D12_DESCRIPTOR_RANGE srv = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4, 0, 0, 0};
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
  pso.CS = {fov_squeeze_dxil, sizeof(fov_squeeze_dxil)};
  if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&g.squeeze_pso)))) { g.failed = true; return false; }
  pso.CS = {fov_expand_dxil, sizeof(fov_expand_dxil)};
  if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&g.expand_pso)))) { g.failed = true; return false; }
  pso.CS = {fov_debug_dxil, sizeof(fov_debug_dxil)};
  if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&g.debug_pso)))) { g.failed = true; return false; }

  D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kRingSize * kSlotsPerFrame,
                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
  if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g.heap)))) { g.failed = true; return false; }
  g.descriptor_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  return true;
}

inline bool EnsureLut(ID3D12Device* device, uint32_t floats) {
  if (g.lut[0] != nullptr && g.lut_floats >= floats) return true;
  for (uint32_t i = 0; i < kRingSize; ++i) {
    Bury(g.lut[i]);   // only on a size change
    g.lut_map[i] = nullptr;
  }
  D3D12_HEAP_PROPERTIES heap = {D3D12_HEAP_TYPE_UPLOAD};
  D3D12_RESOURCE_DESC desc = {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = (UINT64)floats * 4;
  desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  for (uint32_t i = 0; i < kRingSize; ++i) {
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&g.lut[i]))))
      return false;
    const D3D12_RANGE none = {0, 0};
    void* p = nullptr;
    if (FAILED(g.lut[i]->Map(0, &none, &p))) return false;
    g.lut_map[i] = static_cast<float*>(p);
  }
  g.lut_floats = floats;
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

inline void LutSrv(ID3D12Device* d, uint32_t i, uint32_t frame) {
  D3D12_SHADER_RESOURCE_VIEW_DESC v = {};
  v.Format = DXGI_FORMAT_UNKNOWN;
  v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  v.Buffer.NumElements = g.lut_floats;
  v.Buffer.StructureByteStride = 4;
  d->CreateShaderResourceView(g.lut[frame % kRingSize], &v, Cpu(i));
}

inline void Barrier(ID3D12GraphicsCommandList* cmd, ID3D12Resource* r, D3D12_RESOURCE_STATES a,
                    D3D12_RESOURCE_STATES b) {
  D3D12_RESOURCE_BARRIER x = {};
  x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  x.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, a, b};
  cmd->ResourceBarrier(1, &x);
}

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

// Build this frame's mapping, upload it, and squeeze the inputs. Leaves proxy_s, depth_s and
// motion_s NON_PIXEL_SHADER_RESOURCE for the model; expects proxy in that state already and the
// game's depth and motion in a shader-readable state (as DLSS requires of its inputs).
inline bool Squeeze(ID3D12GraphicsCommandList* cmd, ID3D12Resource* proxy, ID3D12Resource* depth,
                    ID3D12Resource* motion, uint32_t full_w, uint32_t full_h, uint32_t render_w,
                    uint32_t render_h, float mv_scale_x, float mv_scale_y, uint32_t frame) {
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return false;
  const auto& st = settings;

  TickGraves();
  UpdateCentre(full_w, full_h);
  g.prev_x = g.x.fwd;
  g.prev_y = g.y.fwd;
  BuildAxis(g.x, full_w, g.centre_x, st.width * full_w, st.falloff * full_h, st.edge);
  BuildAxis(g.y, full_h, g.centre_y, st.height * full_h, st.falloff * full_h, st.edge);
  if (g.prev_x.size() != full_w || g.prev_y.size() != full_h) {
    g.prev_x = g.x.fwd;   // first frame, or a resize: no previous mapping to cross
    g.prev_y = g.y.fwd;
    g.reset = true;
  }

  const uint32_t floats = 2 * (full_w + full_h) + g.x.work + g.y.work;
  if (!Init(device) || !EnsureLut(device, floats) ||
      !EnsureTextures(device, g.x.work, g.y.work)) {
    device->Release();
    return false;
  }

  float* lut = g.lut_map[frame % kRingSize];
  const uint32_t fwd = 0, prev = full_w + full_h, inv = 2 * (full_w + full_h);
  std::memcpy(lut + fwd, g.x.fwd.data(), full_w * 4);
  std::memcpy(lut + fwd + full_w, g.y.fwd.data(), full_h * 4);
  std::memcpy(lut + prev, g.prev_x.data(), full_w * 4);
  std::memcpy(lut + prev + full_w, g.prev_y.data(), full_h * 4);
  std::memcpy(lut + inv, g.x.inv.data(), g.x.work * 4);
  std::memcpy(lut + inv + g.x.work, g.y.inv.data(), g.y.work * 4);

  const uint32_t base = (frame % kRingSize) * kSlotsPerFrame;
  Srv(device, base + 0, proxy, DXGI_FORMAT_R16G16B16A16_FLOAT);
  Srv(device, base + 1, depth, DXGI_FORMAT_R32_FLOAT);
  Srv(device, base + 2, motion, DXGI_FORMAT_R16G16_FLOAT);
  LutSrv(device, base + 3, frame);
  Uav(device, base + 4, g.proxy_s, DXGI_FORMAT_UNKNOWN);
  Uav(device, base + 5, g.depth_s, DXGI_FORMAT_UNKNOWN);
  Uav(device, base + 6, g.motion_s, DXGI_FORMAT_UNKNOWN);

  uint32_t c[kConstants] = {};
  float f;
  c[0] = full_w; c[1] = full_h; c[2] = g.x.work; c[3] = g.y.work;
  c[4] = render_w; c[5] = render_h;
  f = mv_scale_x; std::memcpy(&c[6], &f, 4);
  f = mv_scale_y; std::memcpy(&c[7], &f, 4);
  c[8] = fwd; c[9] = prev; c[10] = inv;

  cmd->SetDescriptorHeaps(1, &g.heap);
  cmd->SetComputeRootSignature(g.root_sig);
  cmd->SetPipelineState(g.squeeze_pso);
  cmd->SetComputeRootDescriptorTable(0, Gpu(base + 0));
  cmd->SetComputeRootDescriptorTable(1, Gpu(base + 4));
  cmd->SetComputeRoot32BitConstants(2, kConstants, c, 0);
  cmd->Dispatch((g.x.work + 7) / 8, (g.y.work + 7) / 8, 1);

  for (ID3D12Resource* t : {g.proxy_s, g.depth_s, g.motion_s})
    Barrier(cmd, t, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  device->Release();
  return true;
}

// After the model: stretch answer_s over the full-size answer (UNORDERED_ACCESS, as the resolve
// expects it) and put every working texture back to UNORDERED_ACCESS.
inline void Expand(ID3D12GraphicsCommandList* cmd, ID3D12Resource* answer, uint32_t frame) {
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return;
  const uint32_t base = (frame % kRingSize) * kSlotsPerFrame + 7;

  Barrier(cmd, g.answer_s, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  Srv(device, base + 0, g.answer_s, DXGI_FORMAT_UNKNOWN);
  Srv(device, base + 1, nullptr, DXGI_FORMAT_R32_FLOAT);
  Srv(device, base + 2, nullptr, DXGI_FORMAT_R16G16_FLOAT);
  LutSrv(device, base + 3, frame);
  Uav(device, base + 4, answer, DXGI_FORMAT_UNKNOWN);
  Uav(device, base + 5, nullptr, DXGI_FORMAT_R32_FLOAT);
  Uav(device, base + 6, nullptr, DXGI_FORMAT_R16G16_FLOAT);

  const uint32_t full_w = g.x.full, full_h = g.y.full;
  uint32_t c[kConstants] = {full_w, full_h, g.x.work, g.y.work, 1, 1, 0, 0,
                            0, full_w + full_h, 2 * (full_w + full_h), 0, 0, 0, 0};
  cmd->SetDescriptorHeaps(1, &g.heap);
  cmd->SetComputeRootSignature(g.root_sig);
  cmd->SetPipelineState(g.expand_pso);
  cmd->SetComputeRootDescriptorTable(0, Gpu(base + 0));
  cmd->SetComputeRootDescriptorTable(1, Gpu(base + 4));
  cmd->SetComputeRoot32BitConstants(2, kConstants, c, 0);
  cmd->Dispatch((full_w + 7) / 8, (full_h + 7) / 8, 1);

  Barrier(cmd, g.answer_s, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  for (ID3D12Resource* t : {g.proxy_s, g.depth_s, g.motion_s})
    Barrier(cmd, t, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  device->Release();
}

// Debug view 4: tint the finished frame by the resolution the model saw there and mark the fovea
// centre and the raw gaze point. The frame is the game's output, UNORDERED_ACCESS after the resolve.
inline void DebugMap(ID3D12GraphicsCommandList* cmd, ID3D12Resource* frame_res, uint32_t frame) {
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return;
  const uint32_t base = (frame % kRingSize) * kSlotsPerFrame + 14;
  Srv(device, base + 0, nullptr, DXGI_FORMAT_R16G16B16A16_FLOAT);
  Srv(device, base + 1, nullptr, DXGI_FORMAT_R32_FLOAT);
  Srv(device, base + 2, nullptr, DXGI_FORMAT_R16G16_FLOAT);
  LutSrv(device, base + 3, frame);
  Uav(device, base + 4, frame_res, DXGI_FORMAT_UNKNOWN);
  Uav(device, base + 5, nullptr, DXGI_FORMAT_R32_FLOAT);
  Uav(device, base + 6, nullptr, DXGI_FORMAT_R16G16_FLOAT);

  const uint32_t full_w = g.x.full, full_h = g.y.full;
  float gx = -1.0f, gy = -1.0f;
  if (GazeLatest(&gx, &gy)) { gx *= (float)full_w; gy *= (float)full_h; }
  else gx = gy = -1.0f;
  const float floats[4] = {g.centre_x, g.centre_y, gx, gy};
  uint32_t c[kConstants] = {full_w, full_h, g.x.work, g.y.work, 1, 1, 0, 0,
                            0, full_w + full_h, 2 * (full_w + full_h)};
  std::memcpy(&c[11], floats, sizeof(floats));

  D3D12_RESOURCE_BARRIER uav = {};
  uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  uav.UAV.pResource = frame_res;
  cmd->ResourceBarrier(1, &uav);   // after the resolve's writes
  cmd->SetDescriptorHeaps(1, &g.heap);
  cmd->SetComputeRootSignature(g.root_sig);
  cmd->SetPipelineState(g.debug_pso);
  cmd->SetComputeRootDescriptorTable(0, Gpu(base + 0));
  cmd->SetComputeRootDescriptorTable(1, Gpu(base + 4));
  cmd->SetComputeRoot32BitConstants(2, kConstants, c, 0);
  cmd->Dispatch((full_w + 7) / 8, (full_h + 7) / 8, 1);
  device->Release();
}

// Undo Squeeze's transitions when the model fails after it ran.
inline void Abort(ID3D12GraphicsCommandList* cmd) {
  for (ID3D12Resource* t : {g.proxy_s, g.depth_s, g.motion_s})
    Barrier(cmd, t, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

}  // namespace fov
