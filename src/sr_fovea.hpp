// Foveated super resolution: the game's DLSS-SR is the largest single cost of its frame (4.6 ms for 1080p ->
// 4K on the 3080), and the foveated pass only looks at the periphery at 896x512 anyway. Live ini:
//   SrPreset=<n>     the DLSS preset the game's upscaler is created with (NVSDK_NGX_DLSS_Hint_Render_Preset;
//                    0 = the game's own); read at the create, so set it before the world loads
//   SrFovea=1        a second DLSS-SR, created here, on a rectangle around the gaze crop only (SrCropPreset),
//                    written over the game's upscale
//   SrFovea=2        the same, and no game upscale: the rest of the frame is a bilinear stretch (sr_upscale)
//   SrFovea=3        as 2, with a small temporal accumulation before the stretch (SrTaa: this frame's weight, %)
//   SrMargin=<px>    output pixels around the crop the rectangle also covers (default 64)
//   SrFeather=1      the crop's answer goes to a texture of its own and fades in over the margin (0: written
//                    straight into the frame, a hard edge)
// The rectangle follows the crop of the frame before (the foveated pass places it later in the frame); when it
// moves, its DLSS history starts over.
#pragma once

#include <windows.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <utility>

#include <d3d12.h>
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>

#include "../build/shaders/sr_blend_dxil.h"
#include "../build/shaders/sr_taa_dxil.h"
#include "../build/shaders/sr_upscale_dxil.h"
#include "diag.hpp"
#include "fov.hpp"
#include "nr_compose.hpp"

// Included by ngx_probe.hpp after its logging helpers (ngx_probe::Log).

namespace sr_fovea {

inline std::atomic<int> preset{0};
inline std::atomic<int> mode{0};
inline std::atomic<int> crop_preset{0};
inline std::atomic<int> margin{64};
inline std::atomic<int> jitter_sign{-1};
inline std::atomic<int> game_flags{0};   // the game's DLSS-SR create flags, reused for the crop's

using LogfFn = void (*)(const char*, ...);
using CreateFn = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*,
                                                NVSDK_NGX_Handle**);
using EvaluateFn = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*,
                                                  const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);

inline void SetPresetHints(NVSDK_NGX_Parameter* p, unsigned v) {
  p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, v);
  p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, v);
  p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, v);
  p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, v);
  p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, v);
  p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality, v);
}

// At the game's DLSS-SR create: log the presets it asked for, and replace them when SrPreset says so.
inline void OnGameCreate(NVSDK_NGX_Parameter* p, LogfFn logf) {
  unsigned perf = 0, ultra = 0, quality = 0;
  p->Get(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, &perf);
  p->Get(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, &ultra);
  p->Get(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, &quality);
  int flags = 0;
  p->Get(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, &flags);
  game_flags = flags;
  const int v = preset.load();
  logf("sr-fovea: game's preset hints performance %u, ultra performance %u, quality %u; SrPreset %d", perf, ultra,
       quality, v);
  if (v > 0) SetPresetHints(p, (unsigned)v);
}

// --- the periphery without the game's upscale (SrFovea=2: a bilinear stretch; 3: a small TAA, then the stretch)

constexpr uint32_t kRing = 16;   // descriptor sets in flight (a frame takes up to two)
constexpr uint32_t kPerFrame = 12;  // descriptors per frame: TAA t0-t2 u0, stretch t0-t2 u0, blend t0-t2 u0

struct Stretch {
  ID3D12RootSignature* root = nullptr;
  ID3D12PipelineState* pso = nullptr;
  ID3D12PipelineState* taa_pso = nullptr;
  ID3D12PipelineState* blend_pso = nullptr;
  ID3D12DescriptorHeap* heap = nullptr;
  uint32_t step = 0, frame = 0;
  ID3D12Resource* hist[2] = {};   // the accumulation, alternating (NON_PIXEL_SHADER_RESOURCE between frames)
  uint32_t hist_w = 0, hist_h = 0, flip = 0;
  bool hist_valid = false;
  bool failed = false;
};
inline Stretch st;
inline std::atomic<int> taa_percent{10};
inline std::atomic<int> feather{1};   // SrTaa: this frame's weight in the accumulation, in percent

// [0] SRV table t0-t2, [1] UAV table u0, [2] eight constants b0.
inline bool StretchInit(ID3D12Device* device) {
  if (st.pso != nullptr) return true;
  if (st.failed) return false;
  st.failed = true;
  D3D12_DESCRIPTOR_RANGE srv = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0, 0};
  D3D12_DESCRIPTOR_RANGE uav = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0};
  D3D12_ROOT_PARAMETER params[3] = {};
  params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  params[0].DescriptorTable = {1, &srv};
  params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  params[1].DescriptorTable = {1, &uav};
  params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  params[2].Constants = {0, 0, 8};
  D3D12_ROOT_SIGNATURE_DESC desc = {3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
  ID3DBlob* blob = nullptr;
  ID3DBlob* error = nullptr;
  if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error))) {
    if (error != nullptr) error->Release();
    return false;
  }
  const HRESULT hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                                 IID_PPV_ARGS(&st.root));
  blob->Release();
  if (FAILED(hr)) return false;
  D3D12_COMPUTE_PIPELINE_STATE_DESC pso = {};
  pso.pRootSignature = st.root;
  pso.CS = {sr_upscale_dxil, sizeof(sr_upscale_dxil)};
  if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&st.pso)))) return false;
  pso.CS = {sr_taa_dxil, sizeof(sr_taa_dxil)};
  if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&st.taa_pso)))) return false;
  pso.CS = {sr_blend_dxil, sizeof(sr_blend_dxil)};
  if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&st.blend_pso)))) return false;
  D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kRing * kPerFrame,
                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
  if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&st.heap)))) return false;
  st.step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  st.failed = false;
  return true;
}

inline bool EnsureHistory(ID3D12Device* device, const D3D12_RESOURCE_DESC& color) {
  if (st.hist[0] != nullptr && st.hist_w == color.Width && st.hist_h == color.Height) return true;
  for (auto*& h : st.hist)
    if (h != nullptr) h->Release(), h = nullptr;
  D3D12_HEAP_PROPERTIES heap = {};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC d = {};
  d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  d.Width = color.Width;
  d.Height = color.Height;
  d.DepthOrArraySize = 1;
  d.MipLevels = 1;
  d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
  d.SampleDesc.Count = 1;
  d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  for (auto*& h : st.hist)
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d,
                                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
                                               IID_PPV_ARGS(&h))))
      return false;
  st.hist_w = (uint32_t)color.Width;
  st.hist_h = color.Height;
  st.hist_valid = false;
  return true;
}

// The game's Color (NON_PIXEL_SHADER_RESOURCE) into its Output (UNORDERED_ACCESS), both left in their states:
// stretched as it is (taa false), or accumulated first. `game` gives the motion vectors, jitter and reset.
inline bool StretchRecord(ID3D12GraphicsCommandList* cmd, const NVSDK_NGX_Parameter* game, bool taa) {
  ID3D12Resource *color = nullptr, *output = nullptr, *motion = nullptr;
  float jx = 0.0f, jy = 0.0f, mvx = 1.0f, mvy = 1.0f;
  int reset = 0;
  game->Get(NVSDK_NGX_Parameter_Color, &color);
  game->Get(NVSDK_NGX_Parameter_Output, &output);
  game->Get(NVSDK_NGX_Parameter_MotionVectors, &motion);
  game->Get(NVSDK_NGX_Parameter_Jitter_Offset_X, &jx);
  game->Get(NVSDK_NGX_Parameter_Jitter_Offset_Y, &jy);
  game->Get(NVSDK_NGX_Parameter_MV_Scale_X, &mvx);
  game->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &mvy);
  game->Get(NVSDK_NGX_Parameter_Reset, &reset);
  if (color == nullptr || output == nullptr || (taa && motion == nullptr)) return false;
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return false;
  const D3D12_RESOURCE_DESC cd = color->GetDesc(), od = output->GetDesc();
  bool ok = StretchInit(device) && (!taa || EnsureHistory(device, cd));
  if (ok) {
    const uint32_t base = (st.frame++ % kRing) * kPerFrame;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = st.heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = st.heap->GetGPUDescriptorHandleForHeapStart();
    auto at_cpu = [&](uint32_t i) { return D3D12_CPU_DESCRIPTOR_HANDLE{cpu.ptr + SIZE_T(base + i) * st.step}; };
    auto at_gpu = [&](uint32_t i) { return D3D12_GPU_DESCRIPTOR_HANDLE{gpu.ptr + UINT64(base + i) * st.step}; };
    auto srv = [&](ID3D12Resource* r, DXGI_FORMAT f, uint32_t i) {
      D3D12_SHADER_RESOURCE_VIEW_DESC v = {};
      v.Format = f;
      v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      v.Texture2D.MipLevels = 1;
      device->CreateShaderResourceView(r, &v, at_cpu(i));
    };
    auto uav = [&](ID3D12Resource* r, DXGI_FORMAT f, uint32_t i) {
      D3D12_UNORDERED_ACCESS_VIEW_DESC v = {};
      v.Format = f;
      v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
      device->CreateUnorderedAccessView(r, nullptr, &v, at_cpu(i));
    };
    auto bits = [](float f) { return *reinterpret_cast<const uint32_t*>(&f); };
    auto transition = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
      D3D12_RESOURCE_BARRIER b = {};
      b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      b.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
      cmd->ResourceBarrier(1, &b);
    };
    ID3D12DescriptorHeap* heaps[] = {st.heap};
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetComputeRootSignature(st.root);

    ID3D12Resource* source = color;
    DXGI_FORMAT source_format = cd.Format;
    float sx = jx * (float)jitter_sign.load(), sy = jy * (float)jitter_sign.load();
    if (taa) {
      ID3D12Resource* prev = st.hist[st.flip];
      ID3D12Resource* next = st.hist[st.flip ^ 1];
      st.flip ^= 1;
      srv(color, cd.Format, 0);
      srv(prev, DXGI_FORMAT_R16G16B16A16_FLOAT, 1);
      srv(motion, motion->GetDesc().Format, 2);
      uav(next, DXGI_FORMAT_R16G16B16A16_FLOAT, 3);
      const float w = (reset != 0 || !st.hist_valid) ? 1.0f : (float)taa_percent.load() / 100.0f;
      st.hist_valid = true;
      const uint32_t c[8] = {(uint32_t)cd.Width, cd.Height, bits(mvx), bits(mvy), bits(w), 0, 0, 0};
      transition(next, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      cmd->SetPipelineState(st.taa_pso);
      cmd->SetComputeRootDescriptorTable(0, at_gpu(0));
      cmd->SetComputeRootDescriptorTable(1, at_gpu(3));
      cmd->SetComputeRoot32BitConstants(2, 8, c, 0);
      cmd->Dispatch(((uint32_t)cd.Width + 15) / 16, (cd.Height + 15) / 16, 1);
      transition(next, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      source = next;
      source_format = DXGI_FORMAT_R16G16B16A16_FLOAT;
      sx = sy = 0.0f;   // the accumulation has no jitter of its own
    }
    srv(source, source_format, 4);
    srv(source, source_format, 5);
    srv(source, source_format, 6);
    uav(output, od.Format, 7);
    const uint32_t c[8] = {(uint32_t)cd.Width, cd.Height, (uint32_t)od.Width, od.Height, bits(sx), bits(sy), 0, 0};
    cmd->SetPipelineState(st.pso);
    cmd->SetComputeRootDescriptorTable(0, at_gpu(4));
    cmd->SetComputeRootDescriptorTable(1, at_gpu(7));
    cmd->SetComputeRoot32BitConstants(2, 8, c, 0);
    cmd->Dispatch(((uint32_t)od.Width + 15) / 16, (od.Height + 15) / 16, 1);
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = output;
    cmd->ResourceBarrier(1, &b);
  }
  device->Release();
  return ok;
}

// --- the crop's own DLSS-SR ---------------------------------------------------

struct Crop {
  NVSDK_NGX_Handle* handle = nullptr;
  NVSDK_NGX_Parameter* params = nullptr;
  uint32_t out_w = 0, out_h = 0, in_w = 0, in_h = 0, ratio = 0;
  int base_x = -1, base_y = -1;     // the rectangle's output origin last frame
  ID3D12Resource* answer = nullptr; // SrFeather: the crop's answer (UNORDERED_ACCESS between frames)
  uint32_t resets = 0, evals = 0;
  bool failed = false;
};
inline Crop cr;
inline diag::Span crop_span{"DLSS-SR crop"};

inline NVSDK_NGX_Parameter* AllocateParameters() {
  using AllocFn = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter**);
  for (const char* name : {"_nvngx.dll", "nvngx.dll"}) {
    if (HMODULE m = GetModuleHandleA(name)) {
      if (auto alloc = reinterpret_cast<AllocFn>(GetProcAddress(m, "NVSDK_NGX_D3D12_AllocateParameters"))) {
        NVSDK_NGX_Parameter* p = nullptr;
        if (alloc(&p) == NVSDK_NGX_Result_Success) return p;
      }
    }
  }
  return nullptr;
}

inline bool EnsureAnswer(ID3D12Device* device, DXGI_FORMAT format) {
  if (cr.answer != nullptr) return true;
  D3D12_HEAP_PROPERTIES heap = {};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC d = {};
  d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  d.Width = cr.out_w;
  d.Height = cr.out_h;
  d.DepthOrArraySize = 1;
  d.MipLevels = 1;
  d.Format = format;
  d.SampleDesc.Count = 1;
  d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d,
                                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                   IID_PPV_ARGS(&cr.answer)));
}

// The crop's answer (UNORDERED_ACCESS, left so) faded into the frame (UNORDERED_ACCESS) at bx,by over `fade` px.
inline void Blend(ID3D12GraphicsCommandList* cmd, ID3D12Device* device, ID3D12Resource* frame, DXGI_FORMAT format,
                  uint32_t bx, uint32_t by, uint32_t fade) {
  const uint32_t base = ((st.frame++) % kRing) * kPerFrame + 8;
  D3D12_CPU_DESCRIPTOR_HANDLE cpu = st.heap->GetCPUDescriptorHandleForHeapStart();
  D3D12_GPU_DESCRIPTOR_HANDLE gpu = st.heap->GetGPUDescriptorHandleForHeapStart();
  auto at_cpu = [&](uint32_t i) { return D3D12_CPU_DESCRIPTOR_HANDLE{cpu.ptr + SIZE_T(base + i) * st.step}; };
  auto at_gpu = [&](uint32_t i) { return D3D12_GPU_DESCRIPTOR_HANDLE{gpu.ptr + UINT64(base + i) * st.step}; };
  D3D12_SHADER_RESOURCE_VIEW_DESC v = {};
  v.Format = format;
  v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  v.Texture2D.MipLevels = 1;
  for (uint32_t i = 0; i < 3; ++i) device->CreateShaderResourceView(cr.answer, &v, at_cpu(i));
  D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
  u.Format = format;
  u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  device->CreateUnorderedAccessView(frame, nullptr, &u, at_cpu(3));
  D3D12_RESOURCE_BARRIER b = {};
  b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  b.Transition = {cr.answer, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
  cmd->ResourceBarrier(1, &b);
  const float f = (float)fade;
  const uint32_t c[8] = {bx, by, cr.out_w, cr.out_h, *reinterpret_cast<const uint32_t*>(&f), 0, 0, 0};
  ID3D12DescriptorHeap* heaps[] = {st.heap};
  cmd->SetDescriptorHeaps(1, heaps);
  cmd->SetComputeRootSignature(st.root);
  cmd->SetPipelineState(st.blend_pso);
  cmd->SetComputeRootDescriptorTable(0, at_gpu(0));
  cmd->SetComputeRootDescriptorTable(1, at_gpu(3));
  cmd->SetComputeRoot32BitConstants(2, 8, c, 0);
  cmd->Dispatch((cr.out_w + 15) / 16, (cr.out_h + 15) / 16, 1);
  std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
  cmd->ResourceBarrier(1, &b);
  D3D12_RESOURCE_BARRIER uav = {};
  uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  uav.UAV.pResource = frame;
  cmd->ResourceBarrier(1, &uav);
}

// The game's DLSS-SR evaluate (`game`, its Color at render size and Output at full size) is done or skipped;
// run the crop's over the rectangle around last frame's gaze crop.
inline void CropEvaluate(ID3D12GraphicsCommandList* cmd, const NVSDK_NGX_Parameter* game, CreateFn create,
                         EvaluateFn evaluate, LogfFn logf) {
  if (cr.failed) return;
  ID3D12Resource *color = nullptr, *output = nullptr, *depth = nullptr, *motion = nullptr, *exposure = nullptr;
  game->Get(NVSDK_NGX_Parameter_Color, &color);
  game->Get(NVSDK_NGX_Parameter_Output, &output);
  game->Get(NVSDK_NGX_Parameter_Depth, &depth);
  game->Get(NVSDK_NGX_Parameter_MotionVectors, &motion);
  game->Get(NVSDK_NGX_Parameter_ExposureTexture, &exposure);
  if (color == nullptr || output == nullptr || depth == nullptr || motion == nullptr) return;
  const D3D12_RESOURCE_DESC cd = color->GetDesc(), od = output->GetDesc();
  const uint32_t full_w = (uint32_t)od.Width, full_h = od.Height;
  if (cd.Width == 0 || full_w % cd.Width != 0 || full_w / cd.Width != full_h / cd.Height) {
    logf("sr-fovea: %ux%u -> %ux%u is not a whole ratio; the crop's upscale is off", (unsigned)cd.Width, cd.Height,
         full_w, full_h);
    cr.failed = true;
    return;
  }
  const uint32_t r = full_w / (uint32_t)cd.Width;

  // The gaze crop as the foveated pass placed it last frame (the frame's centre before it ran).
  uint32_t crop_w = 1024, crop_h = 576, crop_x = (full_w - crop_w) / 2, crop_y = (full_h - crop_h) / 2;
  if (fov::g.x.full == full_w && fov::g.y.full == full_h) {
    const fov::SplitLayout l = fov::SplitSize(full_w, full_h);
    crop_w = l.crop_w;
    crop_h = l.crop_h;
    crop_x = fov::g.crop_x;
    crop_y = fov::g.crop_y;
  }
  const uint32_t m = (uint32_t)(margin.load() < 0 ? 0 : margin.load());
  auto whole = [&](uint32_t v) { return (v + r - 1) / r * r; };   // output lengths/origins on the render grid
  const uint32_t out_w = whole(crop_w + 2 * m < full_w ? crop_w + 2 * m : full_w);
  const uint32_t out_h = whole(crop_h + 2 * m < full_h ? crop_h + 2 * m : full_h);
  auto origin = [&](uint32_t c, uint32_t len, uint32_t full) {
    const int v = (int)c - (int)m;
    const int hi = (int)full - (int)len;
    return (uint32_t)(v < 0 ? 0 : v > hi ? hi : v) / r * r;
  };
  const uint32_t bx = origin(crop_x, out_w, full_w), by = origin(crop_y, out_h, full_h);

  if (cr.handle != nullptr && (cr.out_w != out_w || cr.out_h != out_h || cr.ratio != r)) {
    logf("sr-fovea: the crop's rectangle changed size; its DLSS-SR is not rebuilt (restart the game)");
    cr.failed = true;
    return;
  }
  if (cr.handle == nullptr) {
    cr.params = AllocateParameters();
    if (cr.params == nullptr) {
      logf("sr-fovea: NVSDK_NGX_D3D12_AllocateParameters unavailable");
      cr.failed = true;
      return;
    }
    cr.out_w = out_w;
    cr.out_h = out_h;
    cr.in_w = out_w / r;
    cr.in_h = out_h / r;
    cr.ratio = r;
    NVSDK_NGX_Parameter* p = cr.params;
    p->Set(NVSDK_NGX_Parameter_CreationNodeMask, 1u);
    p->Set(NVSDK_NGX_Parameter_VisibilityNodeMask, 1u);
    p->Set(NVSDK_NGX_Parameter_Width, cr.in_w);
    p->Set(NVSDK_NGX_Parameter_Height, cr.in_h);
    p->Set(NVSDK_NGX_Parameter_OutWidth, cr.out_w);
    p->Set(NVSDK_NGX_Parameter_OutHeight, cr.out_h);
    p->Set(NVSDK_NGX_Parameter_PerfQualityValue, (int)(r >= 3 ? NVSDK_NGX_PerfQuality_Value_UltraPerformance
                                                      : r == 2 ? NVSDK_NGX_PerfQuality_Value_MaxPerf
                                                               : NVSDK_NGX_PerfQuality_Value_DLAA));
    p->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, (int)game_flags.load());
    p->Set(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 1);
    if (crop_preset.load() > 0) SetPresetHints(p, (unsigned)crop_preset.load());
    const NVSDK_NGX_Result res = create(cmd, NVSDK_NGX_Feature_SuperSampling, p, &cr.handle);
    logf("sr-fovea: crop DLSS-SR %ux%u -> %ux%u (preset %d, flags %d) => 0x%x", cr.in_w, cr.in_h, cr.out_w,
         cr.out_h, crop_preset.load(), game_flags.load(), (unsigned)res);
    if (res != NVSDK_NGX_Result_Success) {
      cr.handle = nullptr;
      cr.failed = true;
      return;
    }
  }

  NVSDK_NGX_Parameter* p = cr.params;
  float f = 0.0f;
  int reset = 0;
  p->Set(NVSDK_NGX_Parameter_Color, color);
  p->Set(NVSDK_NGX_Parameter_Output, output);
  p->Set(NVSDK_NGX_Parameter_Depth, depth);
  p->Set(NVSDK_NGX_Parameter_MotionVectors, motion);
  p->Set(NVSDK_NGX_Parameter_ExposureTexture, exposure);
  for (const char* name : {NVSDK_NGX_Parameter_Jitter_Offset_X, NVSDK_NGX_Parameter_Jitter_Offset_Y,
                           NVSDK_NGX_Parameter_MV_Scale_X, NVSDK_NGX_Parameter_MV_Scale_Y,
                           NVSDK_NGX_Parameter_Sharpness, NVSDK_NGX_Parameter_DLSS_Pre_Exposure,
                           NVSDK_NGX_Parameter_DLSS_Exposure_Scale, NVSDK_NGX_Parameter_FrameTimeDeltaInMsec}) {
    f = 0.0f;
    if (game->Get(name, &f) == NVSDK_NGX_Result_Success) p->Set(name, f);
  }
  game->Get(NVSDK_NGX_Parameter_Reset, &reset);
  const bool moved = (int)bx != cr.base_x || (int)by != cr.base_y;
  cr.base_x = (int)bx;
  cr.base_y = (int)by;
  p->Set(NVSDK_NGX_Parameter_Reset, (reset != 0 || moved) ? 1 : 0);
  const unsigned rx = bx / r, ry = by / r;
  p->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X, rx);
  p->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y, ry);
  p->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X, rx);
  p->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, ry);
  p->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X, rx);
  p->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, ry);
  p->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X, bx);
  p->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y, by);
  p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, cr.in_w);
  p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, cr.in_h);

  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return;
  const bool blend = feather.load() != 0 && StretchInit(device) && EnsureAnswer(device, od.Format);
  if (blend) {   // the answer at the origin of a texture of its own; blended into the frame below
    p->Set(NVSDK_NGX_Parameter_Output, cr.answer);
    p->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X, 0u);
    p->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y, 0u);
  }

  diag::SpanStart(crop_span, cmd, ngx_probe::Log);
  const NVSDK_NGX_Result res = evaluate(cmd, cr.handle, p, nullptr);
  if (blend && res == NVSDK_NGX_Result_Success) Blend(cmd, device, output, od.Format, bx, by, m);
  diag::SpanStop(crop_span, cmd);
  device->Release();
  ++cr.evals;
  cr.resets += moved ? 1 : 0;
  if (cr.evals <= 3 || res != NVSDK_NGX_Result_Success)
    logf("sr-fovea: crop evaluate #%u at %u,%u (%s, %s) => 0x%x", cr.evals, bx, by, moved ? "moved, reset" : "held",
         blend ? "feathered" : "hard edge", (unsigned)res);
  if (cr.evals % 600 == 0) {
    logf("sr-fovea: last 600 crop evaluates: %u history resets (the rectangle moved)", cr.resets);
    cr.resets = 0;
  }
}

}  // namespace sr_fovea
