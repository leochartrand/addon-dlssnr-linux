// Foveated super resolution: the game's DLSS-SR is the largest single cost of its frame (4.6 ms for 1080p ->
// 4K on the 3080), and the foveated pass only looks at the periphery at 896x512 anyway. Live ini:
//   SrPreset=<n>     the DLSS preset the game's upscaler is created with (NVSDK_NGX_DLSS_Hint_Render_Preset;
//                    0 = the game's own); read at the create, so set it before the world loads
//   SrFovea=1        a second DLSS-SR, created here, on a rectangle around the gaze crop only (SrCropPreset),
//                    written over the game's upscale
//   SrFovea=2        the same, and no game upscale: the rest of the frame is a bilinear stretch (sr_upscale)
//   SrMargin=<px>    output pixels around the crop the rectangle also covers (default 64)
// The rectangle follows the crop of the frame before (the foveated pass places it later in the frame); when it
// moves, its DLSS history starts over.
#pragma once

#include <windows.h>

#include <atomic>
#include <cmath>
#include <cstdint>

#include <d3d12.h>
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>

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

// --- the bilinear stretch (SrFovea=2) ---------------------------------------

constexpr uint32_t kRing = 8;

struct Stretch {
  ID3D12RootSignature* root = nullptr;
  ID3D12PipelineState* pso = nullptr;
  ID3D12DescriptorHeap* heap = nullptr;
  uint32_t step = 0, frame = 0;
  bool failed = false;
};
inline Stretch st;

// Root signature as nr_compose's: [0] SRV table t0-t1, [1] UAV table u0, [2] eight constants b0.
inline bool StretchInit(ID3D12Device* device) {
  if (st.pso != nullptr) return true;
  if (st.failed) return false;
  st.failed = true;
  D3D12_DESCRIPTOR_RANGE srv = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, 0};
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
  D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kRing * 3,
                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
  if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&st.heap)))) return false;
  st.step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  st.failed = false;
  return true;
}

// color (NON_PIXEL_SHADER_RESOURCE) stretched into output (UNORDERED_ACCESS); both stay in their states.
inline bool StretchRecord(ID3D12GraphicsCommandList* cmd, ID3D12Resource* color, ID3D12Resource* output,
                          float jx, float jy) {
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return false;
  const bool ok = StretchInit(device);
  if (ok) {
    const uint32_t base = (st.frame++ % kRing) * 3;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = st.heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = st.heap->GetGPUDescriptorHandleForHeapStart();
    auto at_cpu = [&](uint32_t i) { return D3D12_CPU_DESCRIPTOR_HANDLE{cpu.ptr + SIZE_T(base + i) * st.step}; };
    auto at_gpu = [&](uint32_t i) { return D3D12_GPU_DESCRIPTOR_HANDLE{gpu.ptr + UINT64(base + i) * st.step}; };
    const D3D12_RESOURCE_DESC cd = color->GetDesc(), od = output->GetDesc();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
    srv.Format = cd.Format;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(color, &srv, at_cpu(0));
    device->CreateShaderResourceView(color, &srv, at_cpu(1));
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format = od.Format;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(output, nullptr, &uav, at_cpu(2));
    const float sign = (float)jitter_sign.load();
    const float fx = jx * sign, fy = jy * sign;
    const uint32_t c[8] = {(uint32_t)cd.Width, cd.Height, (uint32_t)od.Width, od.Height,
                           *reinterpret_cast<const uint32_t*>(&fx), *reinterpret_cast<const uint32_t*>(&fy), 0, 0};
    ID3D12DescriptorHeap* heaps[] = {st.heap};
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetComputeRootSignature(st.root);
    cmd->SetPipelineState(st.pso);
    cmd->SetComputeRootDescriptorTable(0, at_gpu(0));
    cmd->SetComputeRootDescriptorTable(1, at_gpu(2));
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

  diag::SpanStart(crop_span, cmd, ngx_probe::Log);
  const NVSDK_NGX_Result res = evaluate(cmd, cr.handle, p, nullptr);
  diag::SpanStop(crop_span, cmd);
  static uint32_t evals = 0;
  if (++evals <= 3 || res != NVSDK_NGX_Result_Success)
    logf("sr-fovea: crop evaluate #%u at %u,%u (%s) => 0x%x", evals, bx, by, moved ? "moved, reset" : "held",
         (unsigned)res);
}

}  // namespace sr_fovea
