// Variable-rate shading test: a coarse shading rate on the game's own draws, by the render target they draw into
// (live ini VrsTest: 1 the render-resolution targets, before DLSS; 2 the output-resolution float targets, the
// post-processing after it; 3 both. VrsRate: 1 2x2, 2 4x4). Uniform over the screen -- it measures how much
// of the game's frame shading rate can buy, before a gaze-shaped rate image is worth building.
#pragma once

#include <windows.h>

#include <atomic>

#include <d3d12.h>
#include <reshade.hpp>

#include "fov.hpp"
#include "ngx_probe.hpp"

namespace vrs {

inline std::atomic<int> test{0};
inline std::atomic<int> rate{1};

inline D3D12_SHADING_RATE Rate() { return rate.load() == 2 ? D3D12_SHADING_RATE_4X4 : D3D12_SHADING_RATE_2X2; }

inline bool Wanted(const reshade::api::resource_desc& d) {
  const uint32_t render_w = fov::g.render_w, full_w = fov::g.x.full;
  if (render_w == 0 || full_w == 0) return false;
  const int t = test.load();
  if ((t & 1) && d.texture.width == render_w) return true;
  return (t & 2) && d.texture.width == full_w && d.texture.format == reshade::api::format::r16g16b16a16_float;
}

inline void OnBind(reshade::api::command_list* cmd, uint32_t count, const reshade::api::resource_view* rtvs,
                   reshade::api::resource_view) {
  static std::atomic<bool> was_on{false};
  const bool on = test.load() != 0;
  if (!on && !was_on.load()) return;
  was_on = on;
  bool coarse = false;
  if (on && count > 0 && rtvs[0].handle != 0) {
    reshade::api::device* dev = cmd->get_device();
    coarse = Wanted(dev->get_resource_desc(dev->get_resource_from_view(rtvs[0])));
  }
  auto* native = reinterpret_cast<ID3D12GraphicsCommandList*>(cmd->get_native());
  ID3D12GraphicsCommandList5* l5 = nullptr;
  if (FAILED(native->QueryInterface(IID_PPV_ARGS(&l5))) || l5 == nullptr) return;
  l5->RSSetShadingRate(coarse ? Rate() : D3D12_SHADING_RATE_1X1, nullptr);
  l5->Release();
}

inline void LogSupport(ID3D12Device* device) {
  D3D12_FEATURE_DATA_D3D12_OPTIONS6 o = {};
  if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS6, &o, sizeof(o))))
    ngx_probe::Logf("vrs: tier %d, image tile %u, additional rates %d", (int)o.VariableShadingRateTier,
                    o.ShadingRateImageTileSize, (int)o.AdditionalShadingRatesSupported);
}

inline void Register() { reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(OnBind); }
inline void Unregister() {
  reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(OnBind);
}

}  // namespace vrs
