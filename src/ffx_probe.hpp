// The game's own FSR frame generation (AMD FidelityFX, amd_fidelityfx_loader_dx12.dll) after the NR pass. The game
// hands frame generation a HUD-less copy of each frame (HUDLessColor, in its per-frame configure), from which the
// generated frames are made and the UI is told apart; that copy is taken before the NR pass, so the generated frames
// came out without it -- a flicker at half the display rate. Here the configure is hooked and its HUD-less image is
// replaced by the NR pass's answer (nr_runner copies it into hudless after each pass on the graded frame, which is
// the same picture before the UI). The NR pass runs on rendered frames only; the generated ones are made from its
// answers. Live ini: FgNrHudless=0 hands the game's own image back (to compare).
#pragma once

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <utility>

#include <d3d12.h>
#include <detours.h>

#include "third_party/ffx/api/include/dx12/ffx_api_dx12.h"
#include "third_party/ffx/framegeneration/include/dx12/ffx_api_framegeneration_dx12.h"
#include "third_party/ffx/framegeneration/include/ffx_framegeneration.h"

// Included by addon.cpp after ngx_probe.hpp (its logging) and nr_runner.hpp.

namespace ffx_probe {

inline std::atomic<int> enabled{1};   // FgNrHudless
inline bool installed = false;
inline bool blocked = false;
inline PfnFfxCreateContext real_create = nullptr;
inline PfnFfxConfigure real_configure = nullptr;
inline PfnFfxDispatch real_dispatch = nullptr;

// The NR pass's answer for frame generation: written by nr_runner after each pass on the graded frame (left in
// NON_PIXEL_SHADER_RESOURCE), taken by the next frame generation configure.
inline ID3D12Resource* hudless = nullptr;
inline std::atomic<bool> fresh{false};   // written this frame, not yet handed over
inline std::atomic<bool> wanted{false};  // the game configures frame generation with a HUD-less image
inline DXGI_FORMAT game_format = DXGI_FORMAT_UNKNOWN;   // the game's HUD-less image's
inline uint32_t swapped = 0, kept = 0, configures = 0;

// The same pixels, typed or not (the game may hand frame generation a typeless image).
inline DXGI_FORMAT Family(DXGI_FORMAT f) {
  switch (f) {
    case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return DXGI_FORMAT_R8G8B8A8_TYPELESS;
    case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8A8_TYPELESS;
    case DXGI_FORMAT_R10G10B10A2_UNORM: return DXGI_FORMAT_R10G10B10A2_TYPELESS;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return DXGI_FORMAT_R16G16B16A16_TYPELESS;
    default: return f;
  }
}

inline const char* TypeName(uint64_t t) {
  switch (t) {
    case FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION: return "create framegeneration";
    case FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS: return "create framegeneration hudless";
    case FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION: return "configure framegeneration";
    case FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE: return "dispatch prepare";
    case FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2: return "dispatch prepare v2";
    case FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION: return "dispatch framegeneration";
    case FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_WRAP_DX12: return "create swapchain wrap";
    case FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_NEW_DX12: return "create swapchain new";
    case FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_FOR_HWND_DX12: return "create swapchain for hwnd";
    case FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_REGISTERUIRESOURCE_DX12: return "configure ui resource";
    default: return nullptr;
  }
}

// Each descriptor type of a chain, the first time it is seen.
inline void LogChain(const char* call, const ffxApiHeader* h) {
  static uint64_t seen[64] = {};
  static int n = 0;
  for (; h != nullptr; h = h->pNext) {
    bool known = false;
    for (int i = 0; i < n; ++i) known |= seen[i] == h->type;
    if (known || n >= 64) continue;
    seen[n++] = h->type;
    const char* name = TypeName(h->type);
    ngx_probe::Logf("ffx-probe: %s 0x%08llx%s%s", call, (unsigned long long)h->type, name ? " = " : "", name ? name : "");
    if (h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION) {
      const auto* d = reinterpret_cast<const ffxCreateContextDescFrameGeneration*>(h);
      ngx_probe::Logf("ffx-probe:   flags 0x%x display %ux%u max render %ux%u back buffer format %u", d->flags,
                      d->displaySize.width, d->displaySize.height, d->maxRenderSize.width, d->maxRenderSize.height,
                      d->backBufferFormat);
    } else if (h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS) {
      ngx_probe::Logf("ffx-probe:   hudless format %u",
                      reinterpret_cast<const ffxCreateContextDescFrameGenerationHudless*>(h)->hudlessBackBufferFormat);
    }
  }
}

inline ffxReturnCode_t HookCreate(ffxContext* context, ffxCreateContextDescHeader* desc,
                                  const ffxAllocationCallbacks* cb) {
  LogChain("create", desc);
  return real_create(context, desc, cb);
}

inline ffxReturnCode_t HookDispatch(ffxContext* context, const ffxDispatchDescHeader* desc) {
  LogChain("dispatch", desc);
  return real_dispatch(context, desc);
}

inline ffxReturnCode_t HookConfigure(ffxContext* context, const ffxConfigureDescHeader* desc) {
  LogChain("configure", desc);
  if (desc == nullptr || desc->type != FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION) return real_configure(context, desc);
  ffxConfigureDescFrameGeneration d = *reinterpret_cast<const ffxConfigureDescFrameGeneration*>(desc);
  ID3D12Resource* const game = static_cast<ID3D12Resource*>(d.HUDLessColor.resource);
  ++configures;
  if (configures <= 3)
    ngx_probe::Logf("ffx-probe: configure #%u: frame generation %d, HUD-less %p (format %u, %ux%u, state 0x%x), "
                    "flags 0x%x, frame %llu", configures, d.frameGenerationEnabled ? 1 : 0, (void*)game,
                    d.HUDLessColor.description.format, d.HUDLessColor.description.width,
                    d.HUDLessColor.description.height, d.HUDLessColor.state, d.flags,
                    (unsigned long long)d.frameID);
  if (game != nullptr) {
    game_format = game->GetDesc().Format;
    wanted = d.frameGenerationEnabled;
  } else {
    wanted = false;
  }
  bool swap = enabled.load() != 0 && game != nullptr && d.frameGenerationEnabled && hudless != nullptr &&
              fresh.exchange(false);
  if (swap) {
    const D3D12_RESOURCE_DESC ours = hudless->GetDesc(), theirs = game->GetDesc();
    if (Family(ours.Format) != Family(theirs.Format) || ours.Width != theirs.Width || ours.Height != theirs.Height) {
      static bool told = false;
      if (!told)
        ngx_probe::Logf("ffx-probe: the game's HUD-less image (%llux%u format %u) does not match the NR pass's "
                        "(%llux%u format %u): the game's is kept", (unsigned long long)theirs.Width, theirs.Height,
                        (unsigned)theirs.Format, (unsigned long long)ours.Width, ours.Height, (unsigned)ours.Format);
      told = true;
      swap = false;
    }
  }
  if (swap) {
    d.HUDLessColor = ffxApiGetResourceDX12(hudless, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.HUDLessColor.description.format = d.HUDLessColor.description.format == 0
                                            ? reinterpret_cast<const ffxConfigureDescFrameGeneration*>(desc)
                                                  ->HUDLessColor.description.format
                                            : d.HUDLessColor.description.format;
    ++swapped;
  } else if (game != nullptr && d.frameGenerationEnabled) {
    ++kept;
  }
  if (configures % 600 == 0) {
    ngx_probe::Logf("ffx-probe: last 600 configures: the NR pass's HUD-less image %u, the game's %u (FgNrHudless %d)",
                    swapped, kept, enabled.load());
    swapped = kept = 0;
  }
  return real_configure(context, &d.header);
}

// Hooks the loader's exports once the game has loaded it (polled each present, like ngx_probe).
inline void TryInstall() {
  if (installed || blocked) return;
  HMODULE m = GetModuleHandleA("amd_fidelityfx_loader_dx12.dll");
  if (m == nullptr) m = GetModuleHandleA("amd_fidelityfx_dx12.dll");
  if (m == nullptr) return;
  real_create = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(m, "ffxCreateContext"));
  real_configure = reinterpret_cast<PfnFfxConfigure>(GetProcAddress(m, "ffxConfigure"));
  real_dispatch = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(m, "ffxDispatch"));
  if (real_create == nullptr || real_configure == nullptr || real_dispatch == nullptr) {
    ngx_probe::Logf("ffx-probe: the FidelityFX loader lacks the API exports");
    blocked = true;
    return;
  }
  if (DetourTransactionBegin() != NO_ERROR) return;
  DetourUpdateThread(GetCurrentThread());
  const bool ok = DetourAttach(reinterpret_cast<void**>(&real_create), reinterpret_cast<void*>(&HookCreate)) == NO_ERROR &&
                  DetourAttach(reinterpret_cast<void**>(&real_configure), reinterpret_cast<void*>(&HookConfigure)) == NO_ERROR &&
                  DetourAttach(reinterpret_cast<void**>(&real_dispatch), reinterpret_cast<void*>(&HookDispatch)) == NO_ERROR;
  if (!ok || DetourTransactionCommit() != NO_ERROR) {
    if (!ok) DetourTransactionAbort();
    ngx_probe::Logf("ffx-probe: hooking the FidelityFX loader failed");
    blocked = true;
    return;
  }
  installed = true;
  ngx_probe::Logf("ffx-probe: hooked ffxCreateContext, ffxConfigure and ffxDispatch");
}

// After the NR pass landed on the graded frame (`answer`, UNORDERED_ACCESS, left so): its copy for frame generation,
// when the game wants one.
inline void Capture(ID3D12GraphicsCommandList* cmd, ID3D12Resource* answer) {
  if (!wanted.load() || enabled.load() == 0 || answer == nullptr) return;
  const D3D12_RESOURCE_DESC ad = answer->GetDesc();
  if (hudless != nullptr) {
    const D3D12_RESOURCE_DESC hd = hudless->GetDesc();
    if (hd.Width != ad.Width || hd.Height != ad.Height || hd.Format != ad.Format) {
      hudless->Release();   // (a resize: the GPU is idle across it)
      hudless = nullptr;
    }
  }
  if (hudless == nullptr) {
    ID3D12Device* device = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return;
    D3D12_HEAP_PROPERTIES heap = {D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC d = ad;
    d.Flags = D3D12_RESOURCE_FLAG_NONE;
    const HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d,
                                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
                                                       IID_PPV_ARGS(&hudless));
    device->Release();
    if (FAILED(hr)) {
      hudless = nullptr;
      return;
    }
    ngx_probe::Logf("ffx-probe: the NR pass's HUD-less image for frame generation: %llux%u format %u (the game's: %u)",
                    (unsigned long long)ad.Width, ad.Height, (unsigned)ad.Format, (unsigned)game_format);
  }
  D3D12_RESOURCE_BARRIER b[2] = {};
  b[0].Type = b[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  b[0].Transition = {answer, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COPY_SOURCE};
  b[1].Transition = {hudless, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                     D3D12_RESOURCE_STATE_COPY_DEST};
  cmd->ResourceBarrier(2, b);
  cmd->CopyResource(hudless, answer);
  for (auto& t : b) std::swap(t.Transition.StateBefore, t.Transition.StateAfter);
  cmd->ResourceBarrier(2, b);
  fresh = true;
}

}  // namespace ffx_probe
