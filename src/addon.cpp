// DLSSNR Linux — ReShade addon entry point.
// Registers the overlay and present hook, logs swapchain ground truth
// (color space, back buffer format, dimensions, Wine/Proton detection),
// and installs the NGX CreateFeature/EvaluateFeature interception
// (ngx_probe.hpp) that the runner and colour bridge are driven by.

#include <winsock2.h>  // before windows.h: fov.hpp streams the gaze point over UDP
#include <windows.h>

#include <cstdio>
#include <cstdlib>

#pragma comment(lib, "user32")  // GetAsyncKeyState for the F10 toggle

// ImGui before reshade.hpp so the overlay function table lights up; ImTextureID must be 8 bytes
// to match reshade::api::resource_view.
#define ImTextureID ImU64
#define ImDrawIdx unsigned int
#include <imgui.h>

#include <reshade.hpp>

#include "ngx_probe.hpp"

namespace {

const char* ColorSpaceName(reshade::api::color_space cs) {
  switch (cs) {
    case reshade::api::color_space::unknown: return "unknown (treat as SDR sRGB)";
    case reshade::api::color_space::srgb: return "srgb (SDR)";
    case reshade::api::color_space::scrgb: return "scrgb (linear HDR BT.709)";
    case reshade::api::color_space::hdr10_pq: return "hdr10_pq (PQ BT.2020)";
    case reshade::api::color_space::hdr10_hlg: return "hdr10_hlg (HLG BT.2020)";
    default: return "unrecognized";
  }
}

void LogWineVersion() {
  char buf[256];
  const auto* ntdll = GetModuleHandleW(L"ntdll.dll");
  const auto wine_get_version = reinterpret_cast<const char* (*)()>(
      GetProcAddress((HMODULE)ntdll, "wine_get_version"));
  if (wine_get_version != nullptr) {
    std::snprintf(buf, sizeof(buf), "environment: Wine/Proton %s", wine_get_version());
  } else {
    std::snprintf(buf, sizeof(buf), "environment: native Windows");
  }
  reshade::log::message(reshade::log::level::info, buf);
}

// get_resource_desc returns a large struct by value from a virtual method —
// only ABI-safe because we build with clang targeting x86_64-pc-windows-msvc
// (matching MSVC-built ReShade). A MinGW GCC build crashes on this call.
void OnInitSwapchain(reshade::api::swapchain* swapchain, bool resize) {
  auto* device = swapchain->get_device();
  const auto back_buffer = swapchain->get_back_buffer(0);
  const auto desc = device->get_resource_desc(back_buffer);
  const auto color_space = swapchain->get_color_space();

  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "swapchain %s: %ux%u format=%u color_space=%u [%s] buffers=%u api=%u",
                resize ? "resized" : "created",
                desc.texture.width, desc.texture.height,
                static_cast<unsigned>(desc.texture.format),
                static_cast<unsigned>(color_space), ColorSpaceName(color_space),
                swapchain->get_back_buffer_count(),
                static_cast<unsigned>(device->get_api()));
  reshade::log::message(reshade::log::level::info, buf);
}

// The NGX module (_nvngx.dll / nvngx_dlss.dll) loads well after the addon,
// so retry the Detours attach on every present until it lands. Detours
// patches function bodies, so attaching after the game cached the function
// pointers still works.
void PollLiveSettings();  // below, with the rest of the settings code

void OnPresent(reshade::api::command_queue* queue, reshade::api::swapchain*,
               const reshade::api::rect*, const reshade::api::rect*,
               uint32_t, const reshade::api::rect*) {
  ngx_probe::TryInstall();

  // F10 toggles the NR pass, edge-triggered, for instant A/B comparison.
  static bool f10_was_down = false;
  const bool f10_down = (GetAsyncKeyState(VK_F10) & 0x8000) != 0;
  if (f10_down && !f10_was_down) nr_runner::SetEnabled(!nr_runner::IsEnabled());
  f10_was_down = f10_down;

  // The pass's GPU timers need the queue's timestamp rate; any D3D12 queue on the device has it.
  if (diag::timestamp_freq.load(std::memory_order_relaxed) == 0 && queue != nullptr &&
      queue->get_device()->get_api() == reshade::api::device_api::d3d12) {
    UINT64 freq = 0;
    auto* native = reinterpret_cast<ID3D12CommandQueue*>(queue->get_native());
    if (native != nullptr && SUCCEEDED(native->GetTimestampFrequency(&freq))) diag::timestamp_freq = freq;
  }

  PollLiveSettings();
}

// --- settings persistence -------------------------------------------------
// The tuning lives in ReShade's own config (ReShade.ini beside the game exe),
// so it survives a game restart. Loaded once at attach — early enough, since
// the NR feature is only built on the first DLSS evaluate — and saved from the
// overlay whenever a control changes (ReShade batches the actual disk flush).
// The Enabled flag and the debug view are deliberately not persisted: both are
// transient A/B tools, and coming back up in a debug view would read as broken.

constexpr char kConfigSection[] = "ADDON_DLSSNR_LINUX";

float Clamp(float v, float min, float max) { return v < min ? min : (v > max ? max : v); }

void LoadSettings() {
  auto& s = nr_runner::s;
  reshade::get_config_value(nullptr, kConfigSection, "DetailStrength", s.transfer);
  reshade::get_config_value(nullptr, kConfigSection, "ColourStrength", s.colour_strength);
  reshade::get_config_value(nullptr, kConfigSection, "HighlightGuard", s.max_ratio);
  reshade::get_config_value(nullptr, kConfigSection, "WhitePointScale", s.wp_scale);
  reshade::get_config_value(nullptr, kConfigSection, "Intensity", s.intensity);
  reshade::get_config_value(nullptr, kConfigSection, "StructureIntensity", s.local_structure);
  reshade::get_config_value(nullptr, kConfigSection, "GlobalIntensity", s.local_tone);
  reshade::get_config_value(nullptr, kConfigSection, "Preset", s.preset);
  reshade::get_config_value(nullptr, kConfigSection, "Style", s.style);
  reshade::get_config_value(nullptr, kConfigSection, "MaskBlendMode", s.mask_blend_mode);
  reshade::get_config_value(nullptr, kConfigSection, "UICorrection", s.ui_correction);
  reshade::get_config_value(nullptr, kConfigSection, "UseAutoMask", s.auto_mask);
  reshade::get_config_value(nullptr, kConfigSection, "SkinStructureStrength", s.skin_structure);
  reshade::get_config_value(nullptr, kConfigSection, "GlobalToneStrength", s.global_tone);
  reshade::get_config_value(nullptr, kConfigSection, "ScalingRatio", s.scaling_ratio);
  reshade::get_config_value(nullptr, kConfigSection, "SplitGlobalPreset", s.split_global_preset);
  reshade::get_config_value(nullptr, kConfigSection, "SplitGlobalIntensity", s.split_global_intensity);
  reshade::get_config_value(nullptr, kConfigSection, "SplitGlobalStructureIntensity", s.split_global_structure);
  reshade::get_config_value(nullptr, kConfigSection, "SplitGlobalGlobalIntensity", s.split_global_tone);
  reshade::get_config_value(nullptr, kConfigSection, "SplitGlobalStyle", s.split_global_style);
  reshade::get_config_value(nullptr, kConfigSection, "BeforeUpscale", s.before_upscale);

  auto& f = fov::settings;
  reshade::get_config_value(nullptr, kConfigSection, "Foveation", f.enabled);
  reshade::get_config_value(nullptr, kConfigSection, "FoveaWidth", f.width);
  reshade::get_config_value(nullptr, kConfigSection, "FoveaHeight", f.height);
  reshade::get_config_value(nullptr, kConfigSection, "FoveaFalloff", f.falloff);
  reshade::get_config_value(nullptr, kConfigSection, "FoveaEdge", f.edge);
  reshade::get_config_value(nullptr, kConfigSection, "GazeSource", f.gaze_source);
  reshade::get_config_value(nullptr, kConfigSection, "GazePort", f.gaze_port);
  reshade::get_config_value(nullptr, kConfigSection, "GazeSmoothMs", f.gaze_smooth_ms);
  reshade::get_config_value(nullptr, kConfigSection, "GazeJumpReset", f.jump_reset);
  reshade::get_config_value(nullptr, kConfigSection, "FoveaSplit", f.split);
  reshade::get_config_value(nullptr, kConfigSection, "GlobalScale", f.global_scale);
  reshade::get_config_value(nullptr, kConfigSection, "GradeBlur", f.grade_blur);
  reshade::get_config_value(nullptr, kConfigSection, "Relight", f.relight);
  reshade::get_config_value(nullptr, kConfigSection, "RelightDarkenOnly", f.relight_mode);
  reshade::get_config_value(nullptr, kConfigSection, "SkyNoRelight", f.sky_no_relight);
  reshade::get_config_value(nullptr, kConfigSection, "SplitCalls", f.split_calls);
  reshade::get_config_value(nullptr, kConfigSection, "ToneSmoothing", f.tone_smoothing);
  reshade::get_config_value(nullptr, kConfigSection, "ToneDeadband", f.tone_deadband);
  reshade::get_config_value(nullptr, kConfigSection, "GlobalEvery", f.global_every);
  reshade::get_config_value(nullptr, kConfigSection, "FoveaDebugView", f.fovea_debug);
  reshade::get_config_value(nullptr, kConfigSection, "PeripheryDebugView", f.periphery_debug);
  reshade::get_config_value(nullptr, kConfigSection, "PeripheryPass", f.periphery_pass);
  reshade::get_config_value(nullptr, kConfigSection, "NREngine", s.engine);
  reshade::get_config_value(nullptr, kConfigSection, "CropSnap", f.crop_snap);
  reshade::get_config_value(nullptr, kConfigSection, "EngineHistory", s.engine_history);

  // A hand-edited ini must not push values past what the overlay allows.
  s.transfer = Clamp(s.transfer, 0.0f, 1.5f);
  s.colour_strength = Clamp(s.colour_strength, 0.0f, 1.0f);
  s.max_ratio = Clamp(s.max_ratio, 1.0f, 8.0f);
  s.wp_scale = Clamp(s.wp_scale, 0.1f, 4.0f);
  s.intensity = Clamp(s.intensity, 0.0f, 2.0f);
  s.local_structure = Clamp(s.local_structure, 0.0f, 2.0f);
  s.local_tone = Clamp(s.local_tone, 0.0f, 2.0f);
  if (s.preset < 0 || s.preset > 7) s.preset = 0;
  if (s.style < 0 || s.style > 2) s.style = 0;
  s.split_global_intensity = Clamp(s.split_global_intensity, 0.0f, 2.0f);
  s.split_global_structure = Clamp(s.split_global_structure, 0.0f, 2.0f);
  s.split_global_tone = Clamp(s.split_global_tone, 0.0f, 2.0f);
  if (s.split_global_style < 0 || s.split_global_style > 2) s.split_global_style = 0;
  if (s.split_global_preset < -1 || s.split_global_preset > 7) s.split_global_preset = 1;
  if (f.fovea_debug < 0 || f.fovea_debug > 3) f.fovea_debug = 0;
  if (f.periphery_debug < 0 || f.periphery_debug > 3) f.periphery_debug = 0;
  f.width = Clamp(f.width, 0.05f, 1.0f);
  f.height = Clamp(f.height, 0.05f, 1.0f);
  f.falloff = Clamp(f.falloff, 0.0f, 0.5f);
  f.edge = Clamp(f.edge, 0.25f, 1.0f);
  if (f.gaze_source < 0 || f.gaze_source > 1) f.gaze_source = 0;
  if (f.gaze_port < 1024 || f.gaze_port > 65535) f.gaze_port = 47620;
  f.gaze_smooth_ms = Clamp(f.gaze_smooth_ms, 0.0f, 500.0f);
  f.jump_reset = Clamp(f.jump_reset, 0.02f, 2.0f);
  f.global_scale = Clamp(f.global_scale, 0.1f, 0.5f);
  f.grade_blur = Clamp(f.grade_blur, 0.0f, 16.0f);
  f.relight = Clamp(f.relight, 0.0f, 1.0f);
  f.relight_mode = f.relight_mode != 0 ? 1 : 0;
  if (f.split_calls != 1) f.split_calls = 2;
  f.tone_smoothing = Clamp(f.tone_smoothing, 0.0f, 0.95f);
  f.tone_deadband = Clamp(f.tone_deadband, 0.0f, 0.1f);
  if (f.global_every < 1 || f.global_every > 4) f.global_every = 1;
}

void SaveSettings() {
  auto& s = nr_runner::s;
  reshade::set_config_value(nullptr, kConfigSection, "DetailStrength", s.transfer);
  reshade::set_config_value(nullptr, kConfigSection, "ColourStrength", s.colour_strength);
  reshade::set_config_value(nullptr, kConfigSection, "HighlightGuard", s.max_ratio);
  reshade::set_config_value(nullptr, kConfigSection, "WhitePointScale", s.wp_scale);
  reshade::set_config_value(nullptr, kConfigSection, "Intensity", s.intensity);
  reshade::set_config_value(nullptr, kConfigSection, "StructureIntensity", s.local_structure);
  reshade::set_config_value(nullptr, kConfigSection, "GlobalIntensity", s.local_tone);
  reshade::set_config_value(nullptr, kConfigSection, "Preset", s.preset);
  reshade::set_config_value(nullptr, kConfigSection, "Style", s.style);
  reshade::set_config_value(nullptr, kConfigSection, "MaskBlendMode", s.mask_blend_mode);
  reshade::set_config_value(nullptr, kConfigSection, "UICorrection", s.ui_correction);
  reshade::set_config_value(nullptr, kConfigSection, "UseAutoMask", s.auto_mask);
  reshade::set_config_value(nullptr, kConfigSection, "SkinStructureStrength", s.skin_structure);
  reshade::set_config_value(nullptr, kConfigSection, "GlobalToneStrength", s.global_tone);
  reshade::set_config_value(nullptr, kConfigSection, "ScalingRatio", s.scaling_ratio);
  reshade::set_config_value(nullptr, kConfigSection, "SplitGlobalPreset", s.split_global_preset);
  reshade::set_config_value(nullptr, kConfigSection, "SplitGlobalIntensity", s.split_global_intensity);
  reshade::set_config_value(nullptr, kConfigSection, "SplitGlobalStructureIntensity", s.split_global_structure);
  reshade::set_config_value(nullptr, kConfigSection, "SplitGlobalGlobalIntensity", s.split_global_tone);
  reshade::set_config_value(nullptr, kConfigSection, "SplitGlobalStyle", s.split_global_style);
  reshade::set_config_value(nullptr, kConfigSection, "BeforeUpscale", s.before_upscale);

  const auto& f = fov::settings;
  reshade::set_config_value(nullptr, kConfigSection, "Foveation", f.enabled);
  reshade::set_config_value(nullptr, kConfigSection, "FoveaWidth", f.width);
  reshade::set_config_value(nullptr, kConfigSection, "FoveaHeight", f.height);
  reshade::set_config_value(nullptr, kConfigSection, "FoveaFalloff", f.falloff);
  reshade::set_config_value(nullptr, kConfigSection, "FoveaEdge", f.edge);
  reshade::set_config_value(nullptr, kConfigSection, "GazeSource", f.gaze_source);
  reshade::set_config_value(nullptr, kConfigSection, "GazePort", f.gaze_port);
  reshade::set_config_value(nullptr, kConfigSection, "GazeSmoothMs", f.gaze_smooth_ms);
  reshade::set_config_value(nullptr, kConfigSection, "GazeJumpReset", f.jump_reset);
  reshade::set_config_value(nullptr, kConfigSection, "FoveaSplit", f.split);
  reshade::set_config_value(nullptr, kConfigSection, "GlobalScale", f.global_scale);
  reshade::set_config_value(nullptr, kConfigSection, "GradeBlur", f.grade_blur);
  reshade::set_config_value(nullptr, kConfigSection, "Relight", f.relight);
  reshade::set_config_value(nullptr, kConfigSection, "RelightDarkenOnly", f.relight_mode);
  reshade::set_config_value(nullptr, kConfigSection, "SkyNoRelight", f.sky_no_relight);
  reshade::set_config_value(nullptr, kConfigSection, "SplitCalls", f.split_calls);
  reshade::set_config_value(nullptr, kConfigSection, "ToneSmoothing", f.tone_smoothing);
  reshade::set_config_value(nullptr, kConfigSection, "ToneDeadband", f.tone_deadband);
  reshade::set_config_value(nullptr, kConfigSection, "GlobalEvery", f.global_every);
  reshade::set_config_value(nullptr, kConfigSection, "FoveaDebugView", f.fovea_debug);
  reshade::set_config_value(nullptr, kConfigSection, "PeripheryDebugView", f.periphery_debug);
  reshade::set_config_value(nullptr, kConfigSection, "PeripheryPass", f.periphery_pass);
  reshade::set_config_value(nullptr, kConfigSection, "NREngine", s.engine);
  reshade::set_config_value(nullptr, kConfigSection, "CropSnap", f.crop_snap);
  reshade::set_config_value(nullptr, kConfigSection, "EngineHistory", s.engine_history);
}

// --- live settings from another process ----------------------------------
// When this add-on runs inside a helper process (DLSS5-Feeder's host64), its overlay lives in a
// window the player never looks at. The helper's parent can instead write the same settings to
// dlssnr-linux-live.ini beside this exe; it is polled twice a second and whatever it holds is
// applied and saved like an overlay edit. A file of its own, not ReShade.ini: ReShade caches that
// one and would write its stale copy back over another process's edits.

void LiveIniPath(char* out) {
  GetModuleFileNameA(nullptr, out, MAX_PATH);
  if (char* slash = strrchr(out, '\\')) strcpy_s(slash + 1, MAX_PATH - (slash + 1 - out), "dlssnr-linux-live.ini");
}

float LiveFloat(const char* path, const char* key, float current) {
  char buf[64];
  GetPrivateProfileStringA("DLSSNR", key, "", buf, sizeof(buf), path);
  return buf[0] != 0 ? (float)atof(buf) : current;
}

int LiveInt(const char* path, const char* key, int current) {
  return (int)GetPrivateProfileIntA("DLSSNR", key, current, path);
}

void PollLiveSettings() {
  static ULONGLONG next_check = 0;
  static FILETIME last_write = {};
  const ULONGLONG now = GetTickCount64();
  if (now < next_check) return;
  next_check = now + 500;

  char path[MAX_PATH];
  LiveIniPath(path);
  WIN32_FILE_ATTRIBUTE_DATA attr;
  if (!GetFileAttributesExA(path, GetFileExInfoStandard, &attr)) return;
  if (CompareFileTime(&attr.ftLastWriteTime, &last_write) == 0) return;
  last_write = attr.ftLastWriteTime;

  auto& s = nr_runner::s;
  auto& f = fov::settings;
  const float old_model[3] = {s.intensity, s.local_structure, s.local_tone};
  const int old_preset = s.preset, old_style = s.style;
  const int old_ints[4] = {s.mask_blend_mode, s.ui_correction, s.auto_mask, s.split_global_preset};
  const float old_floats[3] = {s.skin_structure, s.global_tone, s.scaling_ratio};
  const float old_periphery[4] = {s.split_global_intensity, s.split_global_structure, s.split_global_tone,
                                  (float)s.split_global_style};

  nr_runner::SetEnabled(LiveInt(path, "Enabled", nr_runner::IsEnabled() ? 1 : 0) != 0);
  s.debug_view = LiveInt(path, "DebugView", s.debug_view);
  s.transfer = LiveFloat(path, "DetailStrength", s.transfer);
  s.colour_strength = LiveFloat(path, "ColourStrength", s.colour_strength);
  s.max_ratio = LiveFloat(path, "HighlightGuard", s.max_ratio);
  s.wp_scale = LiveFloat(path, "WhitePointScale", s.wp_scale);
  s.intensity = LiveFloat(path, "Intensity", s.intensity);
  s.local_structure = LiveFloat(path, "StructureIntensity", s.local_structure);
  s.local_tone = LiveFloat(path, "GlobalIntensity", s.local_tone);
  s.preset = LiveInt(path, "Preset", s.preset);
  s.style = LiveInt(path, "Style", s.style);
  s.mask_blend_mode = LiveInt(path, "MaskBlendMode", s.mask_blend_mode);
  s.ui_correction = LiveInt(path, "UICorrection", s.ui_correction);
  s.auto_mask = LiveInt(path, "UseAutoMask", s.auto_mask);
  s.skin_structure = LiveFloat(path, "SkinStructureStrength", s.skin_structure);
  s.global_tone = LiveFloat(path, "GlobalToneStrength", s.global_tone);
  s.scaling_ratio = LiveFloat(path, "ScalingRatio", s.scaling_ratio);
  s.split_global_preset = LiveInt(path, "SplitGlobalPreset", s.split_global_preset);
  s.split_global_intensity = LiveFloat(path, "SplitGlobalIntensity", s.split_global_intensity);
  s.split_global_structure = LiveFloat(path, "SplitGlobalStructureIntensity", s.split_global_structure);
  s.split_global_tone = LiveFloat(path, "SplitGlobalGlobalIntensity", s.split_global_tone);
  s.split_global_style = LiveInt(path, "SplitGlobalStyle", s.split_global_style);
  s.before_upscale = LiveInt(path, "BeforeUpscale", s.before_upscale ? 1 : 0) != 0;
  f.enabled = LiveInt(path, "Foveation", f.enabled ? 1 : 0) != 0;
  f.width = LiveFloat(path, "FoveaWidth", f.width);
  f.height = LiveFloat(path, "FoveaHeight", f.height);
  f.falloff = LiveFloat(path, "FoveaFalloff", f.falloff);
  f.edge = LiveFloat(path, "FoveaEdge", f.edge);
  f.gaze_source = LiveInt(path, "GazeSource", f.gaze_source);
  f.gaze_port = LiveInt(path, "GazePort", f.gaze_port);
  f.gaze_smooth_ms = LiveFloat(path, "GazeSmoothMs", f.gaze_smooth_ms);
  f.split = LiveInt(path, "FoveaSplit", f.split ? 1 : 0) != 0;
  f.global_scale = LiveFloat(path, "GlobalScale", f.global_scale);
  f.grade_blur = LiveFloat(path, "GradeBlur", f.grade_blur);
  f.relight = LiveFloat(path, "Relight", f.relight);
  f.relight_mode = LiveInt(path, "RelightDarkenOnly", f.relight_mode);
  f.sky_no_relight = LiveInt(path, "SkyNoRelight", f.sky_no_relight ? 1 : 0) != 0;
  f.split_calls = LiveInt(path, "SplitCalls", f.split_calls);
  f.tone_smoothing = LiveFloat(path, "ToneSmoothing", f.tone_smoothing);
  f.tone_deadband = LiveFloat(path, "ToneDeadband", f.tone_deadband);
  f.global_every = LiveInt(path, "GlobalEvery", f.global_every);
  f.fovea_debug = LiveInt(path, "FoveaDebugView", f.fovea_debug);
  f.periphery_debug = LiveInt(path, "PeripheryDebugView", f.periphery_debug);
  f.periphery_pass = LiveInt(path, "PeripheryPass", f.periphery_pass ? 1 : 0) != 0;
  s.engine = LiveInt(path, "NREngine", s.engine ? 1 : 0) != 0;
  f.crop_snap = LiveInt(path, "CropSnap", f.crop_snap);
  if (f.crop_snap < 0 || f.crop_snap > 256) f.crop_snap = 32;
  s.engine_history = Clamp(LiveFloat(path, "EngineHistory", s.engine_history), 0.0f, 1.0f);
  s.engine_motion = Clamp(LiveFloat(path, "EngineMotion", s.engine_motion), 0.0f, 2.0f);
  diag::dump_requested = LiveInt(path, "DumpFrame", diag::dump_requested);
  diag::dump_burst = LiveInt(path, "DumpBurst", diag::dump_burst);

  // The same bounds a hand-edited ReShade.ini gets.
  s.transfer = Clamp(s.transfer, 0.0f, 1.5f);
  s.colour_strength = Clamp(s.colour_strength, 0.0f, 1.0f);
  s.max_ratio = Clamp(s.max_ratio, 1.0f, 8.0f);
  s.wp_scale = Clamp(s.wp_scale, 0.1f, 4.0f);
  s.intensity = Clamp(s.intensity, 0.0f, 2.0f);
  s.local_structure = Clamp(s.local_structure, 0.0f, 2.0f);
  s.local_tone = Clamp(s.local_tone, 0.0f, 2.0f);
  if (s.preset < 0 || s.preset > 7) s.preset = 0;
  if (s.style < 0 || s.style > 2) s.style = 0;
  s.split_global_intensity = Clamp(s.split_global_intensity, 0.0f, 2.0f);
  s.split_global_structure = Clamp(s.split_global_structure, 0.0f, 2.0f);
  s.split_global_tone = Clamp(s.split_global_tone, 0.0f, 2.0f);
  if (s.split_global_style < 0 || s.split_global_style > 2) s.split_global_style = 0;
  if (s.split_global_preset < -1 || s.split_global_preset > 7) s.split_global_preset = 1;
  if (f.fovea_debug < 0 || f.fovea_debug > 3) f.fovea_debug = 0;
  if (f.periphery_debug < 0 || f.periphery_debug > 3) f.periphery_debug = 0;
  if (s.debug_view < 0 || s.debug_view > 4) s.debug_view = 0;
  f.width = Clamp(f.width, 0.05f, 1.0f);
  f.height = Clamp(f.height, 0.05f, 1.0f);
  f.falloff = Clamp(f.falloff, 0.0f, 0.5f);
  f.edge = Clamp(f.edge, 0.25f, 1.0f);
  if (f.gaze_source < 0 || f.gaze_source > 1) f.gaze_source = 0;
  if (f.gaze_port < 1024 || f.gaze_port > 65535) f.gaze_port = 47620;
  f.gaze_smooth_ms = Clamp(f.gaze_smooth_ms, 0.0f, 500.0f);
  f.global_scale = Clamp(f.global_scale, 0.1f, 0.5f);
  f.grade_blur = Clamp(f.grade_blur, 0.0f, 16.0f);
  f.relight = Clamp(f.relight, 0.0f, 1.0f);
  f.relight_mode = f.relight_mode != 0 ? 1 : 0;
  if (f.split_calls != 1) f.split_calls = 2;
  f.tone_smoothing = Clamp(f.tone_smoothing, 0.0f, 0.95f);
  f.tone_deadband = Clamp(f.tone_deadband, 0.0f, 0.1f);
  if (f.global_every < 1 || f.global_every > 4) f.global_every = 1;

  // The model reads these only when its feature is built; foveation geometry rebuilds by itself.
  if (s.intensity != old_model[0] || s.local_structure != old_model[1] ||
      s.local_tone != old_model[2] || s.preset != old_preset || s.style != old_style ||
      s.mask_blend_mode != old_ints[0] || s.ui_correction != old_ints[1] || s.auto_mask != old_ints[2] || s.split_global_preset != old_ints[3] ||
      s.skin_structure != old_floats[0] || s.global_tone != old_floats[1] || s.scaling_ratio != old_floats[2] ||
      s.split_global_intensity != old_periphery[0] || s.split_global_structure != old_periphery[1] ||
      s.split_global_tone != old_periphery[2] || (float)s.split_global_style != old_periphery[3])
    nr_runner::ApplyModelSettings();
  SaveSettings();
  ngx_probe::Log("nr-fwd: live settings applied from dlssnr-linux-live.ini");
}

// A slider with a reset-to-default button on its right.
bool SliderWithReset(const char* label, float* value, float min, float max, const char* fmt,
                     float default_value) {
  ImGui::PushID(label);
  bool changed = ImGui::SliderFloat(label, value, min, max, fmt);
  ImGui::SameLine();
  if (ImGui::SmallButton("reset")) {
    *value = default_value;
    changed = true;
  }
  ImGui::PopID();
  return changed;
}

// The settings panel, under ReShade's Add-ons tab. Compose controls apply live; the model's own
// settings are read once at feature creation, so they take a rebuild (the Apply button).
void OnDrawOverlay(reshade::api::effect_runtime*) {
  auto& s = nr_runner::s;

  bool enabled = nr_runner::IsEnabled();
  if (ImGui::Checkbox("Enable Neural Rendering (F10)", &enabled)) nr_runner::SetEnabled(enabled);

  bool changed = false;
  changed |= SliderWithReset("Detail strength", &s.transfer, 0.0f, 1.5f, "%.2f", 1.0f);
  changed |= SliderWithReset("Colour strength", &s.colour_strength, 0.0f, 1.0f, "%.2f", 0.0f);
  changed |= SliderWithReset("Highlight guard", &s.max_ratio, 1.0f, 8.0f, "%.1fx", 2.0f);
  changed |= SliderWithReset("White point scale", &s.wp_scale, 0.1f, 4.0f, "%.2f", 1.0f);
  if (s.wp_ema > 0.0f) {
    ImGui::Text("Measured white point: %.4f", s.wp_ema);
  } else {
    ImGui::TextUnformatted("Measured white point: (measuring...)");
  }

  ImGui::Combo("Debug view", &s.debug_view,
               "Off\0What the model sees\0Model answer\0Difference x20\0Fovea map (foveated NR)\0");

  ImGui::Separator();
  ImGui::TextUnformatted("Model settings (need Apply, rebuilds the feature)");
  // The three DLSSNR intensities, as in the RenoDX DLSS5 addon. NR intensity is the model's overall
  // hand; Structure intensity drives the fine detail it synthesises; Global intensity shapes the
  // broad local contrast/tone. All default to 1.0 (the model's neutral).
  changed |= SliderWithReset("NR intensity", &s.intensity, 0.0f, 2.0f, "%.2f", 1.0f);
  changed |= SliderWithReset("Structure intensity", &s.local_structure, 0.0f, 2.0f, "%.2f", 1.0f);
  changed |= SliderWithReset("Global intensity", &s.local_tone, 0.0f, 2.0f, "%.2f", 1.0f);
  changed |= ImGui::SliderInt("Preset", &s.preset, 0, 7);
  ImGui::SameLine();
  ImGui::PushID("preset-reset");
  if (ImGui::SmallButton("reset")) {
    s.preset = 0;
    changed = true;
  }
  ImGui::PopID();

  // The model's own processing profiles (DLSSNR.Style). Names from community testing via
  // OptiScaler_DLSSNR -- NVIDIA ships none. Default boosts local contrast hardest and can look
  // stylised; Natural is the same detail with a gentler hand; Cinematic tones down the shine for
  // a film-like look.
  if (s.style > 2) s.style = 2;
  changed |= ImGui::Combo("Style", &s.style, "Default (standard)\0Natural\0Cinematic\0");
  ImGui::SameLine();
  ImGui::PushID("style-reset");
  if (ImGui::SmallButton("reset")) {
    s.style = 0;
    changed = true;
  }
  ImGui::PopID();

  if (ImGui::Button("Apply model settings")) nr_runner::ApplyModelSettings();

  // Only differs from the default when the game's DLSS really upscales (input smaller than output).
  changed |= ImGui::Checkbox("Run before the upscale (the model works at render resolution)",
                             &s.before_upscale);

  // Geometry changes rebuild the feature at its new size by themselves (the next frame); the
  // gaze controls apply live.
  ImGui::Separator();
  ImGui::TextUnformatted("Foveation (the model runs on a squeezed frame)");
  auto& f = fov::settings;
  changed |= ImGui::Checkbox("Foveated NR", &f.enabled);
  changed |= SliderWithReset("Fovea width", &f.width, 0.05f, 1.0f, "%.2f of the frame", 0.40f);
  changed |= SliderWithReset("Fovea height", &f.height, 0.05f, 1.0f, "%.2f of the frame", 0.50f);
  changed |= SliderWithReset("Falloff", &f.falloff, 0.0f, 0.5f, "%.2f of the height", 0.10f);
  changed |= SliderWithReset("Edge resolution", &f.edge, 0.25f, 1.0f, "%.2f", 0.50f);
  changed |= ImGui::Checkbox("Split: full-res crop + low-res frame for the tone", &f.split);
  if (f.split) {
    changed |= SliderWithReset("Low-res frame", &f.global_scale, 0.1f, 0.5f, "%.2f of the frame", 0.25f);
    changed |= SliderWithReset("Tone blur", &f.grade_blur, 0.0f, 16.0f, "%.1f px", 1.0f);
    changed |= SliderWithReset("Relighting kept", &f.relight, 0.0f, 1.0f, "%.2f", 1.0f);
    bool darken = f.relight_mode != 0;
    if (ImGui::Checkbox("Relighting may only darken (no colour; ambient occlusion-like)", &darken)) {
      f.relight_mode = darken ? 1 : 0;
      changed = true;
    }
    changed |= ImGui::Checkbox("No relighting on the sky (depth at the far plane)", &f.sky_no_relight);
    bool two = f.split_calls == 2;
    if (ImGui::Checkbox("A model call per region (independent grading)", &two)) {
      f.split_calls = two ? 2 : 1;
      changed = true;
    }
    changed |= SliderWithReset("Tone smoothing", &f.tone_smoothing, 0.0f, 0.95f, "%.2f", 0.95f);
    changed |= SliderWithReset("Tone dead band", &f.tone_deadband, 0.0f, 0.1f, "%.3f", 0.01f);
    changed |= ImGui::SliderInt("Whole-frame pass every", &f.global_every, 1, 4, "%d frames");
    if (f.split_calls == 2) {
      // Each pass is its own feature: its own network and hand, and a debug view of it alone.
      static const char kPassViews[] = "Off\0What the model sees\0Model answer\0Difference x20\0";
      changed |= ImGui::Checkbox("Our engine (precompiled Ampere cubins) instead of NGX", &s.engine);
      if (s.engine)
        ImGui::Text("Engine: %s -- regions at %ux%u and %ux%u; NGX runs until it is ready", nr_engine::StatusText(),
                    nr_engine::g.progs[0].field_w, nr_engine::g.progs[0].field_h,
                    nr_engine::g.progs[nr_engine::g.count > 1 ? 1 : 0].field_w,
                    nr_engine::g.progs[nr_engine::g.count > 1 ? 1 : 0].field_h);
      if (s.engine)
        changed |= SliderWithReset("Engine history (1 = NVIDIA's)", &s.engine_history, 0.0f, 1.0f, "%.2f", 1.0f);
      ImGui::TextUnformatted("Fovea pass (full-res crop): the model settings above");
      changed |= ImGui::Combo("Fovea pass debug view", &f.fovea_debug, kPassViews);
      ImGui::TextUnformatted("Periphery pass (low-res whole frame)");
      changed |= ImGui::Checkbox("Run the periphery pass (off: the game's own frame outside the crop)",
                                 &f.periphery_pass);
      changed |= ImGui::Combo("Periphery pass debug view", &f.periphery_debug, kPassViews);
      bool model = false;
      model |= SliderWithReset("Periphery NR intensity", &s.split_global_intensity, 0.0f, 2.0f, "%.2f", 1.0f);
      model |= SliderWithReset("Periphery structure intensity", &s.split_global_structure, 0.0f, 2.0f, "%.2f", 1.0f);
      model |= SliderWithReset("Periphery global intensity", &s.split_global_tone, 0.0f, 2.0f, "%.2f", 1.0f);
      model |= ImGui::SliderInt("Periphery preset", &s.split_global_preset, -1, 7, s.split_global_preset < 0 ? "same as fovea" : "%d");
      model |= ImGui::Combo("Periphery style", &s.split_global_style, "Default (standard)\0Natural\0Cinematic\0");
      if (model) nr_runner::ApplyModelSettings();
      changed |= model;
    }
  }
  changed |= ImGui::Combo("Fovea centre", &f.gaze_source, "Fixed (frame centre)\0Gaze over UDP\0");
  if (f.gaze_source == 1) {
    changed |= ImGui::InputInt("Gaze UDP port", &f.gaze_port);
    changed |= SliderWithReset("Gaze smoothing", &f.gaze_smooth_ms, 0.0f, 200.0f, "%.0f ms", 30.0f);
    float gx = 0.0f, gy = 0.0f;
    if (fov::GazeLatest(&gx, &gy)) ImGui::Text("Gaze: %.3f, %.3f", gx, gy);
    else ImGui::TextUnformatted("Gaze: no packets in the last 0.5 s (centre used)");
  }
  if (s.frame_w != 0 && s.frame_h != 0) {
    uint32_t fw = 0, fh = 0;
    nr_runner::FeatureSize(&fw, &fh);
    ImGui::Text("Model works on %ux%u of %ux%u (%.0f%% of the pixels)", fw, fh, s.frame_w, s.frame_h,
                100.0 * fw * fh / ((double)s.frame_w * s.frame_h));
  }

  if (changed) SaveSettings();
}

}  // namespace

extern "C" __declspec(dllexport) const char* NAME = "DLSSNR Linux";
extern "C" __declspec(dllexport) const char* DESCRIPTION =
    "Runs NVIDIA DLSS 5 Neural Rendering (NGX feature 18) under Linux/Proton by driving the "
    "game-local snippet directly, with a display-referred colour bridge";

BOOL APIENTRY DllMain(HMODULE h_module, DWORD fdw_reason, LPVOID lp_reserved) {
  switch (fdw_reason) {
    case DLL_PROCESS_ATTACH:
      if (!reshade::register_addon(h_module)) return FALSE;
      reshade::log::message(reshade::log::level::info,
                            "DLSSNR Linux loaded (clang x86_64-pc-windows-msvc, built on Linux)");
      LogWineVersion();
      LoadSettings();
      reshade::register_event<reshade::addon_event::init_swapchain>(OnInitSwapchain);
    nr_runner::poll_live = PollLiveSettings;
    reshade::register_event<reshade::addon_event::present>(OnPresent);
      reshade::register_overlay("DLSSNR Linux", OnDrawOverlay);
      ngx_probe::TryInstall();  // in case NGX is already resident
      break;
    case DLL_PROCESS_DETACH:
      // Non-null lp_reserved: the process is terminating and every other
      // thread is already dead, possibly mid-call under a lock. Unpatching
      // code or waiting on threads here can deadlock the exit path, and the
      // whole address space is going away — do nothing. Null lp_reserved is a
      // real FreeLibrary (ReShade cycles the addon during device creation),
      // where the detours must be removed before this image disappears.
      if (lp_reserved != nullptr) break;
      ngx_probe::Uninstall();
      nr_runner::Shutdown();
      reshade::unregister_overlay("DLSSNR Linux", OnDrawOverlay);
      reshade::unregister_event<reshade::addon_event::present>(OnPresent);
      reshade::unregister_event<reshade::addon_event::init_swapchain>(OnInitSwapchain);
      reshade::unregister_addon(h_module);
      break;
  }
  return TRUE;
}
