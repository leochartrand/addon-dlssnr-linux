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
#include "ffx_probe.hpp"
#include "ui_probe.hpp"
#include "graded_hook.hpp"

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

// Screenshot=<id> in the live ini: the presented image with the game's UI (ReShade's capture), written as
// dlssnr-shot-<id>.raw (the back buffer's own bytes) + .txt header beside the exe, once per new id. Lets a
// test script see menus without touching the desktop.

// DLSS presets by letter (NVSDK_NGX_DLSS_Hint_Render_Preset: A = 1 ... M = 13; 0 the default).
const char* PresetName(int v) {
  static const char* const kNames[] = {"default", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"};
  return v >= 0 && v <= 13 ? kNames[v] : "?";
}
int NextCropPreset(int v) { return v == 0 ? 5 : v == 5 ? 11 : v == 11 ? 13 : 0; }

void SaveSettings();

char toast[128] = {};
LARGE_INTEGER toast_at = {};
void Toast() {
  std::snprintf(toast, sizeof(toast), "SR crop: %s (%s)   NR crop: %s (%s)",
                sr_fovea::mode.load() != 0 ? "on" : "off", PresetName(sr_fovea::crop_preset.load()),
                fov::settings.crop_pass ? "on" : "off",
                fov::settings.crop_on_periphery ? "on the periphery" : "on the raw frame");
  QueryPerformanceCounter(&toast_at);
}
// The note in the corner, drawn every frame for two seconds after a key (the overlay itself may be closed).
void OnReshadeOverlay(reshade::api::effect_runtime*) {
  if (toast[0] == 0) return;
  LARGE_INTEGER now, freq;
  QueryPerformanceCounter(&now);
  QueryPerformanceFrequency(&freq);
  if ((double)(now.QuadPart - toast_at.QuadPart) / (double)freq.QuadPart > 2.0) return;
  ImGui::SetNextWindowPos(ImVec2(40.0f, 40.0f));
  ImGui::SetNextWindowBgAlpha(0.6f);
  ImGui::Begin("##dlssnr-toast", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoInputs |
                   ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoSavedSettings);
  ImGui::TextUnformatted(toast);
  ImGui::End();
}

void OnPresent(reshade::api::command_queue* queue, reshade::api::swapchain* swapchain,
               const reshade::api::rect*, const reshade::api::rect*,
               uint32_t, const reshade::api::rect*) {
  ngx_probe::TryInstall();
  ffx_probe::TryInstall();

  // F10 toggles the NR pass, edge-triggered, for instant A/B comparison.
  static bool f10_was_down = false;
  const bool f10_down = (GetAsyncKeyState(VK_F10) & 0x8000) != 0;
  if (f10_down && !f10_was_down) nr_runner::SetEnabled(!nr_runner::IsEnabled());
  f10_was_down = f10_down;
  // The crop A/B keys (F9 is the game's quickload, F5 its quicksave): F6 the crop's DLSS-SR on/off, F7 its
  // preset (default, E, K, M), F8 the NR crop pass on/off, F11 whether it works on the periphery's result. Each shows a note on screen for two seconds.
  static bool was[4] = {};
  const int keys[4] = {VK_F6, VK_F7, VK_F8, VK_F11};
  for (int i = 0; i < 4; ++i) {
    const bool down = (GetAsyncKeyState(keys[i]) & 0x8000) != 0;
    if (down && !was[i]) {
      if (i == 0) sr_fovea::mode = sr_fovea::mode.load() == 0 ? 1 : 0;
      if (i == 1) sr_fovea::crop_preset = NextCropPreset(sr_fovea::crop_preset.load());
      if (i == 2) fov::settings.crop_pass = !fov::settings.crop_pass;
      if (i == 3) fov::settings.crop_on_periphery = !fov::settings.crop_on_periphery;
      Toast();
      SaveSettings();
    }
    was[i] = down;
  }

  // The pass's GPU timers need the queue's timestamp rate; any D3D12 queue on the device has it.
  if (diag::timestamp_freq.load(std::memory_order_relaxed) == 0 && queue != nullptr &&
      queue->get_device()->get_api() == reshade::api::device_api::d3d12) {
    UINT64 freq = 0;
    auto* native = reinterpret_cast<ID3D12CommandQueue*>(queue->get_native());
    if (native != nullptr && SUCCEEDED(native->GetTimestampFrequency(&freq))) diag::timestamp_freq = freq;
  }

  PollLiveSettings();
  ui_probe::OnPresent(swapchain, nr_runner::s.game_dir);
  graded_hook::OnPresent();
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
  reshade::get_config_value(nullptr, kConfigSection, "SplitGlobalSkinStructure", s.split_global_skin);
  reshade::get_config_value(nullptr, kConfigSection, "DLAA", s.dlaa);
  reshade::get_config_value(nullptr, kConfigSection, "BeforeUpscale", s.before_upscale);
  reshade::get_config_value(nullptr, kConfigSection, "PassOnGradedFrame", s.graded);

  auto& f = fov::settings;
  reshade::get_config_value(nullptr, kConfigSection, "Foveation", f.enabled);
  reshade::get_config_value(nullptr, kConfigSection, "FoveaWidth", f.width);
  reshade::get_config_value(nullptr, kConfigSection, "FoveaHeight", f.height);
  reshade::get_config_value(nullptr, kConfigSection, "FoveaFalloff", f.falloff);
  reshade::get_config_value(nullptr, kConfigSection, "GazeSource", f.gaze_source);
  reshade::get_config_value(nullptr, kConfigSection, "GazePort", f.gaze_port);
  reshade::get_config_value(nullptr, kConfigSection, "GazeSmoothMs", f.gaze_smooth_ms);
  reshade::get_config_value(nullptr, kConfigSection, "GazeJumpReset", f.jump_reset);
  reshade::get_config_value(nullptr, kConfigSection, "GlobalScale", f.global_scale);
  reshade::get_config_value(nullptr, kConfigSection, "GradeBlur", f.grade_blur);
  reshade::get_config_value(nullptr, kConfigSection, "SplitCalls", f.split_calls);
  reshade::get_config_value(nullptr, kConfigSection, "ToneSmoothing", f.tone_smoothing);
  reshade::get_config_value(nullptr, kConfigSection, "ToneDeadband", f.tone_deadband);
  reshade::get_config_value(nullptr, kConfigSection, "CropToneFromPeriphery", f.crop_tone);
  reshade::get_config_value(nullptr, kConfigSection, "GlobalEvery", f.global_every);
  reshade::get_config_value(nullptr, kConfigSection, "FoveaDebugView", f.fovea_debug);
  reshade::get_config_value(nullptr, kConfigSection, "PeripheryDebugView", f.periphery_debug);
  reshade::get_config_value(nullptr, kConfigSection, "PeripheryPass", f.periphery_pass);
  reshade::get_config_value(nullptr, kConfigSection, "CropPass", f.crop_pass);
  reshade::get_config_value(nullptr, kConfigSection, "CropOnPeriphery", f.crop_on_periphery);
  reshade::get_config_value(nullptr, kConfigSection, "SecondPass", s.second_pass);
  {
    int v[5] = {sr_fovea::preset, sr_fovea::mode, sr_fovea::crop_preset, sr_fovea::shift, sr_fovea::fade_in};
    reshade::get_config_value(nullptr, kConfigSection, "SrPreset", v[0]);
    reshade::get_config_value(nullptr, kConfigSection, "SrFovea", v[1]);
    reshade::get_config_value(nullptr, kConfigSection, "SrCropPreset", v[2]);
    reshade::get_config_value(nullptr, kConfigSection, "SrShift", v[3]);
    reshade::get_config_value(nullptr, kConfigSection, "SrFadeIn", v[4]);
    sr_fovea::preset = v[0];
    sr_fovea::mode = v[1];
    sr_fovea::crop_preset = v[2];
    sr_fovea::shift = v[3];
    sr_fovea::fade_in = v[4];
  }
  reshade::get_config_value(nullptr, kConfigSection, "NREngine", s.engine);
  {
    size_t n = sizeof(s.engine_crop);
    reshade::get_config_value(nullptr, kConfigSection, "EngineCrop", s.engine_crop, &n);
    n = sizeof(s.engine_low);
    reshade::get_config_value(nullptr, kConfigSection, "EngineLow", s.engine_low, &n);
  }
  reshade::get_config_value(nullptr, kConfigSection, "CropSnap", f.crop_snap);
  reshade::get_config_value(nullptr, kConfigSection, "FreeSizes", f.free_sizes);
  reshade::get_config_value(nullptr, kConfigSection, "EngineHistory", s.engine_history);

  // A hand-edited ini must not push values past what the overlay allows.
  s.transfer = Clamp(s.transfer, 0.0f, 3.0f);
  s.colour_strength = Clamp(s.colour_strength, 0.0f, 1.0f);
  s.max_ratio = Clamp(s.max_ratio, 1.0f, 8.0f);
  s.wp_scale = Clamp(s.wp_scale, 0.1f, 4.0f);
  s.intensity = Clamp(s.intensity, 0.0f, 3.0f);
  s.local_structure = Clamp(s.local_structure, 0.0f, 3.0f);
  s.local_tone = Clamp(s.local_tone, 0.0f, 3.0f);
  if (s.preset < 0 || s.preset > 7) s.preset = 0;
  if (s.style < 0 || s.style > 2) s.style = 0;
  s.split_global_intensity = Clamp(s.split_global_intensity, 0.0f, 3.0f);
  s.split_global_structure = Clamp(s.split_global_structure, 0.0f, 3.0f);
  s.split_global_tone = Clamp(s.split_global_tone, 0.0f, 3.0f);
  s.split_global_skin = Clamp(s.split_global_skin, -1.0f, 2.0f);
  if (s.split_global_style < 0 || s.split_global_style > 2) s.split_global_style = 0;
  if (s.split_global_preset < -1 || s.split_global_preset > 7) s.split_global_preset = 1;
  if (f.fovea_debug < 0 || f.fovea_debug > 3) f.fovea_debug = 0;
  if (f.periphery_debug < 0 || f.periphery_debug > 3) f.periphery_debug = 0;
  f.width = Clamp(f.width, 0.05f, 1.0f);
  f.height = Clamp(f.height, 0.05f, 1.0f);
  f.falloff = Clamp(f.falloff, 0.0f, 0.5f);
  if (f.gaze_source < 0 || f.gaze_source > 1) f.gaze_source = 0;
  if (f.gaze_port < 1024 || f.gaze_port > 65535) f.gaze_port = 47620;
  f.gaze_smooth_ms = Clamp(f.gaze_smooth_ms, 0.0f, 500.0f);
  f.jump_reset = Clamp(f.jump_reset, 0.02f, 2.0f);
  f.global_scale = Clamp(f.global_scale, 0.1f, 0.5f);
  f.grade_blur = Clamp(f.grade_blur, 1.0f, 16.0f);
  if (f.split_calls != 1) f.split_calls = 2;
  f.tone_smoothing = Clamp(f.tone_smoothing, 0.0f, 0.95f);
  f.tone_deadband = Clamp(f.tone_deadband, 0.0f, 0.1f);
  f.crop_tone = Clamp(f.crop_tone, 0.0f, 1.0f);
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
  reshade::set_config_value(nullptr, kConfigSection, "SplitGlobalSkinStructure", s.split_global_skin);
  reshade::set_config_value(nullptr, kConfigSection, "DLAA", s.dlaa);
  reshade::set_config_value(nullptr, kConfigSection, "BeforeUpscale", s.before_upscale);
  reshade::set_config_value(nullptr, kConfigSection, "PassOnGradedFrame", s.graded);

  const auto& f = fov::settings;
  reshade::set_config_value(nullptr, kConfigSection, "Foveation", f.enabled);
  reshade::set_config_value(nullptr, kConfigSection, "FoveaWidth", f.width);
  reshade::set_config_value(nullptr, kConfigSection, "FoveaHeight", f.height);
  reshade::set_config_value(nullptr, kConfigSection, "FoveaFalloff", f.falloff);
  reshade::set_config_value(nullptr, kConfigSection, "GazeSource", f.gaze_source);
  reshade::set_config_value(nullptr, kConfigSection, "GazePort", f.gaze_port);
  reshade::set_config_value(nullptr, kConfigSection, "GazeSmoothMs", f.gaze_smooth_ms);
  reshade::set_config_value(nullptr, kConfigSection, "GazeJumpReset", f.jump_reset);
  reshade::set_config_value(nullptr, kConfigSection, "GlobalScale", f.global_scale);
  reshade::set_config_value(nullptr, kConfigSection, "GradeBlur", f.grade_blur);
  reshade::set_config_value(nullptr, kConfigSection, "SplitCalls", f.split_calls);
  reshade::set_config_value(nullptr, kConfigSection, "ToneSmoothing", f.tone_smoothing);
  reshade::set_config_value(nullptr, kConfigSection, "ToneDeadband", f.tone_deadband);
  reshade::set_config_value(nullptr, kConfigSection, "CropToneFromPeriphery", f.crop_tone);
  reshade::set_config_value(nullptr, kConfigSection, "GlobalEvery", f.global_every);
  reshade::set_config_value(nullptr, kConfigSection, "FoveaDebugView", f.fovea_debug);
  reshade::set_config_value(nullptr, kConfigSection, "PeripheryDebugView", f.periphery_debug);
  reshade::set_config_value(nullptr, kConfigSection, "PeripheryPass", f.periphery_pass);
  reshade::set_config_value(nullptr, kConfigSection, "CropPass", f.crop_pass);
  reshade::set_config_value(nullptr, kConfigSection, "CropOnPeriphery", f.crop_on_periphery);
  reshade::set_config_value(nullptr, kConfigSection, "SecondPass", s.second_pass);
  reshade::set_config_value(nullptr, kConfigSection, "SrPreset", sr_fovea::preset.load());
  reshade::set_config_value(nullptr, kConfigSection, "SrFovea", sr_fovea::mode.load());
  reshade::set_config_value(nullptr, kConfigSection, "SrCropPreset", sr_fovea::crop_preset.load());
  reshade::set_config_value(nullptr, kConfigSection, "SrShift", sr_fovea::shift.load());
  reshade::set_config_value(nullptr, kConfigSection, "SrFadeIn", sr_fovea::fade_in.load());
  reshade::set_config_value(nullptr, kConfigSection, "NREngine", s.engine);
  reshade::set_config_value(nullptr, kConfigSection, "EngineCrop", static_cast<const char*>(s.engine_crop));
  reshade::set_config_value(nullptr, kConfigSection, "EngineLow", static_cast<const char*>(s.engine_low));
  reshade::set_config_value(nullptr, kConfigSection, "CropSnap", f.crop_snap);
  reshade::set_config_value(nullptr, kConfigSection, "FreeSizes", f.free_sizes);
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

// A string key, when present (left as it is otherwise).
void LiveString(const char* path, const char* key, char* value, DWORD size) {
  char buf[64] = {};
  if (GetPrivateProfileStringA("DLSSNR", key, "", buf, sizeof(buf), path) > 0) strncpy_s(value, size, buf, _TRUNCATE);
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
  const float old_periphery[5] = {s.split_global_intensity, s.split_global_structure, s.split_global_tone,
                                  (float)s.split_global_style, s.split_global_skin};

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
  s.split_global_skin = LiveFloat(path, "SplitGlobalSkinStructure", s.split_global_skin);
  s.dlaa = LiveInt(path, "DLAA", s.dlaa ? 1 : 0) != 0;
  s.before_upscale = LiveInt(path, "BeforeUpscale", s.before_upscale ? 1 : 0) != 0;
  s.graded = LiveInt(path, "PassOnGradedFrame", s.graded ? 1 : 0) != 0;
  f.enabled = LiveInt(path, "Foveation", f.enabled ? 1 : 0) != 0;
  f.width = LiveFloat(path, "FoveaWidth", f.width);
  f.height = LiveFloat(path, "FoveaHeight", f.height);
  f.falloff = LiveFloat(path, "FoveaFalloff", f.falloff);
  f.gaze_source = LiveInt(path, "GazeSource", f.gaze_source);
  f.gaze_port = LiveInt(path, "GazePort", f.gaze_port);
  f.gaze_smooth_ms = LiveFloat(path, "GazeSmoothMs", f.gaze_smooth_ms);
  f.global_scale = LiveFloat(path, "GlobalScale", f.global_scale);
  f.grade_blur = LiveFloat(path, "GradeBlur", f.grade_blur);
  f.split_calls = LiveInt(path, "SplitCalls", f.split_calls);
  f.tone_smoothing = LiveFloat(path, "ToneSmoothing", f.tone_smoothing);
  f.tone_deadband = LiveFloat(path, "ToneDeadband", f.tone_deadband);
  f.crop_tone = Clamp(LiveFloat(path, "CropToneFromPeriphery", f.crop_tone), 0.0f, 1.0f);
  f.global_every = LiveInt(path, "GlobalEvery", f.global_every);
  f.fovea_debug = LiveInt(path, "FoveaDebugView", f.fovea_debug);
  f.periphery_debug = LiveInt(path, "PeripheryDebugView", f.periphery_debug);
  f.periphery_pass = LiveInt(path, "PeripheryPass", f.periphery_pass ? 1 : 0) != 0;
  f.crop_pass = LiveInt(path, "CropPass", f.crop_pass ? 1 : 0) != 0;
  f.crop_on_periphery = LiveInt(path, "CropOnPeriphery", f.crop_on_periphery ? 1 : 0) != 0;
  s.second_pass = LiveInt(path, "SecondPass", s.second_pass ? 1 : 0) != 0;
  ffx_probe::enabled = LiveInt(path, "FgNrHudless", ffx_probe::enabled.load());
  s.engine = LiveInt(path, "NREngine", s.engine ? 1 : 0) != 0;
  LiveString(path, "EngineCrop", s.engine_crop, sizeof(s.engine_crop));
  LiveString(path, "EngineLow", s.engine_low, sizeof(s.engine_low));
  f.crop_snap = LiveInt(path, "CropSnap", f.crop_snap);
  f.free_sizes = LiveInt(path, "FreeSizes", f.free_sizes ? 1 : 0) != 0;
  f.jump_reset = Clamp(LiveFloat(path, "GazeJumpReset", f.jump_reset), 0.02f, 2.0f);
  if (f.crop_snap < 0 || f.crop_snap > 256) f.crop_snap = 32;
  s.engine_history = Clamp(LiveFloat(path, "EngineHistory", s.engine_history), 0.0f, 1.0f);
  diag::dump_requested = LiveInt(path, "DumpFrame", diag::dump_requested);
  diag::dump_burst = LiveInt(path, "DumpBurst", diag::dump_burst);
  sr_fovea::preset = LiveInt(path, "SrPreset", sr_fovea::preset.load());
  sr_fovea::mode = LiveInt(path, "SrFovea", sr_fovea::mode.load());
  sr_fovea::crop_preset = LiveInt(path, "SrCropPreset", sr_fovea::crop_preset.load());
  sr_fovea::shift = LiveInt(path, "SrShift", sr_fovea::shift.load());
  sr_fovea::fade_in = LiveInt(path, "SrFadeIn", sr_fovea::fade_in.load());
  sr_fovea::margin = LiveInt(path, "SrMargin", 64);
  sr_fovea::feather = LiveInt(path, "SrFeather", 1);

  // The same bounds a hand-edited ReShade.ini gets.
  s.transfer = Clamp(s.transfer, 0.0f, 3.0f);
  s.colour_strength = Clamp(s.colour_strength, 0.0f, 1.0f);
  s.max_ratio = Clamp(s.max_ratio, 1.0f, 8.0f);
  s.wp_scale = Clamp(s.wp_scale, 0.1f, 4.0f);
  s.intensity = Clamp(s.intensity, 0.0f, 3.0f);
  s.local_structure = Clamp(s.local_structure, 0.0f, 3.0f);
  s.local_tone = Clamp(s.local_tone, 0.0f, 3.0f);
  if (s.preset < 0 || s.preset > 7) s.preset = 0;
  if (s.style < 0 || s.style > 2) s.style = 0;
  s.split_global_intensity = Clamp(s.split_global_intensity, 0.0f, 3.0f);
  s.split_global_structure = Clamp(s.split_global_structure, 0.0f, 3.0f);
  s.split_global_tone = Clamp(s.split_global_tone, 0.0f, 3.0f);
  if (s.split_global_style < 0 || s.split_global_style > 2) s.split_global_style = 0;
  if (s.split_global_preset < -1 || s.split_global_preset > 7) s.split_global_preset = 1;
  if (f.fovea_debug < 0 || f.fovea_debug > 3) f.fovea_debug = 0;
  if (f.periphery_debug < 0 || f.periphery_debug > 3) f.periphery_debug = 0;
  if (s.debug_view < 0 || s.debug_view > 4) s.debug_view = 0;
  f.width = Clamp(f.width, 0.05f, 1.0f);
  f.height = Clamp(f.height, 0.05f, 1.0f);
  f.falloff = Clamp(f.falloff, 0.0f, 0.5f);
  if (f.gaze_source < 0 || f.gaze_source > 1) f.gaze_source = 0;
  if (f.gaze_port < 1024 || f.gaze_port > 65535) f.gaze_port = 47620;
  f.gaze_smooth_ms = Clamp(f.gaze_smooth_ms, 0.0f, 500.0f);
  f.global_scale = Clamp(f.global_scale, 0.1f, 0.5f);
  f.grade_blur = Clamp(f.grade_blur, 1.0f, 16.0f);
  if (f.split_calls != 1) f.split_calls = 2;
  f.tone_smoothing = Clamp(f.tone_smoothing, 0.0f, 0.95f);
  f.tone_deadband = Clamp(f.tone_deadband, 0.0f, 0.1f);
  f.crop_tone = Clamp(f.crop_tone, 0.0f, 1.0f);
  if (f.global_every < 1 || f.global_every > 4) f.global_every = 1;

  // The model reads these only when its feature is built; foveation geometry rebuilds by itself.
  if (s.intensity != old_model[0] || s.local_structure != old_model[1] ||
      s.local_tone != old_model[2] || s.preset != old_preset || s.style != old_style ||
      s.mask_blend_mode != old_ints[0] || s.ui_correction != old_ints[1] || s.auto_mask != old_ints[2] || s.split_global_preset != old_ints[3] ||
      s.skin_structure != old_floats[0] || s.global_tone != old_floats[1] || s.scaling_ratio != old_floats[2] ||
      s.split_global_intensity != old_periphery[0] || s.split_global_structure != old_periphery[1] ||
      s.split_global_tone != old_periphery[2] || (float)s.split_global_style != old_periphery[3] ||
      s.split_global_skin != old_periphery[4])
    nr_runner::ApplyModelSettings();
  SaveSettings();
  ngx_probe::Log("nr-fwd: live settings applied from dlssnr-linux-live.ini");
}

// A size picker over the engine's program folders (nr-engine\<prefix>WxH): each size with its share of
// the frame. The programs fix the region's size for both backends (unless free sizes). True on a change.
bool ProgramSize(const char* label, const char* prefix, char* value, size_t size) {
  static std::vector<std::string> crops, lows;
  std::vector<std::string>& list = prefix[0] == 'c' ? crops : lows;
  if (list.empty()) list = nr_engine::Available(prefix);
  const unsigned fw = nr_runner::s.out_w, fh = nr_runner::s.out_h;
  auto describe = [&](const std::string& name) {
    unsigned w = 0, h = 0;
    char text[160];
    if (std::sscanf(name.c_str() + std::strlen(prefix), "%ux%u", &w, &h) == 2 && fw != 0 && fh != 0)
      std::snprintf(text, sizeof(text), "%ux%u  (%.0f%% x %.0f%% of %ux%u, %.1f%% of the pixels)%s", w, h,
                    100.0 * w / fw, 100.0 * h / fh, fw, fh, 100.0 * w * h / ((double)fw * fh),
                    name.find("-ctx") == std::string::npos ? ""
                    : prefix[0] == 'c' ? " + scene context (needs its periphery)" : " + exports scene context");
    else
      std::snprintf(text, sizeof(text), "%s", name.c_str());
    return std::string(text);
  };
  bool changed = false;
  if (ImGui::BeginCombo(label, describe(value).c_str())) {
    for (const std::string& n : list)
      if (ImGui::Selectable(describe(n).c_str(), n == value)) {
        strncpy_s(value, size, n.c_str(), _TRUNCATE);
        changed = true;
      }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  ImGui::PushID(label);
  if (ImGui::SmallButton("rescan")) list = nr_engine::Available(prefix);
  ImGui::PopID();
  return changed;
}

// Keeps the crop and periphery programs a working pair: a crop that reads scene context (crop-WxH-ctxAxB) needs
// the periphery that exports it (low-AxB-ctx). After the crop changed (`crop_changed`) the periphery follows it;
// after the periphery changed, the crop takes the matching variant of its size (with or without context).
void PairPrograms(bool crop_changed) {
  auto& s = nr_runner::s;
  auto exists = [](const std::string& name) {
    for (const std::string& n : nr_engine::Available(name[0] == 'c' ? "crop-" : "low-"))
      if (n == name) return true;
    return false;
  };
  const std::string crop = s.engine_crop, low = s.engine_low;
  const size_t ctx = crop.find("-ctx");
  const std::string crop_size = crop.substr(5, ctx == std::string::npos ? std::string::npos : ctx - 5);
  if (crop_changed) {
    if (ctx != std::string::npos) {
      const std::string want = "low-" + crop.substr(ctx + 4) + "-ctx";
      if (exists(want)) strncpy_s(s.engine_low, sizeof(s.engine_low), want.c_str(), _TRUNCATE);
    }
    return;
  }
  const size_t lctx = low.find("-ctx");
  const std::string want = lctx != std::string::npos ? "crop-" + crop_size + "-ctx" + low.substr(4, lctx - 4)
                                                     : "crop-" + crop_size;
  if (want != crop && exists(want)) strncpy_s(s.engine_crop, sizeof(s.engine_crop), want.c_str(), _TRUNCATE);
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
// The settings page, in this add-on's entry on ReShade's Add-ons tab. Sections, top to bottom: what the
// pass is doing now, the final picture, NVIDIA's model controls, the pipeline around the model, foveation
// (the crop, the periphery, how they are blended, the gaze) and every debug view in one place. Controls a
// mode does not use are not shown.
void OnDrawOverlay(reshade::api::effect_runtime*) {
  auto& s = nr_runner::s;
  auto& f = fov::settings;
  bool changed = false;
  const bool split = f.enabled;
  const bool two_calls = split && f.split_calls == 2;
  const bool engine_sizes = two_calls && f.engine_crop_w != 0;   // the engine's programs fix both regions' sizes

  bool enabled = nr_runner::IsEnabled();
  if (ImGui::Checkbox("Enable Neural Rendering (F10)", &enabled)) nr_runner::SetEnabled(enabled);
  if (s.frame_w != 0 && s.frame_h != 0) {
    uint32_t fw = 0, fh = 0;
    nr_runner::FeatureSize(&fw, &fh);
    ImGui::TextDisabled("The model works on %ux%u of a %ux%u frame (%.0f%% of the pixels)%s", fw, fh, s.frame_w,
                        s.frame_h, 100.0 * fw * fh / ((double)s.frame_w * s.frame_h),
                        nr_runner::EngineCovers() ? "; our engine" : "; NVIDIA's runtime");
  }

  // The three ways to run the model. They set the engine and foveation switches below (Pipeline, Foveation),
  // which stay independent for benchmarks: NVIDIA's runtime under foveation is the fourth combination.
  {
    int mode = !s.engine ? (f.enabled ? 3 : 0) : (f.enabled ? 2 : 1);
    bool pick = false;
    pick |= ImGui::RadioButton("NVIDIA's runtime (openNR FP16)", &mode, 0);
    ImGui::SameLine();
    pick |= ImGui::RadioButton("Our engine, whole frame", &mode, 1);
    ImGui::SameLine();
    pick |= ImGui::RadioButton("Our engine, foveated", &mode, 2);
    if (mode == 3) ImGui::TextDisabled("Now: NVIDIA's runtime, foveated (set under Pipeline and Foveation)");
    if (pick && mode != 3) {
      s.engine = mode != 0;
      f.enabled = mode == 2;
      changed = true;
    }
    if (mode == 1 && s.frame_w != 0 && nr_engine::FullFor(s.frame_w, s.frame_h).empty())
      ImGui::TextDisabled("No whole-frame program for %ux%u (nr-engine\\full-WxH): NVIDIA's runtime runs it.",
                          s.frame_w, s.frame_h);
  }

  if (ImGui::CollapsingHeader("Picture", ImGuiTreeNodeFlags_DefaultOpen)) {
    // Where the pass runs decides what the frame is: the graded picture needs no colour bridge.
    int where = s.graded ? 0 : (s.before_upscale ? 2 : 1);
    ImGui::TextUnformatted("The model works on");
    if (ImGui::RadioButton("The game's graded frame, before its UI (where NVIDIA runs it)", &where, 0)) changed = true;
    if (ImGui::RadioButton("DLSS's output (HDR, before the game's grade; colour bridge)", &where, 1)) changed = true;
    if (ImGui::RadioButton("DLSS's input (HDR, render resolution; colour bridge)", &where, 2)) changed = true;
    s.graded = where == 0;
    s.before_upscale = where == 2;
    changed |= SliderWithReset("Model's share of the picture", &s.transfer, 0.0f, 3.0f, "%.2f", 1.0f);
    if (s.graded)
      changed |= ImGui::Checkbox("Deep fried: a second pass on the first's answer (the model's cost twice)",
                                 &s.second_pass);
    if (!s.graded) {
      ImGui::SeparatorText("Colour bridge (HDR frames only)");
      changed |= SliderWithReset("Colour strength", &s.colour_strength, 0.0f, 1.0f, "%.2f", 0.0f);
      changed |= SliderWithReset("Highlight guard", &s.max_ratio, 1.0f, 8.0f, "%.1fx", 2.0f);
      changed |= SliderWithReset("White point scale", &s.wp_scale, 0.1f, 4.0f, "%.2f", 1.0f);
      if (s.wp_ema > 0.0f) ImGui::TextDisabled("Measured white point: %.4f", s.wp_ema);
      else ImGui::TextDisabled("Measured white point: (measuring...)");
    } else {
      ImGui::TextDisabled("Graded frames: %u this period, not found %u", graded_hook::ran, graded_hook::missed);
    }
  }

  if (ImGui::CollapsingHeader("Model (NVIDIA's controls)", ImGuiTreeNodeFlags_DefaultOpen)) {
    // NR intensity is the model's overall hand; Structure intensity the fine detail it synthesises; Global
    // intensity the broad tone (its relighting). Style: NVIDIA's processing profiles (names from community
    // testing via OptiScaler_DLSSNR). With two calls, each region has its own set, side by side. Our engine
    // reads these live; NGX only at a rebuild (Apply).
    const bool periphery = two_calls && f.periphery_pass;
    if (periphery) ImGui::SeparatorText("The crop");
    changed |= SliderWithReset("NR intensity", &s.intensity, 0.0f, 3.0f, "%.2f", 1.0f);
    changed |= SliderWithReset("Structure intensity", &s.local_structure, 0.0f, 3.0f, "%.2f", 1.0f);
    changed |= SliderWithReset("Global intensity", &s.local_tone, 0.0f, 3.0f, "%.2f", 1.0f);
    changed |= SliderWithReset("Skin detail (-1 = structure)", &s.skin_structure, -1.0f, 2.0f, "%.2f", -1.0f);
    if (s.style > 2) s.style = 2;
    changed |= ImGui::Combo("Style", &s.style, "Default (standard)\0Natural\0Cinematic\0");
    changed |= ImGui::SliderInt("Preset (NGX only)", &s.preset, 0, 7);
    if (periphery) {
      ImGui::SeparatorText("The periphery");
      bool model = false;
      model |= SliderWithReset("NR intensity##periphery", &s.split_global_intensity, 0.0f, 3.0f, "%.2f", 1.0f);
      model |= SliderWithReset("Structure intensity##periphery", &s.split_global_structure, 0.0f, 3.0f, "%.2f", 1.0f);
      model |= SliderWithReset("Global intensity##periphery", &s.split_global_tone, 0.0f, 3.0f, "%.2f", 1.0f);
      model |= SliderWithReset("Skin detail (-1 = structure)##periphery", &s.split_global_skin, -1.0f, 2.0f, "%.2f", -1.0f);
      model |= ImGui::Combo("Style##periphery", &s.split_global_style, "Default (standard)\0Natural\0Cinematic\0");
      model |= ImGui::SliderInt("Preset (NGX only)##periphery", &s.split_global_preset, -1, 7,
                                s.split_global_preset < 0 ? "same as the crop" : "%d");
      if (model) nr_runner::ApplyModelSettings();
      changed |= model;
      ImGui::Spacing();
    }
    if (ImGui::TreeNode("NVIDIA's advanced options (both regions)")) {
      // -1 (or 0 for the ratio) leaves the library's default. Our engine reads the skin mask; the rest is NGX's.
      changed |= ImGui::SliderInt("Skin mask (-1 default, 0 off, 1 on)", &s.auto_mask, -1, 1);
      changed |= ImGui::SliderInt("UI correction (-1 default, NGX only)", &s.ui_correction, -1, 1);
      changed |= ImGui::SliderInt("Mask blend mode (-1 default, NGX only)", &s.mask_blend_mode, -1, 3);
      changed |= SliderWithReset("Global tone strength (-1 unset, NGX only)", &s.global_tone, -1.0f, 2.0f, "%.2f", -1.0f);
      changed |= SliderWithReset("Scaling ratio (0 unset, NGX only)", &s.scaling_ratio, 0.0f, 3.0f, "%.2f", 0.0f);
      ImGui::TreePop();
    }
    if (ImGui::Button("Apply to NGX (rebuild)")) nr_runner::ApplyModelSettings();
    ImGui::SameLine();
    ImGui::TextDisabled("our engine applies them live");
  }

  if (ImGui::CollapsingHeader("Pipeline")) {
    changed |= ImGui::Checkbox("Anti-aliasing (DLAA) on the frame before the pass", &s.dlaa);
    // What runs the network: independent of foveation. The engine replays NVIDIA's Small network from
    // programs traced at fixed sizes; where it has no program for a call, NGX runs that call.
    changed |= ImGui::Checkbox("Our engine (precompiled Ampere cubins) instead of NGX", &s.engine);
    if (s.engine) {
      if (nr_engine::g.count > 0)
        ImGui::TextDisabled(nr_engine::g.count > 1 ? "Engine: %s -- programs %s and %s (the Small network)"
                                                   : "Engine: %s -- program %s (the Small network)",
                            nr_engine::StatusText(), nr_engine::g.names[0].c_str(), nr_engine::g.names[1].c_str());
      else
        ImGui::TextDisabled("Engine: %s", nr_engine::StatusText());
      if (!split && !nr_engine::CoversFrame(s.frame_w, s.frame_h))
        ImGui::TextDisabled("No whole-frame program for this resolution: NVIDIA's runtime runs it.");
      else if (split && f.split_calls != 2)
        ImGui::TextDisabled("Foveation in one call: NVIDIA's runtime runs it.");
      else if (f.free_sizes)
        ImGui::TextDisabled("Free sizes are on (Foveation): NGX runs the regions.");
      changed |= SliderWithReset("Engine history (1 = NVIDIA's)", &s.engine_history, 0.0f, 1.0f, "%.2f", 1.0f);
    }
    if (s.preset != 0 || (two_calls && f.periphery_pass && s.split_global_preset > 0))
      ImGui::TextDisabled("NGX runs preset %d / %d here; the engine always runs Small (preset 0).", s.preset,
                          s.split_global_preset < 0 ? s.preset : s.split_global_preset);
  }

  if (ImGui::CollapsingHeader("Foveation", ImGuiTreeNodeFlags_DefaultOpen)) {
    changed |= ImGui::Checkbox("Foveated NR", &f.enabled);
    if (f.enabled) {
      ImGui::SeparatorText("The crop (full resolution, around the gaze)");
      if (two_calls && nr_engine::g.count > 0)
        changed |= ImGui::Checkbox("Free sizes (NGX only: any size, not just the engine programs')", &f.free_sizes);
      if (engine_sizes) {
        if (ProgramSize("Crop size", "crop-", s.engine_crop, sizeof(s.engine_crop))) {
          PairPrograms(true);
          changed = true;
        }
      } else {
        changed |= SliderWithReset("Crop width", &f.width, 0.05f, 1.0f, "%.2f of the frame", 0.40f);
        changed |= SliderWithReset("Crop height", &f.height, 0.05f, 1.0f, "%.2f of the frame", 0.50f);
      }
      changed |= ImGui::SliderInt("Crop snap", &f.crop_snap, 0, 256, f.crop_snap <= 1 ? "follows the gaze exactly" : "%d px");
      changed |= SliderWithReset("Edge blend", &f.falloff, 0.0f, 0.5f, "%.2f of the height", 0.10f);

      if (two_calls)
        changed |= ImGui::Checkbox("Run the crop pass (F8; off: its answer is its input, the periphery's tone kept)",
                                   &f.crop_pass);
      if (two_calls && f.periphery_pass)
        changed |= ImGui::Checkbox("The crop works on the periphery's result (F11; off: on the raw frame, in parallel)",
                                   &f.crop_on_periphery);
      ImGui::SeparatorText("The periphery (the whole frame, low resolution)");
      if (two_calls) {
        changed |= ImGui::Checkbox("Run the periphery pass (off: the game's own frame outside the crop)",
                                   &f.periphery_pass);
      }
      if (engine_sizes && ProgramSize("Periphery size", "low-", s.engine_low, sizeof(s.engine_low))) {
        PairPrograms(false);
        changed = true;
      }
      else changed |= SliderWithReset("Low-res frame", &f.global_scale, 0.1f, 0.5f, "%.2f of the frame", 0.25f);
      if (two_calls && f.periphery_pass) {
        changed |= ImGui::SliderInt("Run every", &f.global_every, 1, 4, f.global_every == 1 ? "frame" : "%d frames");
        ImGui::TextDisabled("Its model controls are under Model, beside the crop's.");
      }

      ImGui::SeparatorText("How the two are combined");
      bool two = f.split_calls == 2;
      if (ImGui::Checkbox("A model call per region (independent grading)", &two)) {
        f.split_calls = two ? 2 : 1;
        changed = true;
      }
      changed |= SliderWithReset("Tone blur", &f.grade_blur, 1.0f, 16.0f, "%.1f px", 1.0f);
      changed |= SliderWithReset("Tone smoothing", &f.tone_smoothing, 0.0f, 0.95f, "%.2f", 0.95f);
      changed |= SliderWithReset("Tone dead band", &f.tone_deadband, 0.0f, 0.1f, "%.3f", 0.01f);
      changed |= SliderWithReset("Crop's broad tone from the periphery", &f.crop_tone, 0.0f, 1.0f,
                                 f.crop_tone >= 1.0f ? "%.2f (one grading)" : f.crop_tone <= 0.0f ? "%.2f (its own)" : "%.2f",
                                 1.0f);

      ImGui::SeparatorText("Gaze");
      changed |= ImGui::Combo("Fovea centre", &f.gaze_source, "Fixed (frame centre)\0Gaze over UDP\0");
      if (f.gaze_source == 1) {
        changed |= ImGui::InputInt("Gaze UDP port", &f.gaze_port);
        changed |= SliderWithReset("Gaze smoothing", &f.gaze_smooth_ms, 0.0f, 200.0f, "%.0f ms", 30.0f);
        changed |= SliderWithReset("Jump that resets the history", &f.jump_reset, 0.02f, 2.0f, "%.2f of the frame", 0.20f);
        float gx = 0.0f, gy = 0.0f;
        if (fov::GazeLatest(&gx, &gy)) ImGui::TextDisabled("Gaze: %.3f, %.3f", gx, gy);
        else ImGui::TextDisabled("Gaze: no packets in the last 0.5 s (centre used)");
      }
    }
  }

  if (ImGui::CollapsingHeader("Super resolution (DLSS-SR)", ImGuiTreeNodeFlags_DefaultOpen)) {
    // The game's preset is read when it creates DLSS-SR; the crop's own is rebuilt here on a change.
    static const int kPresets[] = {0, 5, 10, 11, 12, 13};
    static const char kPresetNames[] = "Default\0E (CNN)\0J (transformer)\0K (transformer)\0L (transformer 2)\0M (transformer 2)\0";
    auto pick = [&](const char* label, std::atomic<int>& v) {
      int i = 0;
      for (int k = 0; k < 6; ++k)
        if (kPresets[k] == v.load()) i = k;
      if (ImGui::Combo(label, &i, kPresetNames)) {
        v = kPresets[i];
        changed = true;
      }
    };
    pick("The game's preset (restart the game to apply)", sr_fovea::preset);
    int mode = sr_fovea::mode.load();
    bool crop = mode != 0;
    if (ImGui::Checkbox("DLSS-SR of its own on the gaze crop (F6)", &crop)) {
      sr_fovea::mode = crop ? 1 : 0;
      changed = true;
    }
    if (crop) {
      pick("The crop's preset (F7 cycles default, E, K, M)", sr_fovea::crop_preset);
      bool shift = sr_fovea::shift.load() != 0;
      if (ImGui::Checkbox("Carry its history over when it moves", &shift)) {
        sr_fovea::shift = shift ? 1 : 0;
        changed = true;
      }
      int fade = sr_fovea::fade_in.load();
      if (ImGui::SliderInt("Fade-in of pixels new to it", &fade, 0, 16, fade == 0 ? "off" : "%d frames")) {
        sr_fovea::fade_in = fade;
        changed = true;
      }
    }
  }

  if (ImGui::CollapsingHeader("Debug views")) {
    ImGui::Combo("Final picture", &s.debug_view,
                 "Off\0What the model sees\0Model answer\0Difference x20\0Fovea map (where the crop is)\0");
    if (two_calls) {
      // One pass alone, as the model saw and answered it; the crop's wins when both are set.
      static const char kPassViews[] = "Off\0What the model sees\0Model answer\0Difference x20\0";
      changed |= ImGui::Combo("The crop's pass", &f.fovea_debug, kPassViews);
      changed |= ImGui::Combo("The periphery's pass", &f.periphery_debug, kPassViews);
    } else {
      ImGui::TextDisabled("Per-pass views need the split foveation in two calls.");
    }
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
    reshade::register_event<reshade::addon_event::reshade_overlay>(OnReshadeOverlay);
      ui_probe::Register();
      graded_hook::Register();
      ngx_probe::mark = ui_probe::Mark;
      ngx_probe::after_dlss = graded_hook::OnDlssEvaluated;
      nr_runner::after_graded = ffx_probe::Capture;
      reshade::register_overlay(nullptr, OnDrawOverlay);   // the settings, in this add-on's entry on the Add-ons tab
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
      ngx_probe::mark = nullptr;
      ngx_probe::after_dlss = nullptr;
      graded_hook::Unregister();
      ui_probe::Unregister();
      reshade::unregister_overlay(nullptr, OnDrawOverlay);
      reshade::unregister_event<reshade::addon_event::present>(OnPresent);
      reshade::unregister_event<reshade::addon_event::reshade_overlay>(OnReshadeOverlay);
      reshade::unregister_event<reshade::addon_event::init_swapchain>(OnInitSwapchain);
      reshade::unregister_addon(h_module);
      break;
  }
  return TRUE;
}
