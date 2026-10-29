// ui-probe.request beside the exe: records the command stream of the next two frames into ui-probe.txt
// -- per command list, each run of draws into one render target (its size, format and whether it is a
// swapchain backbuffer, how many draws were full-screen and how many partial, the first few vertex counts),
// compute dispatches, copies, the DLSS evaluate, and the order the lists were submitted in. It is how
// the point where a game starts drawing its UI is found (the pass runs just before it).
#pragma once

#include <windows.h>

#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>

#include <d3d12.h>
#include <reshade.hpp>

namespace ui_probe {

struct Run {
  uint64_t rt = 0;
  uint32_t full = 0, partial = 0, dispatches = 0;
  std::string first;   // the first few draws' vertex/index counts
};

inline std::mutex m;
inline bool on = false;
inline int presents_left = 0;
inline uint64_t seq = 0;
inline std::string out;
inline std::unordered_map<uint64_t, Run> runs;   // per command list
inline uint64_t backbuffers[4] = {};
inline wchar_t path[MAX_PATH] = {};

inline const char* FormatName(reshade::api::format f) {
  switch (f) {
    case reshade::api::format::r8g8b8a8_unorm: return "rgba8";
    case reshade::api::format::r8g8b8a8_unorm_srgb: return "rgba8_srgb";
    case reshade::api::format::r8g8b8a8_typeless: return "rgba8_tl";
    case reshade::api::format::b8g8r8a8_unorm: return "bgra8";
    case reshade::api::format::b8g8r8a8_typeless: return "bgra8_tl";
    case reshade::api::format::r10g10b10a2_unorm: return "rgb10a2";
    case reshade::api::format::r10g10b10a2_typeless: return "rgb10a2_tl";
    case reshade::api::format::r11g11b10_float: return "r11g11b10f";
    case reshade::api::format::r16g16b16a16_float: return "rgba16f";
    case reshade::api::format::r16g16b16a16_typeless: return "rgba16_tl";
    default: return nullptr;
  }
}

inline std::string Describe(reshade::api::device* dev, uint64_t res) {
  if (res == 0) return "none";
  char buf[160];
  const auto d = dev->get_resource_desc(reshade::api::resource{res});
  const char* f = FormatName(d.texture.format);
  bool bb = false;
  for (uint64_t b : backbuffers) bb |= b != 0 && b == res;
  if (f != nullptr)
    std::snprintf(buf, sizeof(buf), "%llx %ux%u %s%s", (unsigned long long)res, d.texture.width, d.texture.height, f,
                  bb ? " BACKBUFFER" : "");
  else
    std::snprintf(buf, sizeof(buf), "%llx %ux%u fmt%u%s", (unsigned long long)res, d.texture.width, d.texture.height,
                  (unsigned)d.texture.format, bb ? " BACKBUFFER" : "");
  return buf;
}

inline void Line(const char* fmt, ...) {
  char buf[512];
  va_list a;
  va_start(a, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, a);
  va_end(a);
  char head[48];
  std::snprintf(head, sizeof(head), "%6llu t%05lu ", (unsigned long long)++seq, GetCurrentThreadId());
  out += head;
  out += buf;
  out += '\n';
}

// Closes the command list's current run of draws (caller holds m).
inline void Flush(reshade::api::command_list* cmd) {
  auto it = runs.find((uint64_t)cmd->get_native());
  if (it == runs.end()) return;
  Run& r = it->second;
  if (r.full + r.partial + r.dispatches > 0)
    Line("cl %llx  rt %s: %u full-screen, %u small, %u dispatches  first: %s", (unsigned long long)cmd->get_native(),
         Describe(cmd->get_device(), r.rt).c_str(), r.full, r.partial, r.dispatches, r.first.c_str());
  r = Run{r.rt};
}

inline void OnBind(reshade::api::command_list* cmd, uint32_t count, const reshade::api::resource_view* rtvs,
                   reshade::api::resource_view) {
  if (!on) return;
  std::lock_guard<std::mutex> g(m);
  Flush(cmd);
  runs[(uint64_t)cmd->get_native()].rt =
      count > 0 && rtvs[0].handle != 0 ? cmd->get_device()->get_resource_from_view(rtvs[0]).handle : 0;
}

inline void Draw(reshade::api::command_list* cmd, uint32_t n) {
  if (!on) return;
  std::lock_guard<std::mutex> g(m);
  Run& r = runs[(uint64_t)cmd->get_native()];
  (n <= 6 ? r.full : r.partial)++;
  if (r.full + r.partial <= 6) r.first += std::to_string(n) + " ";
}

inline bool OnDraw(reshade::api::command_list* cmd, uint32_t v, uint32_t, uint32_t, uint32_t) {
  Draw(cmd, v);
  return false;
}
inline bool OnDrawIndexed(reshade::api::command_list* cmd, uint32_t i, uint32_t, uint32_t, int32_t, uint32_t) {
  Draw(cmd, i);
  return false;
}
inline bool OnDispatch(reshade::api::command_list* cmd, uint32_t, uint32_t, uint32_t) {
  if (!on) return false;
  std::lock_guard<std::mutex> g(m);
  runs[(uint64_t)cmd->get_native()].dispatches++;
  return false;
}
inline bool OnCopyResource(reshade::api::command_list* cmd, reshade::api::resource src, reshade::api::resource dst) {
  if (!on) return false;
  std::lock_guard<std::mutex> g(m);
  Flush(cmd);
  Line("cl %llx  copy %s -> %s", (unsigned long long)cmd->get_native(), Describe(cmd->get_device(), src.handle).c_str(),
       Describe(cmd->get_device(), dst.handle).c_str());
  return false;
}
inline bool OnCopyRegion(reshade::api::command_list* cmd, reshade::api::resource src, uint32_t,
                         const reshade::api::subresource_box*, reshade::api::resource dst, uint32_t,
                         const reshade::api::subresource_box*, reshade::api::filter_mode) {
  return OnCopyResource(cmd, src, dst);
}
inline void OnExecute(reshade::api::command_queue*, reshade::api::command_list* cmd) {
  if (!on) return;
  std::lock_guard<std::mutex> g(m);
  Flush(cmd);
  Line("EXECUTE cl %llx", (unsigned long long)cmd->get_native());
}

// From the DLSS evaluate hook, on the game's native list.
inline void Mark(void* native_cmd, const char* what) {
  if (!on) return;
  std::lock_guard<std::mutex> g(m);
  Line("cl %llx  ==== %s ====", (unsigned long long)native_cmd, what);
}

inline void OnPresent(reshade::api::swapchain* sc, const wchar_t* dir) {
  for (uint32_t i = 0; i < 4; ++i)
    backbuffers[i] = i < sc->get_back_buffer_count() ? sc->get_back_buffer(i).handle : 0;
  static uint32_t tick = 0;
  if (!on) {
    if (++tick % 30 != 0 || dir == nullptr || dir[0] == 0) return;
    wchar_t req[MAX_PATH];
    swprintf_s(req, L"%s\\ui-probe.request", dir);
    if (GetFileAttributesW(req) == INVALID_FILE_ATTRIBUTES) return;
    DeleteFileW(req);
    swprintf_s(path, L"%s\\ui-probe.txt", dir);
    std::lock_guard<std::mutex> g(m);
    out.clear();
    runs.clear();
    seq = 0;
    presents_left = 3;
    on = true;
    return;
  }
  std::lock_guard<std::mutex> g(m);
  Line("PRESENT (backbuffer now %s)", Describe(sc->get_device(), sc->get_current_back_buffer().handle).c_str());
  if (--presents_left > 0) return;
  on = false;
  if (FILE* f = _wfopen(path, L"wb")) {
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
  }
  out.clear();
}

inline void Register() {
  reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(OnBind);
  reshade::register_event<reshade::addon_event::draw>(OnDraw);
  reshade::register_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
  reshade::register_event<reshade::addon_event::dispatch>(OnDispatch);
  reshade::register_event<reshade::addon_event::copy_resource>(OnCopyResource);
  reshade::register_event<reshade::addon_event::copy_texture_region>(OnCopyRegion);
  reshade::register_event<reshade::addon_event::execute_command_list>(OnExecute);
}

inline void Unregister() {
  reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(OnBind);
  reshade::unregister_event<reshade::addon_event::draw>(OnDraw);
  reshade::unregister_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
  reshade::unregister_event<reshade::addon_event::dispatch>(OnDispatch);
  reshade::unregister_event<reshade::addon_event::copy_resource>(OnCopyResource);
  reshade::unregister_event<reshade::addon_event::copy_texture_region>(OnCopyRegion);
  reshade::unregister_event<reshade::addon_event::execute_command_list>(OnExecute);
}

}  // namespace ui_probe
