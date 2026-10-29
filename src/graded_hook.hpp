// Graded mode (nr_runner::s.graded): finds the game's graded frame and runs the pass on it just before the
// game draws its UI -- where NVIDIA's integration places the model: on the display-referred picture, after
// the game's tonemapping and grading, with the UI never processed.
//
// How the frame is found. After the DLSS evaluate the game finishes the picture on the same command list:
// HDR post-processing, then one full-screen pass (the tonemap/grade) into an output-size 8- or 10-bit
// surface. The first such surface drawn full-screen after the evaluate is the graded frame. The UI comes
// next, either into that surface (small draws) or into a layer of its own that a last pass composites
// over it (The Witcher 3); the pass runs at whichever comes first -- a small draw into the frame, or the
// frame's render target being switched away. Once per frame.
//
// The pass records straight on the native command list, under ReShade. Afterwards the game's state that
// the pass changes is put back through ReShade's own API, which knows the game's descriptor heaps: the
// heaps (a zero-table bind re-issues them), the PSO, and the descriptor tables bound since the evaluate
// (the game must re-bind after an evaluate anyway, so nothing bound before it can still be in use). The
// pass touches no graphics root signature, render target, viewport or vertex input.
#pragma once

#include <windows.h>

#include <atomic>
#include <mutex>
#include <vector>

#include <d3d12.h>
#include <reshade.hpp>

namespace graded_hook {

struct Tables {
  uint64_t layout = 0;
  std::vector<uint64_t> tables;
};

inline std::mutex m;
inline std::atomic<bool> armed{false};   // between this frame's DLSS evaluate and the pass
inline DWORD arm_thread = 0;
inline uint64_t cl = 0;                  // the native command list the evaluate was recorded on
inline uint64_t rt = 0;                  // its render target 0 now
inline uint64_t scene = 0;               // the graded frame, once found
inline uint32_t scene_state = 0;         // its D3D12 state
inline uint32_t depth_state = 0, motion_state = 0;   // the guides' states, 0 = as the evaluate had them
inline uint64_t pso = 0;
inline Tables graphics, compute;
inline uint32_t ran = 0, failed = 0, missed = 0, presents = 0;
inline bool logged_first = false;

inline uint32_t StateOf(reshade::api::resource_usage u) {
  if (u == reshade::api::resource_usage::general || u == reshade::api::resource_usage::undefined) return 0;
  if (u == reshade::api::resource_usage::present) return D3D12_RESOURCE_STATE_PRESENT;
  return static_cast<uint32_t>(u) & 0xFFFFFFF;
}

// An output-size, display-referred colour surface: what the grade writes.
inline bool IsGradedTarget(reshade::api::device* dev, uint64_t res) {
  if (res == 0) return false;
  const auto d = dev->get_resource_desc(reshade::api::resource{res});
  if (d.texture.width != nr_runner::s.out_w || d.texture.height != nr_runner::s.out_h) return false;
  switch (d.texture.format) {
    case reshade::api::format::r8g8b8a8_unorm:
    case reshade::api::format::r8g8b8a8_unorm_srgb:
    case reshade::api::format::r8g8b8a8_typeless:
    case reshade::api::format::b8g8r8a8_unorm:
    case reshade::api::format::b8g8r8a8_unorm_srgb:
    case reshade::api::format::b8g8r8a8_typeless:
    case reshade::api::format::r10g10b10a2_unorm:
    case reshade::api::format::r10g10b10a2_typeless:
      return true;
    default:
      return false;
  }
}

// Whether this event belongs to the command list being followed (caller holds m). The list is the one the
// evaluate's thread records on next.
inline bool Mine(reshade::api::command_list* cmd) {
  if (!armed.load(std::memory_order_relaxed)) return false;
  const uint64_t native = (uint64_t)cmd->get_native();
  if (cl == 0 && GetCurrentThreadId() == arm_thread) cl = native;
  return cl == native;
}

// From the DLSS evaluate hook, after the evaluate (the runner has taken its guides).
inline void OnDlssEvaluated() {
  if (!nr_runner::s.graded || !nr_runner::graded_guides_valid) return;
  std::lock_guard<std::mutex> g(m);
  arm_thread = GetCurrentThreadId();
  cl = rt = scene = 0;
  scene_state = depth_state = motion_state = 0;
  pso = 0;
  graphics = {};
  compute = {};
  armed = true;
}

inline void Restore(reshade::api::command_list* cmd, const Tables& graphics_, const Tables& compute_, uint64_t pso_) {
  using reshade::api::shader_stage;
  // Zero tables: ReShade re-issues the game's descriptor heaps (and sets this compute root signature, which
  // the pass replaced anyway); the compute tables then go back on the game's compute root signature.
  const uint64_t compute_layout = compute_.layout;
  if (compute_layout != 0) {
    cmd->bind_descriptor_tables(shader_stage::all_compute, reshade::api::pipeline_layout{compute_layout}, 0, 0, nullptr);
    for (uint32_t i = 0; i < compute_.tables.size(); ++i)
      if (compute_.tables[i] != 0)
        cmd->bind_descriptor_tables(shader_stage::all_compute, reshade::api::pipeline_layout{compute_layout}, i, 1,
                                    reinterpret_cast<const reshade::api::descriptor_table*>(&compute_.tables[i]));
  } else {
    cmd->bind_descriptor_tables(shader_stage::all_compute, reshade::api::pipeline_layout{(uint64_t)nr_compose::c.root_sig},
                                0, 0, nullptr);
  }
  // The graphics root signature is untouched, so its tables go back without resetting its root arguments.
  if (graphics_.layout != 0)
    for (uint32_t i = 0; i < graphics_.tables.size(); ++i)
      if (graphics_.tables[i] != 0)
        cmd->bind_descriptor_tables(shader_stage::all_graphics, reshade::api::pipeline_layout{graphics_.layout}, i, 1,
                                    reinterpret_cast<const reshade::api::descriptor_table*>(&graphics_.tables[i]));
  if (pso_ != 0) cmd->bind_pipeline(reshade::api::pipeline_stage::all, reshade::api::pipeline{pso_});
}

// Runs the pass on the graded frame (called without m held: the restore re-enters the bind events).
inline void Trigger(reshade::api::command_list* cmd, const char* why) {
  uint64_t scene_ = 0;
  uint32_t state = 0, ds = 0, ms = 0;
  Tables graphics_, compute_;
  uint64_t pso_ = 0;
  {
    std::lock_guard<std::mutex> g(m);
    if (!armed) return;
    armed = false;
    scene_ = scene;
    state = scene_state;
    ds = depth_state;
    ms = motion_state;
    graphics_ = graphics;
    compute_ = compute;
    pso_ = pso;
  }
  auto* native = reinterpret_cast<ID3D12GraphicsCommandList*>(cmd->get_native());
  auto* scene_res = reinterpret_cast<ID3D12Resource*>(scene_);
  // The guides must be readable by compute; a state the game moved them to since the evaluate goes back.
  const nr_runner::Guides gd = nr_runner::graded_guides;
  const uint32_t readable = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  const bool move_depth = ds != 0 && (ds & readable) == 0;
  const bool move_motion = ms != 0 && (ms & readable) == 0;
  if (move_depth) nr_compose::Barrier(native, gd.depth, (D3D12_RESOURCE_STATES)ds, (D3D12_RESOURCE_STATES)readable);
  if (move_motion) nr_compose::Barrier(native, gd.motion, (D3D12_RESOURCE_STATES)ms, (D3D12_RESOURCE_STATES)readable);
  const bool ok = nr_runner::RunOnGraded(native, scene_res, (D3D12_RESOURCE_STATES)state);
  if (move_depth) nr_compose::Barrier(native, gd.depth, (D3D12_RESOURCE_STATES)readable, (D3D12_RESOURCE_STATES)ds);
  if (move_motion) nr_compose::Barrier(native, gd.motion, (D3D12_RESOURCE_STATES)readable, (D3D12_RESOURCE_STATES)ms);
  Restore(cmd, graphics_, compute_, pso_);
  ++(ok ? ran : failed);
  if (!logged_first) {
    logged_first = true;
    const auto d = cmd->get_device()->get_resource_desc(reshade::api::resource{scene_});
    ngx_probe::Logf("graded: the graded frame is %llx (%ux%u, format %u, state 0x%x); the pass runs at %s (%s)",
                    (unsigned long long)scene_, d.texture.width, d.texture.height, (unsigned)d.texture.format, state,
                    why, ok ? "landed" : "did not run");
  }
}

inline void OnBind(reshade::api::command_list* cmd, uint32_t count, const reshade::api::resource_view* rtvs,
                   reshade::api::resource_view) {
  if (!armed.load(std::memory_order_relaxed)) return;
  bool fire = false;
  {
    std::lock_guard<std::mutex> g(m);
    if (!Mine(cmd)) return;
    const uint64_t next =
        count > 0 && rtvs[0].handle != 0 ? cmd->get_device()->get_resource_from_view(rtvs[0]).handle : 0;
    fire = scene != 0 && rt == scene && next != scene;
    rt = next;
  }
  if (fire) Trigger(cmd, "the frame's target being switched away");
}

inline void Draw(reshade::api::command_list* cmd, uint32_t n) {
  if (!armed.load(std::memory_order_relaxed)) return;
  bool fire = false;
  {
    std::lock_guard<std::mutex> g(m);
    if (!Mine(cmd)) return;
    if (scene == 0) {
      if (n <= 6 && IsGradedTarget(cmd->get_device(), rt)) {
        scene = rt;
        scene_state = D3D12_RESOURCE_STATE_RENDER_TARGET;
      }
    } else {
      fire = rt == scene && n > 6;
    }
  }
  if (fire) Trigger(cmd, "a UI draw into the frame");
}

inline bool OnDraw(reshade::api::command_list* cmd, uint32_t v, uint32_t, uint32_t, uint32_t) {
  Draw(cmd, v);
  return false;
}
inline bool OnDrawIndexed(reshade::api::command_list* cmd, uint32_t i, uint32_t, uint32_t, int32_t, uint32_t) {
  Draw(cmd, i);
  return false;
}

inline void OnBarrier(reshade::api::command_list* cmd, uint32_t count, const reshade::api::resource* res,
                      const reshade::api::resource_usage*, const reshade::api::resource_usage* to) {
  if (!armed.load(std::memory_order_relaxed)) return;
  std::lock_guard<std::mutex> g(m);
  if (!Mine(cmd)) return;
  const uint64_t depth = (uint64_t)nr_runner::graded_guides.depth, motion = (uint64_t)nr_runner::graded_guides.motion;
  for (uint32_t i = 0; i < count; ++i) {
    if (scene != 0 && res[i].handle == scene) scene_state = StateOf(to[i]);
    if (res[i].handle == depth) depth_state = StateOf(to[i]);
    if (res[i].handle == motion) motion_state = StateOf(to[i]);
  }
}

inline void OnBindPipeline(reshade::api::command_list* cmd, reshade::api::pipeline_stage stages,
                           reshade::api::pipeline p) {
  if (!armed.load(std::memory_order_relaxed) || stages == reshade::api::pipeline_stage::all_ray_tracing) return;
  std::lock_guard<std::mutex> g(m);
  if (Mine(cmd)) pso = p.handle;
}

inline void OnBindTables(reshade::api::command_list* cmd, reshade::api::shader_stage stages,
                         reshade::api::pipeline_layout layout, uint32_t first, uint32_t count,
                         const reshade::api::descriptor_table* tables) {
  if (!armed.load(std::memory_order_relaxed)) return;
  std::lock_guard<std::mutex> g(m);
  if (!Mine(cmd)) return;
  Tables& t = (stages & reshade::api::shader_stage::all_graphics) != 0 ? graphics : compute;
  if (layout.handle != t.layout) {
    t.tables.clear();   // a new root signature resets every binding
    t.layout = layout.handle;
  }
  if (t.tables.size() < first + count) t.tables.resize(first + count);
  for (uint32_t i = 0; i < count; ++i) t.tables[first + i] = tables[i].handle;
}

inline void OnPresent() {
  std::lock_guard<std::mutex> g(m);
  if (armed) {
    armed = false;
    ++missed;
  }
  nr_runner::graded_guides_valid = false;
  if (++presents % 600 == 0 && nr_runner::s.graded && nr_runner::s.enabled) {
    ngx_probe::Logf("graded: last 600 frames: pass ran %u, failed %u, graded frame not found %u", ran, failed, missed);
    ran = failed = missed = 0;
  }
}

inline void Register() {
  reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(OnBind);
  reshade::register_event<reshade::addon_event::draw>(OnDraw);
  reshade::register_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
  reshade::register_event<reshade::addon_event::barrier>(OnBarrier);
  reshade::register_event<reshade::addon_event::bind_pipeline>(OnBindPipeline);
  reshade::register_event<reshade::addon_event::bind_descriptor_tables>(OnBindTables);
}

inline void Unregister() {
  reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(OnBind);
  reshade::unregister_event<reshade::addon_event::draw>(OnDraw);
  reshade::unregister_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
  reshade::unregister_event<reshade::addon_event::barrier>(OnBarrier);
  reshade::unregister_event<reshade::addon_event::bind_pipeline>(OnBindPipeline);
  reshade::unregister_event<reshade::addon_event::bind_descriptor_tables>(OnBindTables);
}

}  // namespace graded_hook
