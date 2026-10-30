// Diagnostics for the pass: GPU timestamps around each stage, logged as averages, and a one-shot
// dump of the frame (the original and the final picture) for comparing settings offline.
//
//   timers: start | encode | squeeze | model | expand | resolve, every frame into a ring of
//           readback slots; a slot is read back kRing frames later, when it has surely executed
//   dump:   DumpFrame=<id> in the live ini copies both textures to readback buffers on the next
//           evaluate and writes dlssnr-dump-<id>-{orig,final}.raw (+ .txt header) beside the exe;
//           DumpBurst=N (1-8) makes it N consecutive frames, ids <id> .. <id>+N-1
//           kRing frames later

#pragma once

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <utility>

#include <d3d12.h>

namespace diag {

constexpr uint32_t kRing = 16;   // frames a timestamp slot or a dump waits before it is read back
// The model's span is split per region when our engine runs it (its input block, then its network); with
// NGX, or a region that did not run, those stamps fall at the end of the model. The regions may run in either
// order (the periphery first when the crop reads its scene context): a stage is the time from the stamp just
// before it in time to its own.
enum Stamp : uint32_t { kStart, kEncode, kSqueeze, kCropBase, kCropNet, kLowBase, kLowNet, kModel, kExpand, kResolve,
                        kStamps };
constexpr const char* kStageNames[kStamps - 1] = {"encode", "squeeze", "crop-noise", "crop-net", "low-noise",
                                                  "low-net", "model-rest", "expand", "resolve"};
constexpr uint32_t kReportEvery = 300;

// Ticks per second of the queue's timestamps, from the present hook (0 until known).
inline std::atomic<uint64_t> timestamp_freq{0};

// Set from the live ini; a new id asks for one dump.
inline std::atomic<int> dump_requested{0};

using LogFn = void (*)(const char*);

struct Timers {
  ID3D12QueryHeap* heap = nullptr;
  ID3D12Resource* readback = nullptr;
  bool armed[kRing] = {};     // the slot's queries were all recorded and resolved
  bool open = false;          // Begin ran this frame and End has not
  uint32_t slot = 0;
  double sum[kStamps - 1] = {};
  double sum_total = 0.0;
  uint32_t count = 0;
  LARGE_INTEGER last_begin = {};
  double sum_interval = 0.0;
  uint32_t intervals = 0;
};

inline Timers t;

inline bool CreateReadback(ID3D12Device* device, UINT64 bytes, ID3D12Resource** out) {
  D3D12_HEAP_PROPERTIES heap = {};
  heap.Type = D3D12_HEAP_TYPE_READBACK;
  D3D12_RESOURCE_DESC desc = {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = bytes;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(out)));
}

// Fold the slot recorded kRing frames ago into the averages, then start this frame's queries.
inline void Begin(ID3D12GraphicsCommandList* cmd, uint32_t frame, LogFn log) {
  t.open = false;
  if (t.heap == nullptr) {
    ID3D12Device* device = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return;
    D3D12_QUERY_HEAP_DESC qd = {};
    qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qd.Count = kRing * kStamps;
    const bool ok = SUCCEEDED(device->CreateQueryHeap(&qd, IID_PPV_ARGS(&t.heap))) &&
                    CreateReadback(device, sizeof(uint64_t) * kRing * kStamps, &t.readback);
    device->Release();
    if (!ok) {
      if (t.heap != nullptr) t.heap->Release();
      t.heap = nullptr;
      return;
    }
  }

  LARGE_INTEGER now, qpf;
  QueryPerformanceCounter(&now);
  QueryPerformanceFrequency(&qpf);
  if (t.last_begin.QuadPart != 0) {
    t.sum_interval += double(now.QuadPart - t.last_begin.QuadPart) * 1000.0 / double(qpf.QuadPart);
    ++t.intervals;
  }
  t.last_begin = now;

  t.slot = frame % kRing;
  const uint64_t freq = timestamp_freq.load(std::memory_order_relaxed);
  if (t.armed[t.slot] && freq != 0) {
    D3D12_RANGE range = {t.slot * kStamps * sizeof(uint64_t), (t.slot + 1) * kStamps * sizeof(uint64_t)};
    void* p = nullptr;
    if (SUCCEEDED(t.readback->Map(0, &range, &p))) {
      const uint64_t* ts = (const uint64_t*)((const char*)p + range.Begin);
      bool sane = true;
      for (uint32_t i = 1; i < kStamps; ++i) sane &= ts[i] >= ts[0] && ts[i] <= ts[kStamps - 1];
      if (sane) {
        for (uint32_t i = 1; i < kStamps; ++i) {
          // The stamp just before this one in time (an earlier stamp wins a tie).
          uint64_t prev = ts[0];
          for (uint32_t j = 0; j < kStamps; ++j)
            if (j != i && (ts[j] < ts[i] || (ts[j] == ts[i] && j < i)) && ts[j] > prev) prev = ts[j];
          t.sum[i - 1] += double(ts[i] - prev) * 1000.0 / double(freq);
        }
        t.sum_total += double(ts[kStamps - 1] - ts[0]) * 1000.0 / double(freq);
        ++t.count;
      }
      D3D12_RANGE none = {0, 0};
      t.readback->Unmap(0, &none);
    }
  }
  t.armed[t.slot] = false;

  if (t.count >= kReportEvery) {
    char line[512];
    int n = std::snprintf(line, sizeof(line), "timing: GPU ms/frame");
    for (uint32_t i = 0; i < kStamps - 1; ++i)
      n += std::snprintf(line + n, sizeof(line) - n, " %s %.2f", kStageNames[i], t.sum[i] / t.count);
    std::snprintf(line + n, sizeof(line) - n, " | pass %.2f | evaluate interval %.2f ms (%.1f fps)",
                  t.sum_total / t.count, t.intervals ? t.sum_interval / t.intervals : 0.0,
                  t.intervals ? 1000.0 * t.intervals / t.sum_interval : 0.0);
    log(line);
    for (double& v : t.sum) v = 0.0;
    t.sum_total = t.sum_interval = 0.0;
    t.count = t.intervals = 0;
  }

  cmd->EndQuery(t.heap, D3D12_QUERY_TYPE_TIMESTAMP, t.slot * kStamps + kStart);
  t.open = true;
}

inline void Mark(ID3D12GraphicsCommandList* cmd, Stamp stamp) {
  if (t.open) cmd->EndQuery(t.heap, D3D12_QUERY_TYPE_TIMESTAMP, t.slot * kStamps + stamp);
}

// After the last stage: resolve this frame's stamps into its readback slot.
inline void End(ID3D12GraphicsCommandList* cmd) {
  if (!t.open) return;
  cmd->ResolveQueryData(t.heap, D3D12_QUERY_TYPE_TIMESTAMP, t.slot * kStamps, kStamps, t.readback,
                        t.slot * kStamps * sizeof(uint64_t));
  t.armed[t.slot] = true;
  t.open = false;
}

// --- one GPU span (the game's DLSS-SR evaluate) ---------------------------

// Two timestamps around a call recorded into the game's list, read back kRing frames later and logged as an
// average every kReportEvery frames ("timing: <name> x ms").
struct Span {
  const char* name;
  ID3D12QueryHeap* heap = nullptr;
  ID3D12Resource* readback = nullptr;
  bool armed[kRing] = {};
  uint32_t frame = 0, slot = 0, count = 0;
  double sum = 0.0;
};

inline void SpanStart(Span& sp, ID3D12GraphicsCommandList* cmd, LogFn log) {
  if (sp.heap == nullptr) {
    ID3D12Device* device = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return;
    D3D12_QUERY_HEAP_DESC qd = {};
    qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qd.Count = kRing * 2;
    const bool ok = SUCCEEDED(device->CreateQueryHeap(&qd, IID_PPV_ARGS(&sp.heap))) &&
                    CreateReadback(device, sizeof(uint64_t) * kRing * 2, &sp.readback);
    device->Release();
    if (!ok) {
      if (sp.heap != nullptr) sp.heap->Release();
      sp.heap = nullptr;
      return;
    }
  }
  sp.slot = sp.frame++ % kRing;
  const uint64_t freq = timestamp_freq.load(std::memory_order_relaxed);
  if (sp.armed[sp.slot] && freq != 0) {
    D3D12_RANGE range = {sp.slot * 2 * sizeof(uint64_t), (sp.slot + 1) * 2 * sizeof(uint64_t)};
    void* p = nullptr;
    if (SUCCEEDED(sp.readback->Map(0, &range, &p))) {
      const uint64_t* ts = (const uint64_t*)((const char*)p + range.Begin);
      if (ts[1] >= ts[0]) {
        sp.sum += double(ts[1] - ts[0]) * 1000.0 / double(freq);
        ++sp.count;
      }
      D3D12_RANGE none = {0, 0};
      sp.readback->Unmap(0, &none);
    }
    sp.armed[sp.slot] = false;
  }
  if (sp.count >= kReportEvery) {
    char line[128];
    std::snprintf(line, sizeof(line), "timing: %s %.2f ms/frame (GPU)", sp.name, sp.sum / sp.count);
    log(line);
    sp.sum = 0.0;
    sp.count = 0;
  }
  cmd->EndQuery(sp.heap, D3D12_QUERY_TYPE_TIMESTAMP, sp.slot * 2);
}

inline void SpanStop(Span& sp, ID3D12GraphicsCommandList* cmd) {
  if (sp.heap == nullptr) return;
  cmd->EndQuery(sp.heap, D3D12_QUERY_TYPE_TIMESTAMP, sp.slot * 2 + 1);
  cmd->ResolveQueryData(sp.heap, D3D12_QUERY_TYPE_TIMESTAMP, sp.slot * 2, 2, sp.readback,
                        sp.slot * 2 * sizeof(uint64_t));
  sp.armed[sp.slot] = true;
}

inline Span dlss_sr{"DLSS-SR evaluate"};

// --- frame dump -----------------------------------------------------------

// A dump is a burst of DumpBurst consecutive frames (1-8), written as ids want, want+1, ...
constexpr int kBurstMax = 8;
inline std::atomic<int> dump_burst{1};

struct Slot {
  int id = 0;                 // 0 if free
  uint32_t ready_at = 0;      // frame from which the copies have surely executed
  ID3D12Resource* rb[2] = {};
  char info[512] = {};
};

struct Dump {
  int done = 0;               // last request taken
  int burst_left = 0, next_id = 0;
  Slot slots[kBurstMax];
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
  UINT64 bytes = 0;
};

inline Dump d;

inline void CopyOut(ID3D12GraphicsCommandList* cmd, ID3D12Resource* tex, ID3D12Resource* rb) {
  D3D12_RESOURCE_BARRIER b = {};
  b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  b.Transition.pResource = tex;
  b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
  cmd->ResourceBarrier(1, &b);
  D3D12_TEXTURE_COPY_LOCATION dst = {};
  dst.pResource = rb;
  dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dst.PlacedFootprint = d.fp;
  D3D12_TEXTURE_COPY_LOCATION src = {};
  src.pResource = tex;
  src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  src.SubresourceIndex = 0;
  cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
  cmd->ResourceBarrier(1, &b);
}

// At the end of the pass, with both textures back in UNORDERED_ACCESS. `info` describes the
// settings in effect, for the header.
inline void Capture(ID3D12GraphicsCommandList* cmd, ID3D12Resource* original, ID3D12Resource* final_,
                    uint32_t frame, const char* info, LogFn log) {
  if (d.burst_left == 0) {
    const int want = dump_requested.load(std::memory_order_relaxed);
    if (want == d.done) return;
    for (const Slot& sl : d.slots)
      if (sl.id != 0) return;   // the last burst is still being written
    const int n = dump_burst.load(std::memory_order_relaxed);
    d.burst_left = n < 1 ? 1 : (n > kBurstMax ? kBurstMax : n);
    d.next_id = want;
    d.done = want;
  }
  Slot* slot = nullptr;
  for (Slot& sl : d.slots)
    if (sl.id == 0) { slot = &sl; break; }
  if (slot == nullptr) return;
  ID3D12Device* device = nullptr;
  if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || device == nullptr) return;
  const D3D12_RESOURCE_DESC desc = final_->GetDesc();
  device->GetCopyableFootprints(&desc, 0, 1, 0, &d.fp, nullptr, nullptr, &d.bytes);
  bool ok = CreateReadback(device, d.bytes, &slot->rb[0]) && CreateReadback(device, d.bytes, &slot->rb[1]);
  device->Release();
  if (!ok || original->GetDesc().Format != desc.Format) {
    for (auto*& r : slot->rb) if (r != nullptr) { r->Release(); r = nullptr; }
    d.burst_left = 0;
    log("dump: could not create the readback buffers");
    return;
  }
  CopyOut(cmd, original, slot->rb[0]);
  CopyOut(cmd, final_, slot->rb[1]);
  std::snprintf(slot->info, sizeof(slot->info), "width=%llu\nheight=%u\nformat=%u\nrow_pitch=%u\n%s",
                (unsigned long long)desc.Width, desc.Height, (unsigned)desc.Format,
                d.fp.Footprint.RowPitch, info);
  slot->id = d.next_id++;
  slot->ready_at = frame + kRing;
  --d.burst_left;
}

// Every evaluate: once a dump's copies have executed, write the files beside the exe.
inline void Poll(uint32_t frame, const wchar_t* dir, LogFn log) {
  for (Slot& sl : d.slots) {
    if (sl.id == 0 || frame < sl.ready_at) continue;
    const char* names[2] = {"orig", "final"};
    for (int i = 0; i < 2; ++i) {
      wchar_t path[MAX_PATH];
      swprintf_s(path, L"%s\\dlssnr-dump-%d-%S.raw", dir, sl.id, names[i]);
      void* p = nullptr;
      D3D12_RANGE range = {0, (SIZE_T)d.bytes};
      if (SUCCEEDED(sl.rb[i]->Map(0, &range, &p))) {
        if (FILE* f = _wfopen(path, L"wb")) {
          fwrite(p, 1, (size_t)d.bytes, f);
          fclose(f);
        }
        D3D12_RANGE none = {0, 0};
        sl.rb[i]->Unmap(0, &none);
      }
      sl.rb[i]->Release();
      sl.rb[i] = nullptr;
    }
    wchar_t path[MAX_PATH];
    swprintf_s(path, L"%s\\dlssnr-dump-%d.txt", dir, sl.id);
    if (FILE* f = _wfopen(path, L"wb")) {
      fputs(sl.info, f);
      fclose(f);
    }
    char line[96];
    std::snprintf(line, sizeof(line), "dump: frame %d written", sl.id);
    log(line);
    sl.id = 0;
  }
}

}  // namespace diag
