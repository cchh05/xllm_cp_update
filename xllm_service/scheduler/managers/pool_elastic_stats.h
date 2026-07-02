/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
==============================================================================*/

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace xllm_service {

// PoolElasticStats keeps a coarse rolling-window count of long-vs-short
// request decisions made by the scheduler. The pool elasticity controller
// (driven from the reconcile thread) reads `compute_long_ratio` once per tick
// to decide whether to flip elastic prefill instances between IDLE and
// ACTIVE.
//
// Design notes:
//   - Recording is on the per-request hot path, so it must be lock-free in the
//     common case. We use a fixed-size circular buffer of 1-second buckets
//     plus atomics for counters.
//   - Reading happens at most ~1 Hz and tolerates a slightly stale view, so
//     `compute_long_ratio` takes a short mutex. No coordination with hot path.
//   - The window is capped at `kMaxWindowSeconds` (60 s); callers asking for
//     a longer window are clamped silently.
class PoolElasticStats {
 public:
  static constexpr int kMaxWindowSeconds = 60;

  PoolElasticStats();

  // Record one routing decision. `is_long` is the same `request_is_long`
  // value already computed by the scheduler. `now_ms` is the wall-clock time
  // in milliseconds; pass `current_time_ms()` from the caller.
  void record_request(bool is_long, uint64_t now_ms);

  // Returns the fraction of long requests over the last `window_seconds` s,
  // in [0, 1]. Returns 0.0 if the window is empty. `window_seconds` is
  // clamped to [1, kMaxWindowSeconds].
  double compute_long_ratio(int window_seconds, uint64_t now_ms) const;

  // Total request count over the last `window_seconds` s. Useful for the
  // controller to avoid acting on near-empty windows.
  uint64_t total_in_window(int window_seconds, uint64_t now_ms) const;

  // Test-only / introspection: clear all buckets. Not used in prod paths.
  void reset();

  // Process-wide singleton accessor. The stats are intentionally global
  // because both `select_instance_pair_on_slo` and `get_next_instance_pair`
  // record into the same window.
  static PoolElasticStats& instance();

 private:
  struct Bucket {
    uint64_t epoch_s = 0;  // wall second this bucket represents
    uint64_t total = 0;
    uint64_t longs = 0;
  };

  // 1-second granularity, kMaxWindowSeconds buckets. Indexed by
  // (epoch_s % kMaxWindowSeconds). Stale buckets are detected by epoch_s
  // mismatch and overwritten.
  mutable std::mutex mu_;
  std::vector<Bucket> buckets_;
};

}  // namespace xllm_service
