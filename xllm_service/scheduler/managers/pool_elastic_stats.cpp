/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
==============================================================================*/

#include "scheduler/managers/pool_elastic_stats.h"

#include <algorithm>

namespace xllm_service {

PoolElasticStats::PoolElasticStats() : buckets_(kMaxWindowSeconds) {}

PoolElasticStats& PoolElasticStats::instance() {
  static PoolElasticStats stats;
  return stats;
}

void PoolElasticStats::record_request(bool is_long, uint64_t now_ms) {
  const uint64_t epoch_s = now_ms / 1000;
  const size_t slot = static_cast<size_t>(epoch_s % kMaxWindowSeconds);
  std::lock_guard<std::mutex> lock(mu_);
  Bucket& b = buckets_[slot];
  if (b.epoch_s != epoch_s) {
    b.epoch_s = epoch_s;
    b.total = 0;
    b.longs = 0;
  }
  ++b.total;
  if (is_long) ++b.longs;
}

double PoolElasticStats::compute_long_ratio(int window_seconds,
                                            uint64_t now_ms) const {
  window_seconds = std::clamp(window_seconds, 1, kMaxWindowSeconds);
  const uint64_t epoch_now_s = now_ms / 1000;
  const uint64_t lower_s = epoch_now_s >= static_cast<uint64_t>(window_seconds)
                               ? epoch_now_s - window_seconds + 1
                               : 0;
  uint64_t total = 0;
  uint64_t longs = 0;
  std::lock_guard<std::mutex> lock(mu_);
  for (const Bucket& b : buckets_) {
    if (b.epoch_s >= lower_s && b.epoch_s <= epoch_now_s) {
      total += b.total;
      longs += b.longs;
    }
  }
  if (total == 0) return 0.0;
  return static_cast<double>(longs) / static_cast<double>(total);
}

uint64_t PoolElasticStats::total_in_window(int window_seconds,
                                           uint64_t now_ms) const {
  window_seconds = std::clamp(window_seconds, 1, kMaxWindowSeconds);
  const uint64_t epoch_now_s = now_ms / 1000;
  const uint64_t lower_s = epoch_now_s >= static_cast<uint64_t>(window_seconds)
                               ? epoch_now_s - window_seconds + 1
                               : 0;
  uint64_t total = 0;
  std::lock_guard<std::mutex> lock(mu_);
  for (const Bucket& b : buckets_) {
    if (b.epoch_s >= lower_s && b.epoch_s <= epoch_now_s) {
      total += b.total;
    }
  }
  return total;
}

void PoolElasticStats::reset() {
  std::lock_guard<std::mutex> lock(mu_);
  for (Bucket& b : buckets_) b = Bucket{};
}

}  // namespace xllm_service
