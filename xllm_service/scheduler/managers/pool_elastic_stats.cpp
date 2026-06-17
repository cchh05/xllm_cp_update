/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "pool_elastic_stats.h"

#include <chrono>

namespace xllm_service {

PoolElasticStats& PoolElasticStats::instance() {
  static PoolElasticStats inst;
  return inst;
}

void PoolElasticStats::record_request(bool is_long, uint64_t count) {
  if (count == 0) return;
  std::lock_guard<std::mutex> lock(mu_);
  // Use a coarse timestamp: caller-supplied recording time would be ideal but
  // not all callers thread it through. Capture monotonic ms here.
  // Note: the controller passes its own now_ms when querying; the rolling
  // window logic only needs ordering, not millisecond exactness.
  // We rely on system_clock here to match other ts_ms in the codebase.
  uint64_t now_ms = 0;
  {
    auto now = std::chrono::system_clock::now();
    now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                 now.time_since_epoch())
                 .count();
  }
  entries_.push_back({now_ms, is_long, count});
}

void PoolElasticStats::prune_locked(uint64_t cutoff_ms) const {
  while (!entries_.empty() && entries_.front().ts_ms < cutoff_ms) {
    entries_.pop_front();
  }
}

double PoolElasticStats::compute_long_ratio(int window_s,
                                            uint64_t now_ms) const {
  if (window_s <= 0) return 0.0;
  const uint64_t cutoff =
      now_ms > static_cast<uint64_t>(window_s) * 1000
          ? now_ms - static_cast<uint64_t>(window_s) * 1000
          : 0;
  std::lock_guard<std::mutex> lock(mu_);
  prune_locked(cutoff);
  if (entries_.empty()) return 0.0;
  uint64_t total = 0;
  uint64_t longs = 0;
  for (const auto& e : entries_) {
    total += e.count;
    if (e.is_long) longs += e.count;
  }
  if (total == 0) return 0.0;
  return static_cast<double>(longs) / static_cast<double>(total);
}

uint64_t PoolElasticStats::total_in_window(int window_s,
                                           uint64_t now_ms) const {
  if (window_s <= 0) return 0;
  const uint64_t cutoff =
      now_ms > static_cast<uint64_t>(window_s) * 1000
          ? now_ms - static_cast<uint64_t>(window_s) * 1000
          : 0;
  std::lock_guard<std::mutex> lock(mu_);
  prune_locked(cutoff);
  uint64_t total = 0;
  for (const auto& e : entries_) total += e.count;
  return total;
}

}  // namespace xllm_service
