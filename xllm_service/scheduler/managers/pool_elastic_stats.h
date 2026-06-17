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

#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <utility>

namespace xllm_service {

// Rolling-window counter that splits incoming routing decisions into "long"
// and "short" buckets. Used by the pool elasticity controller to decide when
// IDLE elastic instances should be activated and when ACTIVE ones should be
// drained. Singleton: lifetime is the process. All members are thread-safe.
class PoolElasticStats {
 public:
  static PoolElasticStats& instance();

  // Record a routing decision. is_long should track the same long-request
  // threshold the rest of the service uses. count is the number of requests
  // represented by this call (typically 1).
  void record_request(bool is_long, uint64_t count);

  // Fraction of "long" decisions in the rolling window of window_s seconds
  // ending at now_ms. Returns 0.0 when the window is empty.
  double compute_long_ratio(int window_s, uint64_t now_ms) const;

  // Total decisions in the rolling window.
  uint64_t total_in_window(int window_s, uint64_t now_ms) const;

 private:
  PoolElasticStats() = default;
  PoolElasticStats(const PoolElasticStats&) = delete;
  PoolElasticStats& operator=(const PoolElasticStats&) = delete;

  // Each entry: (timestamp_ms, is_long_count_packed)
  // We only keep the last "long" count per ts; deque is amortized O(1) prune.
  struct Entry {
    uint64_t ts_ms;
    bool is_long;
    uint64_t count;
  };

  void prune_locked(uint64_t cutoff_ms) const;

  mutable std::mutex mu_;
  mutable std::deque<Entry> entries_;
};

}  // namespace xllm_service
