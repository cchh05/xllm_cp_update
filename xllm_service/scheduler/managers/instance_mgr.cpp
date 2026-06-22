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

#include "instance_mgr.h"

#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/global_gflags.h"
#include "common/types.h"
#include "common/utils.h"
#include "common/xllm/output.h"
#include "common/xllm/status.h"
#include "disagg_pd.pb.h"
#include "scheduler/managers/pool_elastic_stats.h"
#include "scheduler/scheduler.h"

namespace {
using xllm_service::InstanceRuntimeState;
using xllm_service::InstanceType;
std::unordered_map<InstanceType, std::string> ETCD_KEYS_PREFIX_MAP = {
    {InstanceType::DEFAULT, "XLLM:DEFAULT:"},
    {InstanceType::PREFILL, "XLLM:PREFILL:"},
    {InstanceType::DECODE, "XLLM:DECODE:"},
    {InstanceType::MIX, "XLLM:MIX:"},
};

std::string ETCD_ALL_KEYS_PREFIX = "XLLM:";
std::string ETCD_LOADMETRICS_PREFIX = "XLLM:LOADMETRICS:";

constexpr char kHealthPath[] = "/health";
constexpr int64_t kDeleteProbeRetryBackoffMs = 100;

double clamp_unit_interval(double value) {
  return std::max(0.0, std::min(1.0, value));
}

enum class DecodeAdmissionPressureTier : int8_t {
  kOpen = 0,
  kMedium = 1,
  kHard = 2,
};

const char* decode_admission_pressure_tier_name(
    DecodeAdmissionPressureTier tier) {
  switch (tier) {
    case DecodeAdmissionPressureTier::kOpen:
      return "open";
    case DecodeAdmissionPressureTier::kMedium:
      return "medium";
    case DecodeAdmissionPressureTier::kHard:
      return "hard";
  }
  return "unknown";
}

struct DecodeAdmissionCandidateState {
  std::string instance_name;
  int64_t active_requests = 0;
  int64_t waiting_requests = 0;
  int64_t estimated_tpot_ms = -1;
  bool has_tpot_signal = false;
  DecodeAdmissionPressureTier pressure_tier =
      DecodeAdmissionPressureTier::kOpen;
};

uint64_t current_time_ms() {
  return static_cast<uint64_t>(
      absl::ToInt64Milliseconds(absl::Now() - absl::UnixEpoch()));
}

int64_t estimate_decode_work_tokens(const std::shared_ptr<xllm_service::Request>& request,
                                    int64_t prompt_tokens,
                                    bool request_is_long) {
  if (!FLAGS_decode_pressure_admission_use_output_work) {
    return prompt_tokens;
  }
  if (request != nullptr && request->max_tokens > 0) {
    return request->max_tokens;
  }
  const int64_t default_output_tokens =
      request_is_long ? FLAGS_decode_pressure_admission_default_long_output_tokens
                      : FLAGS_decode_pressure_admission_default_short_output_tokens;
  return default_output_tokens > 0 ? default_output_tokens : prompt_tokens;
}

DecodeAdmissionPressureTier evaluate_decode_admission_pressure_tier(
    const DecodeAdmissionCandidateState& candidate,
    bool request_is_long,
    int64_t base_active_cap,
    int64_t effective_target_tpot_ms) {
  const int64_t hard_active_cap =
      std::max<int64_t>(0,
                        base_active_cap +
                            FLAGS_decode_pressure_admission_hard_active_burst);
  const int64_t short_active_cap =
      std::max<int64_t>(0,
                        base_active_cap +
                            FLAGS_decode_pressure_admission_short_active_burst);
  const bool hard_waiting =
      FLAGS_decode_pressure_admission_hard_waiting_requests > 0 &&
      candidate.waiting_requests >=
          FLAGS_decode_pressure_admission_hard_waiting_requests;
  const bool hard_active =
      base_active_cap > 0 && candidate.active_requests >= hard_active_cap;
  bool hard_tpot = false;
  bool medium_tpot = false;
  if (candidate.has_tpot_signal && effective_target_tpot_ms > 0) {
    const double estimated_tpot_ms =
        static_cast<double>(candidate.estimated_tpot_ms);
    hard_tpot = estimated_tpot_ms >=
                static_cast<double>(effective_target_tpot_ms) *
                    FLAGS_decode_pressure_admission_hard_tpot_ratio;
    medium_tpot =
        estimated_tpot_ms >=
        static_cast<double>(effective_target_tpot_ms) *
            FLAGS_decode_pressure_admission_soft_tpot_ratio;
  }
  if (hard_waiting || hard_active || hard_tpot) {
    return DecodeAdmissionPressureTier::kHard;
  }

  const bool medium_waiting =
      FLAGS_decode_pressure_admission_medium_waiting_requests > 0 &&
      candidate.waiting_requests >=
          FLAGS_decode_pressure_admission_medium_waiting_requests;
  const bool medium_active =
      base_active_cap > 0 &&
      (request_is_long ? candidate.active_requests >= base_active_cap
                       : candidate.active_requests >= short_active_cap);
  if (medium_waiting || medium_active || medium_tpot) {
    return DecodeAdmissionPressureTier::kMedium;
  }
  return DecodeAdmissionPressureTier::kOpen;
}

double lane_normalized_projected_prefill_time_penalty(
    int64_t projected_prefill_time_ms,
    bool is_long_affine,
    bool is_short_affine) {
  if (!FLAGS_enable_lane_normalized_projected_prefill_time_route_signal ||
      FLAGS_hybrid_prefill_projected_prefill_time_normalized_weight <= 0.0 ||
      projected_prefill_time_ms <= 0) {
    return 0.0;
  }
  double baseline_ms = 1.0;
  if (is_long_affine) {
    baseline_ms =
        FLAGS_hybrid_prefill_long_lane_projected_prefill_time_baseline_ms;
  } else if (is_short_affine) {
    baseline_ms =
        FLAGS_hybrid_prefill_short_pool_projected_prefill_time_baseline_ms;
  }
  if (baseline_ms <= 0.0) {
    return 0.0;
  }
  double normalized_time =
      static_cast<double>(projected_prefill_time_ms) / baseline_ms;
  if (FLAGS_hybrid_prefill_projected_prefill_time_normalized_cap > 0.0) {
    normalized_time =
        std::min(normalized_time,
                 FLAGS_hybrid_prefill_projected_prefill_time_normalized_cap);
  }
  return normalized_time *
         FLAGS_hybrid_prefill_projected_prefill_time_normalized_weight;
}

std::string get_client_request_id(const std::shared_ptr<xllm_service::Request>& request) {
  if (request == nullptr) {
    return "";
  }
  if (!request->client_request_id.empty()) {
    return request->client_request_id;
  }
  if (request->call_data == nullptr) {
    return "";
  }
  return request->call_data->x_request_id;
}

bool should_sample_route_trace(const bool request_is_long,
                               uint64_t* sample_ordinal) {
  if (sample_ordinal != nullptr) {
    *sample_ordinal = 0;
  }
  if (!FLAGS_enable_route_trace) {
    return false;
  }
  if (FLAGS_route_trace_long_requests_only && !request_is_long) {
    return false;
  }

  static std::atomic<uint64_t> route_trace_request_counter{0};
  const int32_t sample_every_n = std::max(1, FLAGS_route_trace_sample_every_n);
  const uint64_t ordinal =
      route_trace_request_counter.fetch_add(1, std::memory_order_relaxed) + 1;
  if (((ordinal - 1) % static_cast<uint64_t>(sample_every_n)) != 0) {
    return false;
  }

  if (sample_ordinal != nullptr) {
    *sample_ordinal = ordinal;
  }
  return true;
}

void append_route_trace_jsonl(const nlohmann::json& trace_entry) {
  static std::mutex route_trace_mutex;
  std::lock_guard<std::mutex> lock(route_trace_mutex);

  const std::filesystem::path log_path(FLAGS_route_trace_log_path);
  if (!log_path.parent_path().empty()) {
    std::error_code ec;
    std::filesystem::create_directories(log_path.parent_path(), ec);
    if (ec) {
      LOG(ERROR) << "Failed to create route trace directory: "
                 << log_path.parent_path() << ", error: " << ec.message();
      return;
    }
  }

  std::ofstream log_stream(log_path, std::ios::app);
  if (!log_stream.is_open()) {
    LOG(ERROR) << "Failed to open route trace log file: " << log_path;
    return;
  }
  log_stream << trace_entry.dump() << "\n";
}

bool is_instance_schedulable(const xllm_service::InstanceMetaInfo& info) {
  // LEASE_LOST instances can still be reused while heartbeats continue.
  // REGISTERING instances are visible for heartbeats/recovery, but must not be
  // selected until their registration/link phase finishes.
  // IDLE / DRAINING instances are kept alive (process, weights, KV cache) but
  // intentionally excluded from the candidate set by the pool elasticity
  // controller.
  return info.runtime_state != InstanceRuntimeState::SUSPECT &&
         info.runtime_state != InstanceRuntimeState::REGISTERING &&
         info.runtime_state != InstanceRuntimeState::IDLE &&
         info.runtime_state != InstanceRuntimeState::DRAINING;
}

bool select_next_schedulable_instance(
    const std::unordered_map<std::string, xllm_service::InstanceMetaInfo>&
        instances,
    const std::vector<std::string>& index,
    uint64_t* next_index,
    std::string* instance_name) {
  if (index.empty()) {
    return false;
  }

  const uint64_t start_index = *next_index % index.size();
  for (uint64_t offset = 0; offset < index.size(); ++offset) {
    const uint64_t index_pos = (start_index + offset) % index.size();
    auto it = instances.find(index[index_pos]);
    if (it == instances.end() || !is_instance_schedulable(it->second)) {
      continue;
    }
    *instance_name = index[index_pos];
    *next_index = index_pos + 1;
    return true;
  }
  return false;
}

size_t count_schedulable_instances(
    const std::unordered_map<std::string, xllm_service::InstanceMetaInfo>&
        instances,
    const std::vector<std::string>& index) {
  size_t count = 0;
  for (const auto& name : index) {
    auto it = instances.find(name);
    if (it == instances.end() || !is_instance_schedulable(it->second)) {
      continue;
    }
    ++count;
  }
  return count;
}

std::vector<std::string> get_schedulable_prefill_instances(
    const std::unordered_map<std::string, xllm_service::InstanceMetaInfo>&
        instances,
    const std::vector<std::string>& prefill_index,
    bool /*has_unschedulable_instances*/) {
  // Pool-elastic correctness: we must always go through is_instance_schedulable
  // so IDLE / DRAINING / SUSPECT / REGISTERING are uniformly excluded. A prior
  // fast-path that skipped the predicate when suspect_instances_ was empty
  // silently let IDLE instances back into candidate sets and broke Line 1.
  std::vector<std::string> result;
  result.reserve(prefill_index.size());
  for (const auto& name : prefill_index) {
    auto it = instances.find(name);
    if (it == instances.end() || !is_instance_schedulable(it->second)) {
      continue;
    }
    result.emplace_back(name);
  }
  return result;
}

std::vector<std::string> parse_instance_selectors(const std::string& csv) {
  std::vector<std::string> selectors;
  for (absl::string_view item : absl::StrSplit(csv, ',', absl::SkipWhitespace())) {
    if (!item.empty()) {
      selectors.emplace_back(item);
    }
  }
  return selectors;
}

bool is_digits_only(const std::string& value) {
  return !value.empty() &&
         std::all_of(value.begin(), value.end(), [](unsigned char ch) {
           return std::isdigit(ch) != 0;
         });
}

bool instance_matches_selector(const std::string& instance_name,
                               const std::string& selector) {
  if (selector.empty()) {
    return false;
  }
  if (instance_name == selector) {
    return true;
  }

  std::string suffix;
  if (selector.front() == ':') {
    suffix = selector;
  } else if (is_digits_only(selector)) {
    suffix = ":" + selector;
  } else {
    return false;
  }

  if (instance_name.size() < suffix.size()) {
    return false;
  }
  return instance_name.compare(instance_name.size() - suffix.size(),
                               suffix.size(),
                               suffix) == 0;
}

std::vector<std::string> filter_prefill_instances_by_selector(
    const std::vector<std::string>& instances,
    const std::vector<std::string>& selectors) {
  if (selectors.empty()) {
    return {};
  }

  std::vector<std::string> filtered;
  filtered.reserve(instances.size());
  for (const auto& name : instances) {
    if (std::any_of(selectors.begin(),
                    selectors.end(),
                    [&](const std::string& selector) {
                      return instance_matches_selector(name, selector);
                    })) {
      filtered.emplace_back(name);
    }
  }
  return filtered;
}

int64_t get_prefill_request_num(
    const std::unordered_map<std::string, xllm_service::RequestMetrics>&
        request_metrics,
    const std::string& instance_name) {
  const auto it = request_metrics.find(instance_name);
  return it == request_metrics.end() ? 0 : it->second.prefill_request_num;
}

int64_t get_prefill_token_num(
    const std::unordered_map<std::string, xllm_service::RequestMetrics>&
        request_metrics,
    const std::string& instance_name) {
  const auto it = request_metrics.find(instance_name);
  return it == request_metrics.end() ? 0 : it->second.prefill_token_num;
}

uint64_t get_waiting_requests_num(
    const std::unordered_map<std::string, xllm_service::LoadMetrics>&
        load_metrics,
    const std::string& instance_name) {
  const auto it = load_metrics.find(instance_name);
  return it == load_metrics.end() ? 0 : it->second.waiting_requests_num;
}

uint64_t get_prefill_combined_load(
    const std::unordered_map<std::string, xllm_service::RequestMetrics>&
        request_metrics,
    const std::unordered_map<std::string, xllm_service::LoadMetrics>&
        load_metrics,
    const std::string& instance_name) {
  return get_waiting_requests_num(load_metrics, instance_name) +
         static_cast<uint64_t>(
             std::max<int64_t>(0,
                               get_prefill_request_num(request_metrics,
                                                       instance_name)));
}

int64_t min_prefill_request_num(
    const std::vector<std::string>& instances,
    const std::unordered_map<std::string, xllm_service::RequestMetrics>&
        request_metrics) {
  int64_t min_value = std::numeric_limits<int64_t>::max();
  for (const auto& instance_name : instances) {
    min_value = std::min(min_value,
                         get_prefill_request_num(request_metrics, instance_name));
  }
  return min_value == std::numeric_limits<int64_t>::max() ? 0 : min_value;
}

int64_t min_prefill_token_num(
    const std::vector<std::string>& instances,
    const std::unordered_map<std::string, xllm_service::RequestMetrics>&
        request_metrics) {
  int64_t min_value = std::numeric_limits<int64_t>::max();
  for (const auto& instance_name : instances) {
    min_value = std::min(min_value,
                         get_prefill_token_num(request_metrics, instance_name));
  }
  return min_value == std::numeric_limits<int64_t>::max() ? 0 : min_value;
}

uint64_t min_prefill_combined_load(
    const std::vector<std::string>& instances,
    const std::unordered_map<std::string, xllm_service::RequestMetrics>&
        request_metrics,
    const std::unordered_map<std::string, xllm_service::LoadMetrics>&
        load_metrics) {
  uint64_t min_value = std::numeric_limits<uint64_t>::max();
  for (const auto& instance_name : instances) {
    min_value = std::min(
        min_value,
        get_prefill_combined_load(request_metrics, load_metrics, instance_name));
  }
  return min_value == std::numeric_limits<uint64_t>::max() ? 0 : min_value;
}

bool use_hybrid_prefill_scoring_v0_3_family() {
  return FLAGS_hybrid_prefill_scoring_version == "v0_3" ||
         FLAGS_hybrid_prefill_scoring_version == "v0.3" ||
         FLAGS_hybrid_prefill_scoring_version == "v0_3a" ||
         FLAGS_hybrid_prefill_scoring_version == "v0.3a" ||
         FLAGS_hybrid_prefill_scoring_version == "v0_3b" ||
         FLAGS_hybrid_prefill_scoring_version == "v0.3b" ||
         FLAGS_hybrid_prefill_scoring_version == "v0_3c" ||
         FLAGS_hybrid_prefill_scoring_version == "v0.3c";
}

bool use_hybrid_prefill_scoring_v0_3a() {
  return FLAGS_hybrid_prefill_scoring_version == "v0_3a" ||
         FLAGS_hybrid_prefill_scoring_version == "v0.3a";
}

bool use_hybrid_prefill_scoring_v0_3b() {
  return FLAGS_hybrid_prefill_scoring_version == "v0_3b" ||
         FLAGS_hybrid_prefill_scoring_version == "v0.3b";
}

bool use_hybrid_prefill_scoring_v0_3c() {
  return FLAGS_hybrid_prefill_scoring_version == "v0_3c" ||
         FLAGS_hybrid_prefill_scoring_version == "v0.3c";
}

int32_t resolve_v0_3_soft_long_lower_tokens(const int32_t hard_long_threshold) {
  if (FLAGS_hybrid_prefill_v0_3_soft_long_lower_tokens > 0) {
    return FLAGS_hybrid_prefill_v0_3_soft_long_lower_tokens;
  }
  return std::max<int32_t>(0, hard_long_threshold);
}

int32_t resolve_v0_3_soft_long_upper_tokens(const int32_t hard_long_threshold,
                                            const int32_t lower_tokens) {
  if (FLAGS_hybrid_prefill_v0_3_soft_long_upper_tokens > lower_tokens) {
    return FLAGS_hybrid_prefill_v0_3_soft_long_upper_tokens;
  }

  const int32_t fallback_upper =
      std::max<int32_t>(hard_long_threshold * 2, lower_tokens + 1);
  return fallback_upper;
}

double compute_v0_3_long_request_scale(const int32_t request_tokens,
                                       const int32_t lower_tokens,
                                       const int32_t upper_tokens) {
  if (upper_tokens <= lower_tokens) {
    return request_tokens > lower_tokens ? 1.0 : 0.0;
  }
  if (request_tokens <= lower_tokens) {
    return 0.0;
  }
  if (request_tokens >= upper_tokens) {
    return 1.0;
  }
  return static_cast<double>(request_tokens - lower_tokens) /
         static_cast<double>(upper_tokens - lower_tokens);
}

std::vector<std::string> merge_unique_instance_lists(
    const std::vector<std::string>& primary,
    const std::vector<std::string>& secondary) {
  std::vector<std::string> merged;
  merged.reserve(primary.size() + secondary.size());
  std::unordered_set<std::string> seen;
  seen.reserve(primary.size() + secondary.size());
  for (const auto& instance_name : primary) {
    if (seen.insert(instance_name).second) {
      merged.emplace_back(instance_name);
    }
  }
  for (const auto& instance_name : secondary) {
    if (seen.insert(instance_name).second) {
      merged.emplace_back(instance_name);
    }
  }
  return merged;
}

bool instance_in_list(const std::vector<std::string>& instances,
                      const std::string& instance_name) {
  return std::find(instances.begin(), instances.end(), instance_name) !=
         instances.end();
}

InstanceType get_cleanup_type(const xllm_service::InstanceMetaInfo& info) {
  if (info.type == InstanceType::DEFAULT) {
    return InstanceType::PREFILL;
  }
  if (info.type == InstanceType::MIX) {
    return info.current_type;
  }
  return info.type;
}
}  // namespace

namespace xllm_service {

InstanceMgr::InstanceMgr(const Options& options,
                         const std::shared_ptr<EtcdClient>& etcd_client,
                         const bool is_master_service,
                         Scheduler* scheduler)
    : options_(options),
      is_master_service_(is_master_service),
      etcd_client_(etcd_client),
      scheduler_(scheduler) {
  auto handle_instance_metainfo =
      std::bind(&InstanceMgr::update_instance_metainfo,
                this,
                std::placeholders::_1,
                std::placeholders::_2);
  for (auto& it : ETCD_KEYS_PREFIX_MAP) {
    etcd_client_->add_watch(it.second, handle_instance_metainfo);
  }
  if (!is_master_service_) {
    auto handle_load_metrics = std::bind(&InstanceMgr::update_load_metrics,
                                         this,
                                         std::placeholders::_1,
                                         std::placeholders::_2);
    etcd_client_->add_watch(ETCD_LOADMETRICS_PREFIX, handle_load_metrics);
  }

  init();

  state_reconcile_thread_ = std::make_unique<std::thread>(
      &InstanceMgr::reconcile_instance_states, this);
}

void InstanceMgr::init() {
  std::unordered_map<std::string, InstanceMetaInfo> loaded_instances;
  for (auto& it : ETCD_KEYS_PREFIX_MAP) {
    etcd_client_->get_prefix(it.second, &loaded_instances);
  }
  LOG(INFO) << "Load instance info from etcd:" << loaded_instances.size();

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    prefill_index_.reserve(loaded_instances.size());
    decode_index_.reserve(loaded_instances.size());
  }

  for (auto& pair : loaded_instances) {
    if (!register_instance(pair.first, pair.second)) {
      LOG(ERROR) << "Fail to register instance: " << pair.first;
    }
  }

  std::unordered_map<std::string, LoadMetrics> loaded_metrics;
  etcd_client_->get_prefix(ETCD_LOADMETRICS_PREFIX, &loaded_metrics);
  {
    std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
    load_metrics_ = std::move(loaded_metrics);
  }

  {
    std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
    for (int i = 0; i < prefill_index_.size(); i++) {
      LOG(INFO) << i << " : " << prefill_index_[i];
    }
  }
}

InstanceMgr::~InstanceMgr() {
  exited_ = true;
  if (state_reconcile_thread_ && state_reconcile_thread_->joinable()) {
    state_reconcile_thread_->join();
  }
}

InstanceMetaInfo InstanceMgr::get_instance_info(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Get instance info failed, instance is not registered, "
                  "instance_name: "
               << instance_name;
    return InstanceMetaInfo();
  }
  return instances_[instance_name];
}

bool InstanceMgr::can_route_prefill_without_decode_locked(
    const std::string& prefill_name) const {
  auto selected_it = instances_.find(prefill_name);
  if (selected_it == instances_.end() ||
      selected_it->second.type != InstanceType::DEFAULT) {
    LOG(ERROR) << "No decode instance and selected prefill is not default, "
               << "instance: " << prefill_name;
    return false;
  }
  return true;
}

bool InstanceMgr::get_next_instance_pair(const std::shared_ptr<Request>& request) {
  if (request == nullptr) {
    LOG(ERROR) << "Request is null when selecting RR instance pair";
    return false;
  }

  Routing* routing = &request->routing;
  std::unique_lock<std::shared_mutex> cluster_lock(cluster_mutex_);
  std::shared_lock<std::shared_mutex> metrics_lock(metrics_mutex_);
  if (prefill_index_.empty()) {
    LOG(ERROR) << "No prefill or default instance found!";
    return false;
  }

  const bool request_is_long =
      request->token_ids.size() >=
      static_cast<size_t>(std::max<int32_t>(
          0, options_.long_request_threshold_tokens()));
  const int32_t request_tokens =
      static_cast<int32_t>(request->token_ids.size());
  PoolElasticStats::instance().record_request(request_is_long,
                                              current_time_ms());
  uint64_t route_trace_sample_ordinal = 0;
  const bool route_trace_sampled =
      should_sample_route_trace(request_is_long, &route_trace_sample_ordinal);
  const bool has_unschedulable_instances = !suspect_instances_.empty();
  const std::vector<std::string> candidate_prefill_instances =
      get_schedulable_prefill_instances(
          instances_, prefill_index_, has_unschedulable_instances);
  const std::vector<std::string> short_selectors =
      parse_instance_selectors(FLAGS_static_prefill_short_instance_selectors);
  const std::vector<std::string> long_selectors =
      parse_instance_selectors(FLAGS_static_prefill_long_instance_selectors);
  const std::vector<std::string> short_filtered_instances =
      filter_prefill_instances_by_selector(candidate_prefill_instances,
                                          short_selectors);
  const std::vector<std::string> long_filtered_instances =
      filter_prefill_instances_by_selector(candidate_prefill_instances,
                                          long_selectors);
  const std::vector<std::string>& affine_candidates =
      request_is_long ? long_filtered_instances : short_filtered_instances;
  const std::vector<std::string>& primary_selectors =
      request_is_long ? long_selectors : short_selectors;

  routing->decode_name.clear();
  std::string selection_reason = "rr_fast_path_no_suspect";
  if (suspect_instances_.empty()) {
    // Fast path for the common case: no suspect instances, keep plain RR.
    next_prefill_index_ = next_prefill_index_ % prefill_index_.size();
    routing->prefill_name = prefill_index_[next_prefill_index_];
    next_prefill_index_++;

    if (!decode_index_.empty()) {
      next_decode_index_ = next_decode_index_ % decode_index_.size();
      routing->decode_name = decode_index_[next_decode_index_];
      next_decode_index_++;
    }
  } else {
    if (!select_next_schedulable_instance(instances_,
                                          prefill_index_,
                                          &next_prefill_index_,
                                          &routing->prefill_name)) {
      LOG(ERROR) << "No schedulable prefill or default instance found!";
      return false;
    }

    if (!decode_index_.empty()) {
      select_next_schedulable_instance(instances_,
                                       decode_index_,
                                       &next_decode_index_,
                                       &routing->decode_name);
    }
  }

  if (has_unschedulable_instances) {
    selection_reason = "rr_schedulable_scan";
  }

  if (decode_index_.empty()) {
    if (!can_route_prefill_without_decode_locked(routing->prefill_name)) {
      return false;
    }
  }

  if (route_trace_sampled) {
    nlohmann::json selected_prefill_affine = nullptr;
    if (!primary_selectors.empty()) {
      selected_prefill_affine =
          instance_in_list(affine_candidates, routing->prefill_name);
    }

    nlohmann::json candidate_breakdown = nlohmann::json::array();
    for (const auto& prefill_instance : candidate_prefill_instances) {
      candidate_breakdown.emplace_back(nlohmann::json{
          {"instance", prefill_instance},
          {"active_requests",
           get_prefill_request_num(request_metrics_, prefill_instance)},
          {"waiting_requests",
           get_waiting_requests_num(load_metrics_, prefill_instance)},
          {"combined_load",
           get_prefill_combined_load(
               request_metrics_, load_metrics_, prefill_instance)},
          {"current_prefill_tokens",
           get_prefill_token_num(request_metrics_, prefill_instance)}});
    }

    nlohmann::json route_trace_entry{
        {"timestamp_ms", current_time_ms()},
        {"route_path", "rr"},
        {"selection_reason", selection_reason},
        {"sample_ordinal", route_trace_sample_ordinal},
        {"client_request_id", get_client_request_id(request)},
        {"service_request_id", request->service_request_id},
        {"request_tokens", request_tokens},
        {"request_type", request_is_long ? "long" : "short"},
        {"selected_prefill_instance", routing->prefill_name},
        {"selected_prefill_instance_final", routing->prefill_name},
        {"selected_decode_instance", routing->decode_name},
        {"candidate_prefill_instances", candidate_prefill_instances},
        {"short_candidates", short_filtered_instances},
        {"long_candidates", long_filtered_instances},
        {"affine_candidates", affine_candidates},
        {"selected_prefill_affine", selected_prefill_affine},
        {"pool_long_ratio_30s",
         PoolElasticStats::instance().compute_long_ratio(
             FLAGS_pool_elastic_window_s, current_time_ms())},
        {"pool_active_set", snapshot_active_prefill_instances_locked()},
        {"pool_pressure_now", compute_pool_pressure_locked()},
        {"pool_pressure_long_lane", compute_pool_pressure_lane_locked(1)},
        {"pool_pressure_short_lane", compute_pool_pressure_lane_locked(2)},
        {"pool_elastic_use_pressure_signal",
         FLAGS_pool_elastic_use_pressure_signal},
        {"pool_state_change_event",
         build_pool_state_change_event_json(
             FLAGS_pool_elastic_window_s * 1000)},
        {"candidate_breakdown", candidate_breakdown}};
    append_route_trace_jsonl(route_trace_entry);
  }

  return true;
}

// TODO: refactor later, currently return all decode instances
std::vector<std::string> InstanceMgr::get_static_decode_list(
    const std::string& instance_name) {
  std::vector<std::string> decode_list;
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  for (auto& inst : instances_) {
    if (inst.second.type == InstanceType::DECODE &&
        is_instance_schedulable(inst.second)) {
      decode_list.emplace_back(inst.second.name);
    }
  }

  return decode_list;
}

bool InstanceMgr::wait_for_decode_pressure_admission(
    const std::shared_ptr<Request>& request) {
  if (!FLAGS_enable_decode_pressure_admission_control) {
    return true;
  }

  const bool use_active_cap =
      FLAGS_decode_pressure_admission_max_active_requests > 0;
  const bool use_estimated_tpot =
      FLAGS_decode_pressure_admission_use_estimated_tpot;
  if (!use_active_cap && !use_estimated_tpot) {
    return true;
  }

  const int64_t request_tokens =
      request == nullptr ? 0 : static_cast<int64_t>(request->token_ids.size());
  const bool request_is_long =
      request_tokens >= static_cast<int64_t>(
                            std::max<int32_t>(0,
                                              FLAGS_long_request_threshold_tokens));
  const int64_t decode_work_tokens =
      estimate_decode_work_tokens(request, request_tokens, request_is_long);
  const int64_t effective_target_tpot =
      FLAGS_decode_pressure_admission_target_tpot_ms > 0
          ? FLAGS_decode_pressure_admission_target_tpot_ms
          : FLAGS_target_tpot;
  const uint64_t start_ms = current_time_ms();
  const uint64_t timeout_ms =
      static_cast<uint64_t>(FLAGS_decode_pressure_admission_timeout_ms);
  int32_t attempts = 0;
  int64_t last_best_active = -1;
  int64_t last_best_waiting = -1;
  int64_t last_best_estimated_tpot = -1;
  std::string last_best_decode_instance;
  DecodeAdmissionPressureTier last_best_pressure_tier =
      DecodeAdmissionPressureTier::kOpen;

  while (true) {
    bool admitted = false;
    bool has_decode_candidate = false;
    DecodeAdmissionCandidateState best_candidate;
    bool best_candidate_initialized = false;

    {
      std::scoped_lock<std::shared_mutex, std::shared_mutex> lock(
          cluster_mutex_, metrics_mutex_);
      for (const auto& decode_instance : decode_index_) {
        auto instance_it = instances_.find(decode_instance);
        if (instance_it == instances_.end() ||
            !is_instance_schedulable(instance_it->second)) {
          continue;
        }

        has_decode_candidate = true;
        DecodeAdmissionCandidateState candidate;
        candidate.instance_name = decode_instance;
        candidate.active_requests =
            request_metrics_[decode_instance].decode_request_num;
        candidate.waiting_requests = static_cast<int64_t>(
            get_waiting_requests_num(load_metrics_, decode_instance));
        const int64_t token_num =
            request_metrics_[decode_instance].decode_token_num;
        if (use_estimated_tpot &&
            !instance_it->second.tpot_profiling_data.empty()) {
          candidate.estimated_tpot_ms =
              get_time_predictor(decode_instance)
                  .predict_tpot(token_num + decode_work_tokens,
                                candidate.active_requests + 1);
          candidate.has_tpot_signal = candidate.estimated_tpot_ms > 0;
        }
        candidate.pressure_tier = evaluate_decode_admission_pressure_tier(
            candidate,
            request_is_long,
            FLAGS_decode_pressure_admission_max_active_requests,
            effective_target_tpot);

        if (!best_candidate_initialized ||
            candidate.pressure_tier < best_candidate.pressure_tier ||
            (candidate.pressure_tier == best_candidate.pressure_tier &&
             candidate.active_requests < best_candidate.active_requests) ||
            (candidate.pressure_tier == best_candidate.pressure_tier &&
             candidate.active_requests == best_candidate.active_requests &&
             candidate.waiting_requests < best_candidate.waiting_requests) ||
            (candidate.pressure_tier == best_candidate.pressure_tier &&
             candidate.active_requests == best_candidate.active_requests &&
             candidate.waiting_requests == best_candidate.waiting_requests &&
             candidate.estimated_tpot_ms < best_candidate.estimated_tpot_ms)) {
          best_candidate = candidate;
          best_candidate_initialized = true;
        }

        if (candidate.pressure_tier == DecodeAdmissionPressureTier::kOpen) {
          admitted = true;
          break;
        }
        if (!request_is_long &&
            candidate.pressure_tier == DecodeAdmissionPressureTier::kMedium) {
          admitted = true;
          break;
        }
      }
    }

    if (!has_decode_candidate) {
      LOG(ERROR) << "Decode pressure admission found no schedulable decode "
                    "instance.";
      return false;
    }
    last_best_active = best_candidate_initialized ? best_candidate.active_requests
                                                  : -1;
    last_best_waiting = best_candidate_initialized
                            ? best_candidate.waiting_requests
                            : -1;
    last_best_estimated_tpot =
        best_candidate_initialized ? best_candidate.estimated_tpot_ms : -1;
    last_best_decode_instance =
        best_candidate_initialized ? best_candidate.instance_name : "";
    last_best_pressure_tier = best_candidate_initialized
                                  ? best_candidate.pressure_tier
                                  : DecodeAdmissionPressureTier::kHard;

    if (FLAGS_enable_decode_pressure_soft_signal_only) {
      if (request != nullptr) {
        request->stage_timing_trace.decode_admission_wait_ms =
            static_cast<int64_t>(current_time_ms() - start_ms);
        request->stage_timing_trace.decode_admission_attempts = attempts;
        request->stage_timing_trace.decode_admission_active_requests =
            last_best_active;
        request->stage_timing_trace.decode_admission_waiting_requests =
            last_best_waiting;
        request->stage_timing_trace.decode_admission_estimated_tpot_ms =
            last_best_estimated_tpot;
        request->stage_timing_trace.decode_admission_pressure_tier =
            decode_admission_pressure_tier_name(last_best_pressure_tier);
        request->stage_timing_trace
            .decode_admission_preferred_decode_instance =
            last_best_decode_instance;
      }
      return true;
    }

    if (admitted) {
      if (request != nullptr) {
        request->stage_timing_trace.decode_admission_wait_ms =
            static_cast<int64_t>(current_time_ms() - start_ms);
        request->stage_timing_trace.decode_admission_attempts = attempts;
        request->stage_timing_trace.decode_admission_active_requests =
            last_best_active;
        request->stage_timing_trace.decode_admission_waiting_requests =
            last_best_waiting;
        request->stage_timing_trace.decode_admission_estimated_tpot_ms =
            last_best_estimated_tpot;
        request->stage_timing_trace.decode_admission_pressure_tier =
            decode_admission_pressure_tier_name(last_best_pressure_tier);
        request->stage_timing_trace
            .decode_admission_preferred_decode_instance =
            last_best_decode_instance;
      }
      return true;
    }

    ++attempts;
    const uint64_t waited_ms = current_time_ms() - start_ms;
    if (waited_ms >= timeout_ms) {
      LOG(ERROR) << "Decode pressure admission timeout after " << waited_ms
                 << " ms, attempts=" << attempts
                 << ", request_type=" << (request_is_long ? "long" : "short")
                 << ", best_decode_instance=" << best_candidate.instance_name
                 << ", best_active_requests=" << last_best_active
                 << ", best_waiting_requests=" << last_best_waiting
                 << ", best_estimated_tpot_ms=" << last_best_estimated_tpot;
      return false;
    }

    if (attempts % FLAGS_decode_pressure_admission_log_every_n == 0) {
      LOG(INFO) << "Decode pressure admission waiting, waited_ms=" << waited_ms
                << ", attempts=" << attempts
                << ", request_type=" << (request_is_long ? "long" : "short")
                << ", best_decode_instance=" << best_candidate.instance_name
                << ", best_active_requests=" << last_best_active
                << ", best_waiting_requests=" << last_best_waiting
                << ", best_pressure_tier="
                << decode_admission_pressure_tier_name(last_best_pressure_tier)
                << ", best_estimated_tpot_ms=" << last_best_estimated_tpot;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(
        FLAGS_decode_pressure_admission_check_interval_ms));
  }
}

// TODO: refactor later, currently return all prefill instances
std::vector<std::string> InstanceMgr::get_static_prefill_list(
    const std::string& instance_name) {
  std::vector<std::string> prefill_list;
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  for (auto& inst : instances_) {
    if ((inst.second.type == InstanceType::PREFILL ||
         inst.second.type == InstanceType::DEFAULT) &&
        is_instance_schedulable(inst.second)) {
      prefill_list.emplace_back(inst.second.name);
    }
  }

  return prefill_list;
}

void InstanceMgr::get_load_metrics(LoadBalanceInfos* infos) {
  std::shared_lock<std::shared_mutex> inst_lock(cluster_mutex_);
  std::shared_lock<std::shared_mutex> metric_lock(metrics_mutex_);

  for (auto name : infos->overlap_scores.instances) {
    auto it = load_metrics_.find(name);
    if (it == load_metrics_.end()) {
      continue;
    }
    auto instance_it = instances_.find(name);
    if (instance_it == instances_.end() ||
        !is_instance_schedulable(instance_it->second)) {
      continue;
    }

    if (instance_it->second.type == InstanceType::DECODE) {
      infos->decode_load_metrics.insert(std::make_pair(name, it->second));
      infos->decode_max_waiting_requests_num =
          std::max(infos->decode_max_waiting_requests_num,
                   it->second.waiting_requests_num);
    } else {
      infos->prefill_load_metrics.insert(std::make_pair(name, it->second));
      infos->prefill_max_waiting_requests_num =
          std::max(infos->prefill_max_waiting_requests_num,
                   it->second.waiting_requests_num);
    }
  }

  std::string least_loaded_prefill_instance;
  float least_loaded_prefill_gpu_cache_usage_perc = 1;
  std::string least_loaded_decode_instance;
  float least_loaded_decode_gpu_cache_usage_perc = 1;

  if (infos->prefill_load_metrics.size() == 0 ||
      infos->decode_load_metrics.size() == 0) {
    for (const auto& metric : load_metrics_) {
      auto instance_it = instances_.find(metric.first);
      if (instance_it == instances_.end() ||
          !is_instance_schedulable(instance_it->second)) {
        continue;
      }
      if (instance_it->second.type != InstanceType::DECODE) {
        if (metric.second.gpu_cache_usage_perc <
            least_loaded_prefill_gpu_cache_usage_perc) {
          least_loaded_prefill_gpu_cache_usage_perc =
              metric.second.gpu_cache_usage_perc;
          least_loaded_prefill_instance = metric.first;
        }
      } else {
        if (metric.second.gpu_cache_usage_perc <
            least_loaded_decode_gpu_cache_usage_perc) {
          least_loaded_decode_gpu_cache_usage_perc =
              metric.second.gpu_cache_usage_perc;
          least_loaded_decode_instance = metric.first;
        }
      }
    }
  }

  if (infos->prefill_load_metrics.size() == 0 &&
      !least_loaded_prefill_instance.empty()) {
    infos->prefill_load_metrics.insert(
        std::make_pair(least_loaded_prefill_instance,
                       load_metrics_[least_loaded_prefill_instance]));
  }

  if (infos->decode_load_metrics.size() == 0 &&
      !least_loaded_decode_instance.empty()) {
    infos->decode_load_metrics.insert(
        std::make_pair(least_loaded_decode_instance,
                       load_metrics_[least_loaded_decode_instance]));
  }
}

void InstanceMgr::record_load_metrics_update(
    const std::string& instance_name,
    const proto::LoadMetrics& load_metrics) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

  updated_metrics_.insert_or_assign(
      instance_name,
      LoadMetrics(load_metrics.waiting_requests_num(),
                  load_metrics.gpu_cache_usage_perc()));
}

bool InstanceMgr::upload_load_metrics() {
  std::unordered_map<std::string, LoadMetrics> upload_snapshot;
  std::unordered_set<std::string> remove_snapshot;
  {
    std::unique_lock<std::shared_mutex> lk(metrics_mutex_);
    for (auto& iter : updated_metrics_) {
      load_metrics_.insert_or_assign(iter.first, iter.second);
    }
    for (auto& iter : removed_instance_) {
      load_metrics_.erase(iter);
    }
    upload_snapshot = updated_metrics_;
    remove_snapshot = removed_instance_;
    updated_metrics_.clear();
    removed_instance_.clear();
  }
  bool status = etcd_client_->set(ETCD_LOADMETRICS_PREFIX, upload_snapshot);
  status = status && etcd_client_->rm(ETCD_LOADMETRICS_PREFIX, remove_snapshot);
  return status;
}

void InstanceMgr::set_as_master() {
  is_master_service_ = true;
  etcd_client_->remove_watch(ETCD_LOADMETRICS_PREFIX);
}

std::shared_ptr<brpc::Channel> InstanceMgr::get_channel(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);
  auto iter = cached_channels_.find(instance_name);
  if (iter == cached_channels_.end()) {
    return nullptr;
  }
  return iter->second;
}

bool InstanceMgr::bind_request_instance_incarnations(
    const std::shared_ptr<Request>& request) {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);

  // Bind the selected routing to a concrete incarnation before dispatch.
  request->prefill_incarnation_id.clear();
  request->decode_incarnation_id.clear();

  if (!request->routing.prefill_name.empty()) {
    auto prefill_it = instances_.find(request->routing.prefill_name);
    if (prefill_it == instances_.end()) {
      LOG(ERROR) << "Prefill instance is not registered when binding request: "
                 << request->routing.prefill_name;
      return false;
    }
    if (!is_instance_schedulable(prefill_it->second)) {
      LOG(ERROR) << "Prefill instance is not schedulable when binding request: "
                 << request->routing.prefill_name << ", state: "
                 << runtime_state_name(prefill_it->second.runtime_state);
      return false;
    }
    request->prefill_incarnation_id = prefill_it->second.incarnation_id;
  }

  if (!request->routing.decode_name.empty()) {
    auto decode_it = instances_.find(request->routing.decode_name);
    if (decode_it == instances_.end()) {
      LOG(ERROR) << "Decode instance is not registered when binding request: "
                 << request->routing.decode_name;
      return false;
    }
    if (!is_instance_schedulable(decode_it->second)) {
      LOG(ERROR) << "Decode instance is not schedulable when binding request: "
                 << request->routing.decode_name << ", state: "
                 << runtime_state_name(decode_it->second.runtime_state);
      return false;
    }
    request->decode_incarnation_id = decode_it->second.incarnation_id;
  }

  return true;
}

bool InstanceMgr::record_instance_heartbeat(const std::string& instance_name,
                                            const std::string& incarnation_id) {
  std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
  auto it = instances_.find(instance_name);
  if (it == instances_.end()) {
    LOG(WARNING) << "Ignore heartbeat from unknown instance: " << instance_name;
    return false;
  }

  if (it->second.incarnation_id != incarnation_id) {
    LOG(WARNING) << "Ignore stale heartbeat from instance: " << instance_name
                 << ", current incarnation_id: " << it->second.incarnation_id
                 << ", heartbeat incarnation_id: " << incarnation_id;
    return false;
  }

  it->second.latest_timestamp = current_time_ms();
  if (it->second.runtime_state == InstanceRuntimeState::REGISTERING) {
    return true;
  }
  if (it->second.runtime_state == InstanceRuntimeState::SUSPECT) {
    // A recovered suspect instance first goes back to LEASE_LOST and must
    // keep heartbeating before becoming fully active again via registration.
    clear_suspect_instance(instance_name, incarnation_id);
    it->second.runtime_state = InstanceRuntimeState::LEASE_LOST;
    LOG(WARNING) << "Heartbeat recovered for suspect instance, move to "
                 << "lease lost state: " << instance_name
                 << ", incarnation_id: " << incarnation_id;
  }
  return true;
}

bool InstanceMgr::recover_instance_from_etcd(const std::string& instance_name,
                                             const std::string& incarnation_id) {
  for (const auto& it : ETCD_KEYS_PREFIX_MAP) {
    const auto& key_prefix = it.second;
    InstanceMetaInfo metainfo;
    if (!etcd_client_->get(key_prefix + instance_name, &metainfo)) {
      continue;
    }
    if (!incarnation_id.empty() && metainfo.incarnation_id != incarnation_id) {
      LOG(WARNING) << "Warmtrace recover_instance_from_etcd incarnation mismatch: "
                   << instance_name << ", etcd incarnation_id: "
                   << metainfo.incarnation_id << ", heartbeat incarnation_id: "
                   << incarnation_id;
      return false;
    }

    LOG(INFO) << "Warmtrace recover_instance_from_etcd hit key_prefix="
              << key_prefix << ", instance_name=" << instance_name
              << ", type=" << static_cast<int>(metainfo.type)
              << ", rpc_address=" << metainfo.rpc_address
              << ", incarnation_id=" << metainfo.incarnation_id;

    std::string old_incarnation_id;
    {
      std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
      auto existing_it = instances_.find(instance_name);
      if (existing_it != instances_.end()) {
        if (existing_it->second.incarnation_id == metainfo.incarnation_id) {
          refresh_instance_registration(instance_name, metainfo);
          clear_suspect_instance(instance_name, metainfo.incarnation_id);
          return true;
        }
        old_incarnation_id = existing_it->second.incarnation_id;
      }
    }

    if (!old_incarnation_id.empty()) {
      deregister_instance(instance_name, old_incarnation_id);
    }
    return register_instance(instance_name, metainfo);
  }

  LOG(WARNING) << "Warmtrace recover_instance_from_etcd failed, instance not found in etcd: "
               << instance_name << ", incarnation_id=" << incarnation_id;
  return false;
}

bool InstanceMgr::init_brpc_channel(
    const std::string& instance_name,
    std::shared_ptr<brpc::Channel>* out_channel) {
  auto channel = std::make_shared<brpc::Channel>();
  brpc::ChannelOptions options;
  // Add to params
  // options.protocol = "http";
  options.timeout_ms = options_.timeout_ms(); /*milliseconds*/
  options.max_retry = 3;
  options.connect_timeout_ms = options_.connect_timeout_ms();
  std::string load_balancer = "";
  if (channel->Init(instance_name.c_str(), load_balancer.c_str(), &options) !=
      0) {
    LOG(ERROR) << "Fail to initialize channel for " << instance_name;
    return false;
  }
  *out_channel = std::move(channel);
  return true;
}

bool InstanceMgr::probe_instance_health(const std::string& instance_name) {
  const int attempts = std::max(1, options_.instance_delete_probe_attempts());
  const int timeout_ms =
      std::max(1, options_.instance_delete_probe_timeout_ms());
  const std::string url = "http://" + instance_name + kHealthPath;

  for (int attempt = 1; attempt <= attempts; ++attempt) {
    brpc::Channel channel;
    brpc::ChannelOptions options;
    options.protocol = "http";
    options.timeout_ms = timeout_ms;
    options.connect_timeout_ms = timeout_ms;
    options.max_retry = 0;
    if (channel.Init(url.c_str(), "", &options) != 0) {
      LOG(WARNING) << "Failed to initialize health probe channel, instance: "
                   << instance_name << ", attempt: " << attempt << "/"
                   << attempts;
    } else {
      brpc::Controller cntl;
      cntl.http_request().uri() = url;
      cntl.http_request().set_method(brpc::HTTP_METHOD_GET);
      channel.CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);
      if (!cntl.Failed() && cntl.http_response().status_code() == 200) {
        return true;
      }
      LOG(WARNING) << "Health probe failed, instance: " << instance_name
                   << ", attempt: " << attempt << "/" << attempts << ", error: "
                   << (cntl.Failed() ? cntl.ErrorText()
                                     : std::to_string(
                                           cntl.http_response().status_code()));
    }

    if (attempt < attempts) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kDeleteProbeRetryBackoffMs));
    }
  }

  return false;
}

void InstanceMgr::update_instance_metainfo(const etcd::Response& response,
                                           const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }

  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    for (const auto& event : response.events()) {
      const std::string instance_name = get_event_key_suffix(event, prefix_len);
      if (instance_name.empty()) {
        continue;
      }

      if (event.event_type() == etcd::Event::EventType::PUT) {
        InstanceMetaInfo metainfo;
        LOG(INFO) << "Warmtrace service etcd PUT raw instance_name=" << instance_name << ", prefix_len=" << prefix_len;
        auto json_str = get_event_value(event);
        LOG(INFO) << "Warmtrace service etcd PUT parse begin instance_name=" << instance_name << ", json=" << json_str;
        if (!metainfo.parse_from_json(json_str)) {
          LOG(ERROR) << "Parse instance json failed: " << json_str;
          continue;
        }

        std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
        auto existing_it = instances_.find(instance_name);
        if (existing_it == instances_.end()) {
          lock.unlock();
          if (!register_instance(instance_name, metainfo)) {
            LOG(ERROR) << "Fail to register instance: " << instance_name;
          }
          continue;
        }

        if (existing_it->second.incarnation_id == metainfo.incarnation_id) {
          const auto previous_state = existing_it->second.runtime_state;
          refresh_instance_registration(instance_name, metainfo);
          clear_suspect_instance(instance_name, metainfo.incarnation_id);
          if (previous_state != InstanceRuntimeState::ACTIVE) {
            LOG(INFO) << "Instance registration restored, back to active: "
                      << instance_name
                      << ", incarnation_id: " << metainfo.incarnation_id
                      << ", previous_state: "
                      << runtime_state_name(previous_state);
          }
          continue;
        }

        const std::string old_incarnation_id =
            existing_it->second.incarnation_id;
        LOG(WARNING) << "Detected instance replacement, instance_name: "
                     << instance_name
                     << ", old incarnation_id: " << old_incarnation_id
                     << ", new incarnation_id: " << metainfo.incarnation_id;
        lock.unlock();
        deregister_instance(instance_name, old_incarnation_id);
        if (!register_instance(instance_name, metainfo)) {
          LOG(ERROR) << "Fail to register replacement instance: "
                     << instance_name;
        }
        continue;
      }

      if (event.event_type() != etcd::Event::EventType::DELETE_) {
        continue;
      }

      InstanceMetaInfo deleted_info;
      std::string deleted_incarnation_id;
      const auto deleted_value = get_event_value(event);
      if (!deleted_value.empty() &&
          deleted_info.parse_from_json(deleted_value)) {
        deleted_incarnation_id = deleted_info.incarnation_id;
      }
      std::string tracked_incarnation_id;
      {
        std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
        auto existing_it = instances_.find(instance_name);
        if (existing_it == instances_.end()) {
          continue;
        }

        if (!deleted_incarnation_id.empty() &&
            existing_it->second.incarnation_id != deleted_incarnation_id) {
          LOG(INFO) << "Ignore stale delete for replaced instance: "
                    << instance_name
                    << ", deleted incarnation_id: " << deleted_incarnation_id
                    << ", current incarnation_id: "
                    << existing_it->second.incarnation_id;
          continue;
        }
        tracked_incarnation_id = existing_it->second.incarnation_id;
      }

      // Keep delete handling event-driven: use this event, one health probe,
      // and later heartbeats or PUT events to drive recovery.
      const bool probe_success = probe_instance_health(instance_name);

      std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
      auto existing_it = instances_.find(instance_name);
      if (existing_it == instances_.end() ||
          existing_it->second.incarnation_id != tracked_incarnation_id) {
        continue;
      }

      if (probe_success) {
        // Keep the instance in service temporarily and wait for heartbeats.
        clear_suspect_instance(instance_name, tracked_incarnation_id);
        existing_it->second.runtime_state = InstanceRuntimeState::LEASE_LOST;
        existing_it->second.latest_timestamp = current_time_ms();
        LOG(WARNING) << "Instance lease deleted, probe succeeded, enter "
                     << "lease lost state: " << instance_name
                     << ", incarnation_id: " << tracked_incarnation_id;
        continue;
      }

      mark_instance_suspect(instance_name, tracked_incarnation_id);
      LOG(WARNING) << "Instance lease deleted, probe failed, enter suspect "
                   << "state: " << instance_name
                   << ", incarnation_id: " << tracked_incarnation_id;
    }
  });
}

void InstanceMgr::update_load_metrics(const etcd::Response& response,
                                      const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }
  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    std::unordered_map<std::string, LoadMetrics> put_map;
    std::vector<std::string> delete_list;

    for (const auto& event : response.events()) {
      std::string instance_name = event.kv().key().substr(prefix_len);

      if (event.event_type() == etcd::Event::EventType::PUT) {
        LoadMetrics load_metrics;
        auto json_str = event.kv().as_string();
        if (!load_metrics.parse_from_json(json_str)) {
          LOG(ERROR) << "pase json:" << json_str << " error!";
          continue;
        }

        put_map.insert(std::make_pair(instance_name, std::move(load_metrics)));

      } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
        delete_list.push_back(instance_name);
      }
    }

    {
      std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
      for (auto& iter : put_map) {
        load_metrics_.insert_or_assign(iter.first, std::move(iter.second));
      }

      for (auto& iter : delete_list) {
        load_metrics_.erase(iter);
      }
    }
  });
}

void InstanceMgr::update_latency_metrics(
    const std::string& instance_name,
    const proto::LatencyMetrics& latency_metrics) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

  latency_metrics_.insert_or_assign(
      instance_name,
      LatencyMetrics(latency_metrics.recent_max_ttft(),
                     latency_metrics.recent_max_tbt()));
}

void InstanceMgr::tick_pool_elasticity(uint64_t now_ms) {
  // Two ways to declare the elastic pool:
  //   (1) selector list (legacy): pool_elastic_idle_default_instances names
  //       which instances start IDLE. Other instances stay ACTIVE forever.
  //   (2) auto mode (P4-NEW): pool_elastic_auto_mode=true. Every PREFILL/MIX
  //       instance is treated as elastic, and the controller computes a
  //       desired ACTIVE count from min_active/max_active/target_per_inst.
  // If both are set, auto mode takes precedence; the legacy selector list is
  // ignored. If neither is set, the controller does nothing.
  const bool auto_mode = FLAGS_pool_elastic_auto_mode;
  const bool selector_mode = !FLAGS_pool_elastic_idle_default_instances.empty();
  if (!auto_mode && !selector_mode) return;

  // Selector list parsed once (only used when auto_mode == false).
  static const std::vector<std::string> elastic_selectors =
      parse_instance_selectors(FLAGS_pool_elastic_idle_default_instances);
  if (!auto_mode && elastic_selectors.empty()) return;

  const double long_ratio = PoolElasticStats::instance().compute_long_ratio(
      FLAGS_pool_elastic_window_s, now_ms);
  const uint64_t total_in_window = PoolElasticStats::instance().total_in_window(
      FLAGS_pool_elastic_window_s, now_ms);
  if (total_in_window == 0) return;

  const bool use_pressure = FLAGS_pool_elastic_use_pressure_signal;
  std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
  const double pool_pressure =
      use_pressure ? compute_pool_pressure_locked() : 0.0;

  // Predicate: is this instance considered part of the elastic pool right now?
  auto is_elastic_inst = [&](const std::string& name,
                             const InstanceMetaInfo& info) -> bool {
    if (info.type != InstanceType::PREFILL && info.type != InstanceType::MIX)
      return false;
    if (auto_mode) {
      // In auto mode every prefill/mix instance is fair game.
      return true;
    }
    for (const auto& sel : elastic_selectors) {
      if (instance_matches_selector(name, sel)) return true;
    }
    return false;
  };

  // Count current ACTIVE / IDLE elastic instances.
  int32_t active_count = 0;
  int32_t idle_count = 0;
  int32_t total_elastic = 0;
  for (auto& [inst_name, info] : instances_) {
    if (!is_elastic_inst(inst_name, info)) continue;
    ++total_elastic;
    if (info.runtime_state == InstanceRuntimeState::ACTIVE) ++active_count;
    else if (info.runtime_state == InstanceRuntimeState::IDLE) ++idle_count;
  }
  if (total_elastic == 0) return;

  // Desired ACTIVE count under auto mode. clamp(ceil(pressure / target_per_inst),
  // min_active, max_active). max_active=0 means unbounded -> use total_elastic.
  int32_t desired_active = active_count;  // legacy path keeps current count
  if (auto_mode) {
    const double target_per_inst = std::max(
        0.001, FLAGS_pool_elastic_target_pressure_per_instance);
    const int32_t pressure_demand =
        use_pressure
            ? static_cast<int32_t>(
                  std::ceil(pool_pressure / target_per_inst))
            : 1;  // no pressure signal -> fall back to min_active
    const int32_t min_active =
        std::max(0, FLAGS_pool_elastic_min_active);
    const int32_t max_active_cap =
        FLAGS_pool_elastic_max_active > 0
            ? std::min(FLAGS_pool_elastic_max_active, total_elastic)
            : total_elastic;
    desired_active = std::clamp(pressure_demand, min_active, max_active_cap);
  }

  // Step-activation cool-down at the pool level. Limits how often the
  // controller can change ACTIVE membership (in either direction).
  const uint64_t cool_down_ms = static_cast<uint64_t>(std::max<int32_t>(
      0, FLAGS_pool_elastic_activation_cool_down_s)) * 1000ULL;
  const bool pool_cool_down_active =
      last_pool_activation_ts_ms_ != 0 &&
      now_ms - last_pool_activation_ts_ms_ < cool_down_ms;

  // Decide on at most one promotion or one demotion this tick.
  std::string activation_target;
  uint64_t activation_target_ts = std::numeric_limits<uint64_t>::max();
  std::string deactivation_target;
  uint64_t deactivation_target_ts = 0;
  bool need_activate = false;
  bool need_deactivate = false;

  if (auto_mode) {
    if (active_count < desired_active && idle_count > 0 &&
        !pool_cool_down_active) {
      need_activate = true;
    } else if (active_count > desired_active && active_count > 0 &&
               !pool_cool_down_active) {
      need_deactivate = true;
    }
  } else if (use_pressure) {
    // Legacy selector path: activate when pressure or long_ratio crosses
    // thresholds; deactivate via per-instance ACTIVE-state branch below.
    if (!pool_cool_down_active) {
      const bool pressure_high =
          pool_pressure >= FLAGS_pool_elastic_activate_pressure_threshold;
      const bool long_ratio_high_with_load =
          long_ratio >= FLAGS_pool_elastic_activate_long_ratio &&
          pool_pressure >= FLAGS_pool_elastic_activate_min_pressure;
      if (pressure_high || long_ratio_high_with_load) need_activate = true;
    }
  }

  // P5: lane-aware preference. When long_ratio >= lane-aware threshold the
  // controller prefers a long-lane (kv_split_size>1) candidate; otherwise it
  // prefers a short-lane (kv_split_size<=1) candidate. The preference is a
  // soft tie-break: if no candidate of the preferred lane is eligible, we
  // fall back to any eligible IDLE instance (so an active scale-up never
  // stalls because the ideal lane is exhausted).
  const bool lane_aware =
      FLAGS_pool_elastic_lane_aware &&
      (auto_mode || use_pressure);  // only meaningful in pressure paths
  const bool prefer_long_lane =
      lane_aware && long_ratio >=
                        FLAGS_pool_elastic_lane_aware_threshold_long_ratio;

  auto matches_preferred_lane = [&](const InstanceMetaInfo& info) -> bool {
    if (!lane_aware) return true;
    if (prefer_long_lane) return info.kv_split_size > 1;
    return info.kv_split_size <= 1;
  };

  // Pick activation_target = oldest IDLE instance (by last_activated_ts_ms),
  // preferring the lane that matches current traffic shape if lane_aware is on.
  if (need_activate) {
    // Pass 1: only consider IDLE instances of the preferred lane.
    for (auto& [inst_name, info] : instances_) {
      if (!is_elastic_inst(inst_name, info)) continue;
      if (info.runtime_state != InstanceRuntimeState::IDLE) continue;
      if (info.last_activated_ts_ms != 0 &&
          now_ms - info.last_activated_ts_ms < cool_down_ms) {
        continue;
      }
      if (!matches_preferred_lane(info)) continue;
      if (info.last_activated_ts_ms < activation_target_ts) {
        activation_target_ts = info.last_activated_ts_ms;
        activation_target = inst_name;
      }
    }
    // Pass 2: fallback to any eligible IDLE instance.
    if (activation_target.empty()) {
      for (auto& [inst_name, info] : instances_) {
        if (!is_elastic_inst(inst_name, info)) continue;
        if (info.runtime_state != InstanceRuntimeState::IDLE) continue;
        if (info.last_activated_ts_ms != 0 &&
            now_ms - info.last_activated_ts_ms < cool_down_ms) {
          continue;
        }
        if (info.last_activated_ts_ms < activation_target_ts) {
          activation_target_ts = info.last_activated_ts_ms;
          activation_target = inst_name;
        }
      }
    }
    if (activation_target.empty()) need_activate = false;
  }

  // Pick deactivation_target = ACTIVE instance least recently activated
  // (longest tenure as ACTIVE). Newly-promoted instances are protected.
  if (need_deactivate) {
    // P5: prefer to demote the lane that does NOT match current traffic
    // shape. Long burst -> demote cp_size<=1 first; short burst -> demote
    // cp_size>1 first. Newest activation wins as tiebreaker so we don't
    // demote an instance the controller just promoted.
    auto matches_demote_preferred_lane =
        [&](const InstanceMetaInfo& info) -> bool {
          if (!lane_aware) return true;
          if (prefer_long_lane) return info.kv_split_size <= 1;
          return info.kv_split_size > 1;
        };

    // Pass 1: only demote-preferred-lane candidates.
    for (auto& [inst_name, info] : instances_) {
      if (!is_elastic_inst(inst_name, info)) continue;
      if (info.runtime_state != InstanceRuntimeState::ACTIVE) continue;
      if (info.last_activated_ts_ms == 0) continue;  // never tracked, skip
      if (!matches_demote_preferred_lane(info)) continue;
      if (info.last_activated_ts_ms > deactivation_target_ts) {
        deactivation_target_ts = info.last_activated_ts_ms;
        deactivation_target = inst_name;
      }
    }
    // Pass 2: fallback to any ACTIVE instance with last_activated_ts_ms set.
    if (deactivation_target.empty()) {
      for (auto& [inst_name, info] : instances_) {
        if (!is_elastic_inst(inst_name, info)) continue;
        if (info.runtime_state != InstanceRuntimeState::ACTIVE) continue;
        if (info.last_activated_ts_ms == 0) continue;
        if (info.last_activated_ts_ms > deactivation_target_ts) {
          deactivation_target_ts = info.last_activated_ts_ms;
          deactivation_target = inst_name;
        }
      }
    }
    // If we still couldn't find a candidate (all ACTIVE have ts_ms==0,
    // e.g. on first tick when "ACTIVE" instances were registered without
    // ever being IDLE), pick the lexicographically-largest name as a
    // deterministic fallback.
    if (deactivation_target.empty()) {
      for (auto& [inst_name, info] : instances_) {
        if (!is_elastic_inst(inst_name, info)) continue;
        if (info.runtime_state != InstanceRuntimeState::ACTIVE) continue;
        if (inst_name > deactivation_target) deactivation_target = inst_name;
      }
    }
    if (deactivation_target.empty()) need_deactivate = false;
  }

  std::vector<PoolElasticTransition> transitions;
  for (auto& [inst_name, info] : instances_) {
    if (!is_elastic_inst(inst_name, info)) continue;

    const uint64_t combined_load = get_prefill_combined_load(
        request_metrics_, load_metrics_, inst_name);

    const InstanceRuntimeState prev_state = info.runtime_state;

    switch (info.runtime_state) {
      case InstanceRuntimeState::IDLE: {
        bool should_activate = false;
        if (auto_mode) {
          should_activate = need_activate && (inst_name == activation_target);
        } else if (use_pressure) {
          should_activate = need_activate && (inst_name == activation_target);
        } else {
          should_activate =
              long_ratio >= FLAGS_pool_elastic_activate_long_ratio &&
              static_cast<int32_t>(combined_load) >=
                  FLAGS_pool_elastic_activate_min_load;
        }
        if (should_activate) {
          info.runtime_state = InstanceRuntimeState::ACTIVE;
          info.draining_since_ms = 0;
          info.deactivate_condition_since_ms = 0;
          info.last_activated_ts_ms = now_ms;
          last_pool_activation_ts_ms_ = now_ms;
          LOG(INFO) << "Pool elastic state change: " << inst_name
                    << " IDLE->ACTIVE, long_ratio=" << long_ratio
                    << " pool_pressure=" << pool_pressure
                    << " desired_active=" << desired_active
                    << " active_count=" << active_count
                    << " kv_split_size=" << info.kv_split_size
                    << " lane_aware=" << (lane_aware ? 1 : 0)
                    << " prefer_long_lane=" << (prefer_long_lane ? 1 : 0)
                    << " mode=" << (auto_mode ? "auto"
                                              : (use_pressure ? "pressure"
                                                              : "long_ratio"));
        }
        break;
      }
      case InstanceRuntimeState::ACTIVE: {
        bool should_deactivate = false;
        if (auto_mode) {
          // In auto mode, deactivation is fully driven by desired_active.
          // Per-instance ACTIVE -> DRAINING happens immediately when picked
          // (no persist timer needed; the desired_active itself smoothes
          // over short pressure dips by virtue of clamp + cool-down).
          should_deactivate =
              need_deactivate && (inst_name == deactivation_target);
        } else {
          // Legacy path: persist-timer + threshold gate.
          bool below_deactivate = false;
          if (use_pressure) {
            below_deactivate =
                pool_pressure <
                FLAGS_pool_elastic_deactivate_pressure_threshold;
          } else {
            below_deactivate =
                long_ratio < FLAGS_pool_elastic_deactivate_long_ratio;
          }
          if (below_deactivate) {
            if (info.deactivate_condition_since_ms == 0) {
              info.deactivate_condition_since_ms = now_ms;
            }
            const uint64_t persist_ms =
                now_ms - info.deactivate_condition_since_ms;
            if (persist_ms >=
                static_cast<uint64_t>(
                    FLAGS_pool_elastic_deactivate_persist_s) * 1000 &&
                combined_load == 0) {
              should_deactivate = true;
            }
          } else {
            info.deactivate_condition_since_ms = 0;
          }
        }
        if (should_deactivate) {
          info.runtime_state = InstanceRuntimeState::DRAINING;
          info.draining_since_ms = now_ms;
          info.deactivate_condition_since_ms = 0;
          last_pool_activation_ts_ms_ = now_ms;
          LOG(INFO) << "Pool elastic state change: " << inst_name
                    << " ACTIVE->DRAINING, long_ratio=" << long_ratio
                    << " pool_pressure=" << pool_pressure
                    << " desired_active=" << desired_active
                    << " active_count=" << active_count
                    << " combined_load=" << combined_load
                    << " kv_split_size=" << info.kv_split_size
                    << " lane_aware=" << (lane_aware ? 1 : 0)
                    << " prefer_long_lane=" << (prefer_long_lane ? 1 : 0)
                    << " mode=" << (auto_mode ? "auto"
                                              : (use_pressure ? "pressure"
                                                              : "long_ratio"));
        }
        break;
      }
      case InstanceRuntimeState::DRAINING: {
        if (combined_load == 0 ||
            (now_ms - info.draining_since_ms >
             static_cast<uint64_t>(FLAGS_pool_elastic_drain_timeout_ms))) {
          info.runtime_state = InstanceRuntimeState::IDLE;
          info.draining_since_ms = 0;
          info.deactivate_condition_since_ms = 0;
          info.last_activated_ts_ms = now_ms;
          last_pool_activation_ts_ms_ = now_ms;
          LOG(INFO) << "Pool elastic state change: " << inst_name
                    << " DRAINING->IDLE, combined_load=" << combined_load
                    << " pool_pressure=" << pool_pressure;
        }
        break;
      }
      default:
        break;
    }

    if (info.runtime_state != prev_state) {
      transitions.push_back({inst_name,
                             prev_state,
                             info.runtime_state,
                             now_ms,
                             long_ratio,
                             combined_load});
    }
  }
  if (!transitions.empty()) {
    publish_pool_elastic_transitions(std::move(transitions));
  }
}

void InstanceMgr::reconcile_instance_states() {
  const auto suspect_interval_ms =
      std::max<int64_t>(1, options_.detect_disconnected_instance_interval()) *
      1000;
  const auto heartbeat_timeout_ms =
      std::max<int64_t>(1, options_.lease_lost_heartbeat_timeout_ms());
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    tick_pool_elasticity(current_time_ms());

    std::vector<std::pair<std::string, std::string>> to_deregister;

    {
      std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
      if (exited_) {
        return;
      }

      const uint64_t now_ms = current_time_ms();
      for (auto& [instance_name, info] : instances_) {
        // LEASE_LOST is a grace period after etcd delete but before hard
        // eviction.
        if (info.runtime_state != InstanceRuntimeState::LEASE_LOST) {
          continue;
        }
        if (now_ms - info.latest_timestamp < heartbeat_timeout_ms) {
          continue;
        }
        mark_instance_suspect(instance_name, info.incarnation_id);
        LOG(WARNING)
            << "Lease lost instance heartbeat timed out, enter suspect "
            << "state: " << instance_name
            << ", incarnation_id: " << info.incarnation_id;
      }

      for (auto it = suspect_instances_.begin();
           it != suspect_instances_.end();) {
        const std::string instance_name = it->first;
        const std::string incarnation_id = it->second.incarnation_id;
        const uint64_t enter_ts_ms = it->second.enter_ts_ms;
        ++it;

        if (now_ms - enter_ts_ms < suspect_interval_ms) {
          continue;
        }

        auto inst_it = instances_.find(instance_name);
        if (inst_it == instances_.end() ||
            inst_it->second.incarnation_id != incarnation_id) {
          suspect_instances_.erase(instance_name);
          continue;
        }

        LOG(WARNING) << "Suspect window expired, deregister instance: "
                     << instance_name << ", incarnation_id: " << incarnation_id;
        to_deregister.emplace_back(instance_name, incarnation_id);
      }
    }

    for (const auto& p : to_deregister) {
      deregister_instance(p.first, p.second);
    }
  }
}

void InstanceMgr::refresh_instance_registration(const std::string& name,
                                                const InstanceMetaInfo& info) {
  auto it = instances_.find(name);
  if (it == instances_.end()) {
    return;
  }

  // Preserve local scheduling/index state across etcd refreshes.
  const auto instance_index = it->second.instance_index;
  const auto current_type = it->second.current_type;
  // Pool-elastic correctness: runtime_state, draining_since_ms and the
  // per-instance deactivate condition timer are runtime-only signals not
  // reflected in etcd. They MUST survive an etcd refresh so the pool elastic
  // controller can keep advancing IDLE/ACTIVE/DRAINING. The previous
  // implementation unconditionally reset runtime_state to ACTIVE on every
  // refresh, which silently undid every pool elastic decision.
  const auto runtime_state = it->second.runtime_state;
  const auto draining_since_ms = it->second.draining_since_ms;
  const auto deactivate_condition_since_ms =
      it->second.deactivate_condition_since_ms;

  it->second = info;
  it->second.instance_index = instance_index;
  it->second.current_type = current_type;
  it->second.latest_timestamp = current_time_ms();
  // For lifecycle transients (REGISTERING/SUSPECT/LEASE_LOST) we re-resolve
  // the desired initial state, so a recovered instance can re-enter IDLE if
  // it matches the elastic selector. For IDLE/ACTIVE/DRAINING we keep the
  // existing runtime decision so the elastic controller is the only writer.
  if (runtime_state == InstanceRuntimeState::REGISTERING ||
      runtime_state == InstanceRuntimeState::SUSPECT ||
      runtime_state == InstanceRuntimeState::LEASE_LOST) {
    it->second.runtime_state = resolve_initial_runtime_state(name, it->second);
    it->second.draining_since_ms = 0;
    it->second.deactivate_condition_since_ms = 0;
  } else {
    it->second.runtime_state = runtime_state;
    it->second.draining_since_ms = draining_since_ms;
    it->second.deactivate_condition_since_ms = deactivate_condition_since_ms;
  }
}

InstanceRuntimeState InstanceMgr::resolve_initial_runtime_state(
    const std::string& name,
    const InstanceMetaInfo& info) const {
  // Decode instances are not part of the elastic pool today.
  if (info.type == InstanceType::DECODE) {
    return InstanceRuntimeState::ACTIVE;
  }
  if (info.type != InstanceType::DEFAULT &&
      info.type != InstanceType::PREFILL &&
      info.type != InstanceType::MIX) {
    return InstanceRuntimeState::ACTIVE;
  }

  // P4-NEW: auto-classify a freshly-registered prefill/mix instance based on
  // the *current* ACTIVE count in the elastic pool. If we're below
  // min_active, the new arrival joins as ACTIVE so cold-start can be served
  // immediately. Otherwise it joins as IDLE and the controller will promote
  // it later if pool_pressure rises.
  if (FLAGS_pool_elastic_auto_mode) {
    const int32_t min_active =
        std::max(0, FLAGS_pool_elastic_min_active);
    int32_t active_elastic = 0;
    for (const auto& [n, i] : instances_) {
      if (n == name) continue;  // exclude the instance being registered
      if (i.type != InstanceType::PREFILL && i.type != InstanceType::MIX)
        continue;
      if (i.runtime_state == InstanceRuntimeState::ACTIVE) ++active_elastic;
    }
    return active_elastic < min_active ? InstanceRuntimeState::ACTIVE
                                        : InstanceRuntimeState::IDLE;
  }

  // Legacy selector path.
  if (FLAGS_pool_elastic_idle_default_instances.empty()) {
    return InstanceRuntimeState::ACTIVE;
  }
  const std::vector<std::string> idle_selectors =
      parse_instance_selectors(FLAGS_pool_elastic_idle_default_instances);
  for (const auto& selector : idle_selectors) {
    if (instance_matches_selector(name, selector)) {
      return InstanceRuntimeState::IDLE;
    }
  }
  return InstanceRuntimeState::ACTIVE;
}

void InstanceMgr::mark_instance_suspect(const std::string& name,
                                        const std::string& incarnation_id) {
  SuspectInstanceInfo info;
  info.incarnation_id = incarnation_id;
  info.enter_ts_ms = current_time_ms();
  suspect_instances_[name] = std::move(info);
  auto it = instances_.find(name);
  if (it != instances_.end() && it->second.incarnation_id == incarnation_id) {
    it->second.runtime_state = InstanceRuntimeState::SUSPECT;
  }
}

void InstanceMgr::clear_suspect_instance(const std::string& name,
                                         const std::string& incarnation_id) {
  auto it = suspect_instances_.find(name);
  if (it == suspect_instances_.end()) {
    return;
  }
  if (!incarnation_id.empty() && it->second.incarnation_id != incarnation_id) {
    return;
  }
  suspect_instances_.erase(it);
}

void InstanceMgr::publish_pool_elastic_transitions(
    std::vector<PoolElasticTransition> events) {
  if (events.empty()) return;
  std::lock_guard<std::mutex> lock(pool_elastic_events_mutex_);
  for (auto& e : events) {
    pool_elastic_events_.emplace_back(std::move(e));
    while (pool_elastic_events_.size() > kPoolElasticEventsCap) {
      pool_elastic_events_.pop_front();
    }
  }
}

std::vector<InstanceMgr::PoolElasticTransition>
InstanceMgr::take_recent_pool_elastic_transitions(size_t max_age_ms) const {
  std::vector<PoolElasticTransition> out;
  const uint64_t now_ms = current_time_ms();
  std::lock_guard<std::mutex> lock(pool_elastic_events_mutex_);
  out.reserve(pool_elastic_events_.size());
  for (const auto& e : pool_elastic_events_) {
    if (max_age_ms == 0 || now_ms - e.ts_ms <= max_age_ms) {
      out.push_back(e);
    }
  }
  return out;
}

std::vector<std::string>
InstanceMgr::snapshot_active_prefill_instances_locked() const {
  std::vector<std::string> out;
  out.reserve(prefill_index_.size());
  for (const auto& name : prefill_index_) {
    auto it = instances_.find(name);
    if (it == instances_.end()) continue;
    if (it->second.runtime_state != InstanceRuntimeState::ACTIVE) continue;
    out.push_back(name);
  }
  return out;
}

nlohmann::json InstanceMgr::build_pool_state_change_event_json(
    uint64_t window_ms) const {
  const auto recent = take_recent_pool_elastic_transitions(window_ms);
  nlohmann::json arr = nlohmann::json::array();
  for (const auto& e : recent) {
    arr.push_back({{"instance", e.instance_name},
                   {"from_state", runtime_state_name(e.from_state)},
                   {"to_state", runtime_state_name(e.to_state)},
                   {"ts_ms", e.ts_ms},
                   {"long_ratio", e.long_ratio},
                   {"combined_load", e.combined_load}});
  }
  return arr;
}

double InstanceMgr::compute_pool_pressure_locked() const {
  return compute_pool_pressure_lane_locked(/*lane_filter=*/0);
}

double InstanceMgr::compute_pool_pressure_lane_locked(int lane_filter) const {
  // Aggregator across the ACTIVE prefill / mix set:
  //   max over instances of:
  //     load_w * combined_load
  //   + wait_w * waiting_requests
  //   + pft_w  * (current_prefill_tokens / pft_baseline_tokens)
  //
  // current_prefill_tokens is used as a proxy for projected prefill time:
  // bigger backlog -> longer head-of-line wait. Using the raw
  // projected_prefill_time_ms would require running the time predictor here,
  // which is expensive and only meaningful per-request, not per-instance.
  //
  // P5: lane_filter selects which subset to aggregate over.
  //     0 = entire pool (legacy behavior)
  //     1 = long-lane subset (kv_split_size > 1)  <-- "CP big lane"
  //     2 = short-lane subset (kv_split_size <= 1) <-- "non-CP small P"
  // Other values fall back to the entire pool.
  const double load_w = FLAGS_pool_elastic_pressure_load_weight;
  const double wait_w = FLAGS_pool_elastic_pressure_wait_weight;
  const double pft_w  = FLAGS_pool_elastic_pressure_pft_weight;
  // P7: token budget for PFT normalization is now directly configurable via
  // pool_elastic_pressure_pft_baseline_tokens. The old pft_baseline_ms gflag
  // is kept for backward compatibility: if pft_baseline_tokens<=0 we fall
  // back to the legacy 8x conversion (1ms ~= 8 tokens) so existing operators
  // get the same behavior. Setting baseline_tokens explicitly bypasses the
  // implicit conversion entirely.
  double pft_baseline_tokens = static_cast<double>(
      std::max<int32_t>(0, FLAGS_pool_elastic_pressure_pft_baseline_tokens));
  if (pft_baseline_tokens <= 0.0) {
    const int32_t pft_baseline_ms = std::max<int32_t>(
        1, FLAGS_pool_elastic_pressure_pft_baseline_ms);
    pft_baseline_tokens = static_cast<double>(pft_baseline_ms) * 8.0;
  }

  double max_pressure = 0.0;
  for (const auto& name : prefill_index_) {
    auto it = instances_.find(name);
    if (it == instances_.end()) continue;
    if (it->second.runtime_state != InstanceRuntimeState::ACTIVE) continue;
    if (lane_filter == 1 && it->second.kv_split_size <= 1) continue;
    if (lane_filter == 2 && it->second.kv_split_size > 1) continue;

    const double load = static_cast<double>(
        get_prefill_combined_load(request_metrics_, load_metrics_, name));
    const double wait = static_cast<double>(
        get_waiting_requests_num(load_metrics_, name));
    const double tokens = static_cast<double>(
        get_prefill_token_num(request_metrics_, name));

    double pressure = 0.0;
    if (load_w > 0.0) pressure += load_w * load;
    if (wait_w > 0.0) pressure += wait_w * wait;
    if (pft_w > 0.0 && pft_baseline_tokens > 0.0) {
      pressure += pft_w * (tokens / pft_baseline_tokens);
    }
    if (pressure > max_pressure) max_pressure = pressure;
  }
  return max_pressure;
}

void InstanceMgr::update_request_metrics(std::shared_ptr<Request> request,
                                         RequestAction action) {
  // skip request metrics update if policy is not SLO_AWARE
  if (options_.load_balance_policy() != "SLO_AWARE") {
    return;
  }

  std::scoped_lock<std::shared_mutex, std::shared_mutex> lock(cluster_mutex_,
                                                              metrics_mutex_);

  auto prefill_it = request_metrics_.find(request->routing.prefill_name);
  if (prefill_it == request_metrics_.end()) {
    LOG(ERROR) << "Failed to find instance request metrics, instance name : "
               << request->routing.prefill_name;
    return;
  }

  auto decode_it = request_metrics_.find(request->routing.decode_name);
  if (decode_it == request_metrics_.end()) {
    LOG(ERROR) << "Failed to find instance request metrics, instance name : "
               << request->routing.decode_name;
    return;
  }

  int64_t num_prompt_tokens = request->token_ids.size();
  int64_t num_generated_tokens = request->num_generated_tokens;
  switch (action) {
    case RequestAction::SCHEDULE:
      // update the request metrics for prefill and decode instances when
      // request is scheduled
      prefill_it->second.prefill_request_num += 1;
      prefill_it->second.prefill_token_num += num_prompt_tokens;

      decode_it->second.decode_request_num += 1;
      decode_it->second.decode_token_num += num_prompt_tokens;
      break;
    case RequestAction::FINISH_PREFILL:
      // update the request metrics for prefill and decode instance when request
      // finishes the prefill phase
      prefill_it->second.prefill_request_num -= 1;
      prefill_it->second.prefill_token_num -= num_prompt_tokens;
      prefill_it->second.estimated_prefill_time -= request->estimated_ttft;

      decode_it->second.decode_token_num += 1;
      break;
    case RequestAction::GENERATE:
      // update the request metrics for decode instance when request generate a
      // token
      decode_it->second.decode_token_num += 1;
      break;
    case RequestAction::FINISH_DECODE:
      // update the request metrics for decode instance when request finishes
      // the decode phase
      decode_it->second.decode_request_num -= 1;
      decode_it->second.decode_token_num -=
          (num_prompt_tokens + num_generated_tokens);

      break;
    case RequestAction::CANCEL:
      // update the request metrics for prefill and decode instances when
      // request is cancelled
      prefill_it->second.prefill_request_num -= 1;
      prefill_it->second.prefill_token_num -= num_prompt_tokens;
      prefill_it->second.estimated_prefill_time -= request->estimated_ttft;

      decode_it->second.decode_request_num -= 1;
      decode_it->second.decode_token_num -=
          (num_prompt_tokens + num_generated_tokens);

      break;
    default:
      LOG(ERROR) << "Unknown RequestAction: " << static_cast<int32_t>(action);
      break;
  }

  if (decode_it->second.decode_request_num == 0) {
    flip_decode_to_prefill(request->routing.decode_name);
  }
}

bool InstanceMgr::select_instance_pair_on_slo(
    std::shared_ptr<Request> request) {
  std::scoped_lock<std::shared_mutex, std::shared_mutex> lock(cluster_mutex_,
                                                              metrics_mutex_);
  const bool has_unschedulable_instances = !suspect_instances_.empty();
  const bool request_is_long =
      request->token_ids.size() >=
      static_cast<size_t>(std::max<int32_t>(
          0, options_.long_request_threshold_tokens()));
  PoolElasticStats::instance().record_request(request_is_long,
                                              current_time_ms());
  const int32_t request_tokens = static_cast<int32_t>(request->token_ids.size());
  const int64_t decode_work_tokens =
      estimate_decode_work_tokens(request, request_tokens, request_is_long);
  const int32_t hard_long_threshold =
      std::max<int32_t>(0, options_.long_request_threshold_tokens());
  const int32_t v0_3_soft_long_lower_tokens =
      resolve_v0_3_soft_long_lower_tokens(hard_long_threshold);
  const int32_t v0_3_soft_long_upper_tokens =
      resolve_v0_3_soft_long_upper_tokens(hard_long_threshold,
                                          v0_3_soft_long_lower_tokens);
  const double v0_3_long_request_scale = compute_v0_3_long_request_scale(
      request_tokens, v0_3_soft_long_lower_tokens, v0_3_soft_long_upper_tokens);
  const double v0_3_short_request_scale = 1.0 - v0_3_long_request_scale;
  const bool use_v0_3_family_scoring = use_hybrid_prefill_scoring_v0_3_family();
  const bool use_v0_3a_scoring = use_hybrid_prefill_scoring_v0_3a();
  const bool use_v0_3b_scoring = use_hybrid_prefill_scoring_v0_3b();
  const bool use_v0_3c_scoring = use_hybrid_prefill_scoring_v0_3c();
  std::vector<std::string> all_prefill_instances =
      get_schedulable_prefill_instances(
          instances_, prefill_index_, has_unschedulable_instances);
  std::vector<std::string> candidate_prefill_instances = all_prefill_instances;
  const bool enable_affinity_routing =
      FLAGS_enable_hybrid_prefill_affinity_routing;
  uint64_t hybrid_route_decision_ordinal = 0;
  uint64_t hybrid_route_elapsed_ms = 0;
  static std::atomic<int32_t> consecutive_short_long_borrow_counter{0};
  if (enable_affinity_routing) {
    static const uint64_t hybrid_route_guard_start_ms = current_time_ms();
    static std::atomic<uint64_t> hybrid_route_decision_counter{0};
    hybrid_route_decision_ordinal =
        hybrid_route_decision_counter.fetch_add(1, std::memory_order_relaxed) +
        1;
    const uint64_t now_ms = current_time_ms();
    hybrid_route_elapsed_ms = now_ms >= hybrid_route_guard_start_ms
                                  ? now_ms - hybrid_route_guard_start_ms
                                  : 0;
  }
  bool static_soft_fallback_triggered = false;
  int64_t static_primary_min_prefill_request_num = 0;
  int64_t static_opposite_min_prefill_request_num = 0;
  int64_t static_primary_min_prefill_token_num = 0;
  int64_t static_opposite_min_prefill_token_num = 0;
  const std::vector<std::string> short_selectors =
      parse_instance_selectors(FLAGS_static_prefill_short_instance_selectors);
  const std::vector<std::string> long_selectors =
      parse_instance_selectors(FLAGS_static_prefill_long_instance_selectors);
  const std::vector<std::string> short_filtered_instances =
      filter_prefill_instances_by_selector(all_prefill_instances,
                                           short_selectors);
  const std::vector<std::string> long_filtered_instances =
      filter_prefill_instances_by_selector(all_prefill_instances,
                                           long_selectors);
  const std::vector<std::string>& primary_selectors =
      request_is_long ? long_selectors : short_selectors;
  const std::vector<std::string>& opposite_selectors =
      request_is_long ? short_selectors : long_selectors;
  const std::vector<std::string>& primary_filtered_instances =
      request_is_long ? long_filtered_instances : short_filtered_instances;
  const std::vector<std::string>& opposite_filtered_instances =
      request_is_long ? short_filtered_instances : long_filtered_instances;
  if (FLAGS_enable_static_prefill_instance_split && !enable_affinity_routing) {
    if (!primary_filtered_instances.empty()) {
      candidate_prefill_instances = primary_filtered_instances;
    } else if (!primary_selectors.empty()) {
      LOG_EVERY_N(WARNING, 20)
          << "Static prefill split enabled, but no candidate instance matched "
          << (request_is_long ? "long" : "short")
          << " selectors. Fall back to all schedulable prefill instances. "
          << "selectors=" << absl::StrJoin(primary_selectors, ",");
    }

    if (FLAGS_enable_static_prefill_soft_fallback &&
        !primary_filtered_instances.empty() &&
        !opposite_filtered_instances.empty()) {
      static_primary_min_prefill_request_num = min_prefill_request_num(
          primary_filtered_instances, request_metrics_);
      static_opposite_min_prefill_request_num = min_prefill_request_num(
          opposite_filtered_instances, request_metrics_);
      static_primary_min_prefill_token_num = min_prefill_token_num(
          primary_filtered_instances, request_metrics_);
      static_opposite_min_prefill_token_num = min_prefill_token_num(
          opposite_filtered_instances, request_metrics_);

      const bool request_gap_triggered =
          static_primary_min_prefill_request_num >
          static_opposite_min_prefill_request_num +
              FLAGS_static_prefill_soft_fallback_request_num_gap;
      const bool token_gap_triggered =
          static_primary_min_prefill_token_num >
          static_opposite_min_prefill_token_num +
              FLAGS_static_prefill_soft_fallback_token_gap;
      if (request_gap_triggered || token_gap_triggered) {
        candidate_prefill_instances = merge_unique_instance_lists(
            primary_filtered_instances, opposite_filtered_instances);
        static_soft_fallback_triggered = true;
      }
    }
  }

  std::string min_prefill_instance;
  int64_t min_prefill_time = std::numeric_limits<int64_t>::max();
  int64_t total_prefill_time = 0;
  size_t schedulable_prefill_count = 0;
  std::vector<std::string> min_prefill_candidates;
  for (const auto& prefill_instance : candidate_prefill_instances) {
    int64_t prefill_time =
        request_metrics_[prefill_instance].estimated_prefill_time;
    total_prefill_time += prefill_time;
    if (prefill_time < min_prefill_time) {
      min_prefill_instance = prefill_instance;
      min_prefill_time = prefill_time;
      min_prefill_candidates.clear();
      min_prefill_candidates.emplace_back(prefill_instance);
    } else if (prefill_time == min_prefill_time) {
      min_prefill_candidates.emplace_back(prefill_instance);
    }
    ++schedulable_prefill_count;
  }

  if (schedulable_prefill_count == 0) {
    LOG(ERROR) << "No prefill or default instance found!";
    return false;
  }
  int64_t avg_prefill_time = total_prefill_time / schedulable_prefill_count;

  std::string min_decode_instance;
  int64_t min_estimated_tpot = std::numeric_limits<int64_t>::max();
  std::string target_decode_instance;
  size_t schedulable_decode_count = 0;
  for (const auto& decode_instance : decode_index_) {
    if (has_unschedulable_instances) {
      auto it = instances_.find(decode_instance);
      if (it == instances_.end() || !is_instance_schedulable(it->second)) {
        continue;
      }
    }

    int64_t token_num = request_metrics_[decode_instance].decode_token_num;
    int64_t request_num = request_metrics_[decode_instance].decode_request_num;
    auto& time_predictor = get_time_predictor(decode_instance);
    int64_t estimated_tpot = time_predictor.predict_tpot(
        token_num + request->token_ids.size(), request_num + 1);
    if (estimated_tpot <= FLAGS_target_tpot && target_decode_instance.empty()) {
      target_decode_instance = decode_instance;
    }

    if (estimated_tpot < min_estimated_tpot) {
      min_decode_instance = decode_instance;
      min_estimated_tpot = estimated_tpot;
    }
    ++schedulable_decode_count;
  }

  if (schedulable_decode_count == 0) {
    LOG(ERROR) << "No decode instance found!";
    return false;
  }

  if (!target_decode_instance.empty()) {
    request->routing.decode_name = target_decode_instance;
  } else {
    request->routing.decode_name = min_decode_instance;
  }

  std::string selected_prefill_instance = min_prefill_instance;
  size_t min_prefill_token_tie_count = min_prefill_candidates.size();
  size_t min_prefill_request_tie_count = min_prefill_candidates.size();
  size_t min_prefill_waiting_tie_count = min_prefill_candidates.size();
  double selected_prefill_score = std::numeric_limits<double>::lowest();
  double selected_prefill_score_v0_2 = std::numeric_limits<double>::lowest();
  double selected_prefill_score_v0_3 = std::numeric_limits<double>::lowest();
  double selected_affinity_bonus = 0.0;
  double selected_rescue_bonus = 0.0;
  double selected_boundary_long_escape_bonus = 0.0;
  double selected_long_affine_busy_penalty = 0.0;
  bool selected_long_affine_busy_admission_guarded = false;
  double selected_long_affinity_base_bonus = 0.0;
  double selected_dynamic_long_affinity_bonus = 0.0;
  double selected_effective_long_affinity_bonus = 0.0;
  double selected_dynamic_load_gain = 0.0;
  double selected_dynamic_time_gain = 0.0;
  double selected_request_penalty = 0.0;
  double selected_waiting_penalty = 0.0;
  double selected_token_penalty = 0.0;
  double selected_prefill_time_penalty = 0.0;
  uint64_t selected_combined_load = 0;
  bool affine_pool_overloaded = false;
  bool rescue_token_eligible = false;
  uint64_t affine_pool_min_combined_load = 0;
  uint64_t opposite_pool_min_combined_load = 0;
  int64_t pool_min_combined_load_gap = 0;
  int64_t affine_pool_min_projected_prefill_time = 0;
  int64_t opposite_pool_min_projected_prefill_time = 0;
  bool has_affine_pool_projected_prefill_time = false;
  bool has_opposite_pool_projected_prefill_time = false;
  double dynamic_long_request_token_cost_load_gain = 0.0;
  double dynamic_long_request_token_cost_time_gain = 0.0;
  double dynamic_long_request_token_cost_gap_gain = 0.0;
  double dynamic_long_request_token_cost_multiplier =
      FLAGS_long_request_token_cost_multiplier;
  bool selected_prefill_affine = false;
  bool selected_prefill_short_to_long_lane = false;
  int32_t consecutive_short_long_borrows_before_route = 0;
  int32_t consecutive_short_long_borrows_after_route = 0;
  std::string selected_candidate_summary;
  std::string hybrid_candidate_summaries;
  uint64_t route_trace_sample_ordinal = 0;
  bool route_trace_sampled =
      should_sample_route_trace(request_is_long, &route_trace_sample_ordinal);
  nlohmann::json route_trace_candidate_breakdown = nlohmann::json::array();
  nlohmann::json dynamic_cp_shadow_candidate_breakdown =
      nlohmann::json::array();
  std::vector<std::string> hybrid_best_candidates_for_trace;
  struct DynamicCpShadowCandidate {
    std::string prefill_instance;
    std::string decode_instance;
    std::string lane_type;
    double cost = std::numeric_limits<double>::max();
    double prefill_cost = 0.0;
    double decode_pressure_cost = 0.0;
    double cp_benefit = 0.0;
    double interference_cost = 0.0;
    int64_t projected_prefill_time = 0;
    int64_t projected_prefill_tokens = 0;
    int64_t predicted_request_ttft = 0;
    int64_t active_requests = 0;
    uint64_t waiting_requests = 0;
    uint64_t combined_load = 0;
    int64_t decode_active_requests = -1;
    int64_t decode_waiting_requests = -1;
    int64_t decode_estimated_tpot_ms = -1;
  };
  bool dynamic_cp_shadow_has_selected = false;
  DynamicCpShadowCandidate dynamic_cp_shadow_selected_candidate;
  const bool dynamic_cp_shadow_enabled =
      FLAGS_enable_dynamic_cp_pricing_shadow;
  const bool dynamic_cp_takeover_enabled =
      dynamic_cp_shadow_enabled && FLAGS_enable_dynamic_cp_pricing_takeover;
  const bool dynamic_cp_decode_takeover_enabled =
      dynamic_cp_takeover_enabled &&
      FLAGS_enable_dynamic_cp_pricing_decode_takeover;
  const std::string decode_admission_preferred_decode =
      request->stage_timing_trace.decode_admission_preferred_decode_instance;

  auto dynamic_cp_shadow_lane_type =
      [&](const std::string& prefill_instance) -> std::string {
    const bool short_affine =
        short_filtered_instances.empty() ||
        instance_in_list(short_filtered_instances, prefill_instance);
    const bool long_affine =
        long_filtered_instances.empty() ||
        instance_in_list(long_filtered_instances, prefill_instance);
    if (long_affine && short_affine) {
      return "shared";
    }
    if (long_affine) {
      return "long";
    }
    if (short_affine) {
      return "short";
    }
    return "unclassified";
  };

  auto dynamic_cp_shadow_decode_cost =
      [&](const std::string& decode_instance,
          DynamicCpShadowCandidate* candidate) -> double {
    if (decode_instance.empty()) {
      return 0.0;
    }
    const auto metrics_it = request_metrics_.find(decode_instance);
    if (metrics_it == request_metrics_.end()) {
      return 0.0;
    }
    const int64_t active_requests = metrics_it->second.decode_request_num;
    const int64_t waiting_requests = static_cast<int64_t>(
        get_waiting_requests_num(load_metrics_, decode_instance));
    const int64_t token_num = metrics_it->second.decode_token_num;
    int64_t estimated_tpot_ms = -1;
    const auto instance_it = instances_.find(decode_instance);
    if (instance_it != instances_.end() &&
        !instance_it->second.tpot_profiling_data.empty()) {
      estimated_tpot_ms =
          get_time_predictor(decode_instance)
              .predict_tpot(token_num + decode_work_tokens,
                            active_requests + 1);
    }
    if (candidate != nullptr) {
      candidate->decode_active_requests = active_requests;
      candidate->decode_waiting_requests = waiting_requests;
      candidate->decode_estimated_tpot_ms = estimated_tpot_ms;
    }
    const int64_t effective_target_tpot =
        FLAGS_decode_pressure_admission_target_tpot_ms > 0
            ? FLAGS_decode_pressure_admission_target_tpot_ms
            : FLAGS_target_tpot;
    const double tpot_overflow =
        estimated_tpot_ms > 0 && effective_target_tpot > 0
            ? std::max(0.0,
                       static_cast<double>(estimated_tpot_ms -
                                           effective_target_tpot))
            : 0.0;
    return static_cast<double>(active_requests) *
               FLAGS_dynamic_cp_pricing_active_weight +
           static_cast<double>(waiting_requests) *
               FLAGS_dynamic_cp_pricing_waiting_weight +
           tpot_overflow * FLAGS_dynamic_cp_pricing_decode_tpot_weight;
  };

  auto evaluate_dynamic_cp_shadow_candidate =
      [&](const std::string& prefill_instance)
      -> DynamicCpShadowCandidate {
    DynamicCpShadowCandidate candidate;
    candidate.prefill_instance = prefill_instance;
    candidate.decode_instance = decode_admission_preferred_decode.empty()
                                    ? request->routing.decode_name
                                    : decode_admission_preferred_decode;
    candidate.lane_type = dynamic_cp_shadow_lane_type(prefill_instance);
    candidate.active_requests =
        get_prefill_request_num(request_metrics_, prefill_instance);
    candidate.waiting_requests =
        get_waiting_requests_num(load_metrics_, prefill_instance);
    candidate.combined_load = get_prefill_combined_load(
        request_metrics_, load_metrics_, prefill_instance);
    const int64_t current_prefill_tokens =
        get_prefill_token_num(request_metrics_, prefill_instance);
    candidate.predicted_request_ttft =
        get_time_predictor(prefill_instance).predict_ttft(request_tokens);
    candidate.projected_prefill_tokens =
        current_prefill_tokens + request_tokens;
    candidate.projected_prefill_time =
        request_metrics_[prefill_instance].estimated_prefill_time +
        candidate.predicted_request_ttft;
    candidate.prefill_cost =
        static_cast<double>(candidate.projected_prefill_time) *
            FLAGS_dynamic_cp_pricing_prefill_time_weight +
        static_cast<double>(candidate.projected_prefill_tokens) *
            FLAGS_dynamic_cp_pricing_token_weight +
        static_cast<double>(candidate.active_requests) *
            FLAGS_dynamic_cp_pricing_active_weight +
        static_cast<double>(candidate.waiting_requests) *
            FLAGS_dynamic_cp_pricing_waiting_weight;

    const bool short_affine =
        short_filtered_instances.empty() ||
        instance_in_list(short_filtered_instances, prefill_instance);
    const bool long_affine =
        long_filtered_instances.empty() ||
        instance_in_list(long_filtered_instances, prefill_instance);
    const bool long_to_short_lane =
        request_is_long && short_affine && !long_affine;
    if (long_to_short_lane) {
      const uint64_t threshold = static_cast<uint64_t>(
          FLAGS_hybrid_prefill_long_short_pool_busy_combined_load_threshold);
      const double load_over_threshold =
          threshold > 0
              ? std::max(0.0,
                         static_cast<double>(candidate.combined_load) -
                             static_cast<double>(threshold))
              : static_cast<double>(candidate.combined_load);
      candidate.interference_cost =
          (load_over_threshold +
           FLAGS_hybrid_prefill_long_short_pool_busy_surcharge +
           FLAGS_hybrid_prefill_long_short_pool_affine_gap_penalty) *
          FLAGS_dynamic_cp_pricing_interference_weight;
    }
    if (request_is_long && long_affine &&
        has_opposite_pool_projected_prefill_time &&
        opposite_pool_min_projected_prefill_time >
            candidate.projected_prefill_time) {
      candidate.cp_benefit =
          static_cast<double>(opposite_pool_min_projected_prefill_time -
                              candidate.projected_prefill_time) *
          FLAGS_dynamic_cp_pricing_cp_benefit_weight;
    }
    candidate.decode_pressure_cost =
        dynamic_cp_shadow_decode_cost(candidate.decode_instance, &candidate);
    candidate.cost = candidate.prefill_cost + candidate.decode_pressure_cost +
                     candidate.interference_cost - candidate.cp_benefit;
    return candidate;
  };

  auto dynamic_cp_shadow_candidate_to_json =
      [&](const DynamicCpShadowCandidate& candidate) -> nlohmann::json {
    return nlohmann::json{
        {"prefill_instance", candidate.prefill_instance},
        {"decode_instance", candidate.decode_instance},
        {"lane_type", candidate.lane_type},
        {"cost", candidate.cost},
        {"prefill_cost", candidate.prefill_cost},
        {"decode_pressure_cost", candidate.decode_pressure_cost},
        {"cp_benefit", candidate.cp_benefit},
        {"interference_cost", candidate.interference_cost},
        {"projected_prefill_time_ms", candidate.projected_prefill_time},
        {"projected_prefill_tokens", candidate.projected_prefill_tokens},
        {"predicted_request_ttft_ms", candidate.predicted_request_ttft},
        {"active_requests", candidate.active_requests},
        {"waiting_requests", candidate.waiting_requests},
        {"combined_load", candidate.combined_load},
        {"decode_active_requests", candidate.decode_active_requests},
        {"decode_waiting_requests", candidate.decode_waiting_requests},
        {"decode_estimated_tpot_ms", candidate.decode_estimated_tpot_ms}};
  };
  if (enable_affinity_routing) {
    rescue_token_eligible =
        request_tokens >= FLAGS_hybrid_prefill_rescue_min_tokens &&
        request_tokens <= FLAGS_hybrid_prefill_rescue_max_tokens;

    struct HybridCandidateScore {
      double score = std::numeric_limits<double>::lowest();
      double score_v0_2 = std::numeric_limits<double>::lowest();
      double score_v0_3 = std::numeric_limits<double>::lowest();
      double affinity_bonus = 0.0;
      double affinity_bonus_v0_2 = 0.0;
      double affinity_bonus_v0_3 = 0.0;
      double long_affinity_base_bonus = 0.0;
      double dynamic_long_affinity_bonus = 0.0;
      double effective_long_affinity_bonus = 0.0;
      double dynamic_load_gain = 0.0;
      double dynamic_time_gain = 0.0;
      double rescue_bonus = 0.0;
      double short_long_lane_penalty = 0.0;
      double score_before_short_long_lane_penalty = 0.0;
      double decode_pressure_route_cost = 0.0;
      double score_before_decode_pressure_route_cost = 0.0;
      double long_short_pool_busy_surcharge = 0.0;
      bool long_short_pool_busy_surcharge_guarded = false;
      bool long_short_pool_continuous_surcharge_enabled = false;
      double long_short_pool_affine_gap_penalty = 0.0;
      bool long_short_pool_affine_gap_penalty_guarded = false;
      double request_penalty = 0.0;
      double waiting_penalty = 0.0;
      double token_penalty = 0.0;
      double token_penalty_v0_2 = 0.0;
      double token_penalty_v0_3 = 0.0;
      double boundary_long_escape_bonus_v0_3 = 0.0;
      double long_affine_busy_penalty = 0.0;
      double prefill_time_penalty = 0.0;
      double prefill_time_penalty_v0_2 = 0.0;
      double prefill_time_penalty_v0_3 = 0.0;
      double normalized_projected_prefill_time_penalty = 0.0;
      uint64_t combined_load = 0;
      uint64_t waiting_requests = 0;
      int64_t active_requests = 0;
      int64_t current_prefill_tokens = 0;
      int64_t prefill_time = 0;
      int64_t predicted_request_ttft = 0;
      int64_t projected_prefill_tokens = 0;
      int64_t projected_prefill_time = 0;
      bool is_affine = false;
      bool is_short_affine = false;
      bool is_long_affine = false;
      double v0_3_long_request_scale = 0.0;
      double v0_3_short_request_scale = 0.0;
      double request_token_cost_multiplier_v0_2 = 1.0;
      double request_token_cost_multiplier_v0_3 = 1.0;
      bool is_short_to_long_lane = false;
      bool long_affine_busy_admission_guarded = false;
      bool short_long_lane_guarded = false;
      bool consecutive_borrow_cap_guard = false;
      std::string short_long_lane_guard_reason;
    };

    affine_pool_min_combined_load = 0;
    bool has_affine_candidates = false;
    for (const auto& prefill_instance : primary_filtered_instances) {
      const uint64_t combined_load =
          get_prefill_combined_load(request_metrics_, load_metrics_,
                                    prefill_instance);
      if (!has_affine_candidates || combined_load < affine_pool_min_combined_load) {
        affine_pool_min_combined_load = combined_load;
      }
      has_affine_candidates = true;
    }
    if (!opposite_filtered_instances.empty()) {
      opposite_pool_min_combined_load = min_prefill_combined_load(
          opposite_filtered_instances, request_metrics_, load_metrics_);
    }
    if (has_affine_candidates && !opposite_filtered_instances.empty()) {
      pool_min_combined_load_gap =
          static_cast<int64_t>(affine_pool_min_combined_load) -
          static_cast<int64_t>(opposite_pool_min_combined_load);
    }
    affine_pool_overloaded =
        has_affine_candidates && FLAGS_hybrid_prefill_overload_threshold > 0 &&
        affine_pool_min_combined_load >=
            static_cast<uint64_t>(FLAGS_hybrid_prefill_overload_threshold);
    dynamic_long_request_token_cost_multiplier =
        FLAGS_long_request_token_cost_multiplier;
    if (request_is_long && has_affine_candidates &&
        FLAGS_dynamic_long_request_token_cost_enabled) {
      const double load_gain_window = static_cast<double>(
          FLAGS_dynamic_long_request_token_cost_load_gain_window);
      if (load_gain_window > 0.0 &&
          FLAGS_dynamic_long_request_token_cost_overload_threshold > 0) {
        dynamic_long_request_token_cost_load_gain = clamp_unit_interval(
            (static_cast<double>(affine_pool_min_combined_load) -
             static_cast<double>(
                 FLAGS_dynamic_long_request_token_cost_overload_threshold)) /
            load_gain_window);
      } else if (FLAGS_dynamic_long_request_token_cost_overload_threshold == 0) {
        dynamic_long_request_token_cost_load_gain = 1.0;
      }
      if (load_gain_window > 0.0 && pool_min_combined_load_gap > 0) {
        dynamic_long_request_token_cost_gap_gain = clamp_unit_interval(
            static_cast<double>(pool_min_combined_load_gap) / load_gain_window);
      }
    }
    has_affine_pool_projected_prefill_time = false;
    has_opposite_pool_projected_prefill_time = false;

    auto compute_decode_pressure_route_cost =
        [&](int64_t projected_prefill_time_ms) -> double {
      if (!FLAGS_enable_decode_pressure_route_score ||
          FLAGS_decode_pressure_route_score_weight <= 0.0) {
        return 0.0;
      }
      const std::string decode_instance =
          request->stage_timing_trace.decode_admission_preferred_decode_instance
                  .empty()
              ? request->routing.decode_name
              : request->stage_timing_trace
                    .decode_admission_preferred_decode_instance;
      if (decode_instance.empty()) {
        return 0.0;
      }
      const auto metrics_it = request_metrics_.find(decode_instance);
      if (metrics_it == request_metrics_.end()) {
        return 0.0;
      }
      const int64_t active_requests = metrics_it->second.decode_request_num;
      const int64_t waiting_requests = static_cast<int64_t>(
          get_waiting_requests_num(load_metrics_, decode_instance));
      int64_t estimated_tpot_ms = -1;
      const auto instance_it = instances_.find(decode_instance);
      if (instance_it != instances_.end() &&
          !instance_it->second.tpot_profiling_data.empty()) {
        estimated_tpot_ms =
            get_time_predictor(decode_instance)
                .predict_tpot(metrics_it->second.decode_token_num +
                                  decode_work_tokens,
                              active_requests + 1);
      }
      const int64_t effective_target_tpot =
          FLAGS_decode_pressure_admission_target_tpot_ms > 0
              ? FLAGS_decode_pressure_admission_target_tpot_ms
              : FLAGS_target_tpot;
      const double tpot_overflow =
          estimated_tpot_ms > 0 && effective_target_tpot > 0
              ? std::max(0.0,
                         static_cast<double>(estimated_tpot_ms -
                                             effective_target_tpot))
              : 0.0;
      double raw_cost =
          static_cast<double>(active_requests) *
              FLAGS_decode_pressure_route_active_weight +
          static_cast<double>(waiting_requests) *
              FLAGS_decode_pressure_route_waiting_weight +
          tpot_overflow * FLAGS_decode_pressure_route_estimated_tpot_weight;
      if (FLAGS_decode_pressure_route_prefill_delay_decay_ms > 0 &&
          projected_prefill_time_ms > 0) {
        raw_cost *= std::exp(
            -static_cast<double>(projected_prefill_time_ms) /
            static_cast<double>(
                FLAGS_decode_pressure_route_prefill_delay_decay_ms));
      }
      double aging_credit = 0.0;
      if (request_is_long &&
          FLAGS_decode_pressure_admission_long_rescue_wait_ms > 0 &&
          FLAGS_decode_pressure_route_long_aging_credit_weight > 0.0) {
        aging_credit =
            static_cast<double>(
                request->stage_timing_trace.decode_admission_wait_ms) /
            static_cast<double>(
                FLAGS_decode_pressure_admission_long_rescue_wait_ms) *
            FLAGS_decode_pressure_route_long_aging_credit_weight;
        if (FLAGS_decode_pressure_route_long_aging_credit_cap > 0.0) {
          aging_credit = std::min(
              aging_credit, FLAGS_decode_pressure_route_long_aging_credit_cap);
        }
      }
      return std::max(0.0,
                      raw_cost * FLAGS_decode_pressure_route_score_weight -
                          aging_credit);
    };

    auto evaluate_hybrid_candidate =
        [&](const std::string& prefill_instance) -> HybridCandidateScore {
      HybridCandidateScore candidate;
      candidate.is_affine =
          primary_filtered_instances.empty() ||
          instance_in_list(primary_filtered_instances, prefill_instance);
      candidate.is_short_affine =
          short_filtered_instances.empty() ||
          instance_in_list(short_filtered_instances, prefill_instance);
      candidate.is_long_affine =
          long_filtered_instances.empty() ||
          instance_in_list(long_filtered_instances, prefill_instance);
      candidate.active_requests =
          get_prefill_request_num(request_metrics_, prefill_instance);
      candidate.waiting_requests =
          get_waiting_requests_num(load_metrics_, prefill_instance);
      candidate.current_prefill_tokens =
          get_prefill_token_num(request_metrics_, prefill_instance);
      candidate.combined_load = get_prefill_combined_load(
          request_metrics_, load_metrics_, prefill_instance);
      const uint64_t rescue_load_gap =
          static_cast<uint64_t>(FLAGS_hybrid_prefill_rescue_load_gap);
      const bool rescue_load_advantaged =
          !candidate.is_affine &&
          candidate.combined_load + rescue_load_gap <=
              affine_pool_min_combined_load;
      if (!candidate.is_affine && affine_pool_overloaded &&
          rescue_token_eligible && rescue_load_advantaged) {
        candidate.rescue_bonus = FLAGS_hybrid_prefill_rescue_bonus;
      }
      candidate.is_short_to_long_lane =
          !request_is_long && candidate.is_long_affine &&
          !candidate.is_short_affine;
      const bool candidate_is_long_to_short_pool =
          request_is_long && candidate.is_short_affine &&
          !candidate.is_long_affine;
      if (request_is_long && candidate.is_affine && candidate.is_long_affine &&
          FLAGS_enable_hybrid_prefill_long_affine_busy_admission &&
          FLAGS_hybrid_prefill_long_affine_busy_combined_load_threshold > 0 &&
          candidate.combined_load >=
              static_cast<uint64_t>(
                  FLAGS_hybrid_prefill_long_affine_busy_combined_load_threshold)) {
        candidate.long_affine_busy_admission_guarded = true;
        candidate.long_affine_busy_penalty =
            FLAGS_hybrid_prefill_long_affine_busy_penalty;
      }
      if (candidate_is_long_to_short_pool &&
          FLAGS_enable_hybrid_prefill_long_short_pool_busy_surcharge &&
          candidate.combined_load >=
              static_cast<uint64_t>(
                  FLAGS_hybrid_prefill_long_short_pool_busy_combined_load_threshold)) {
        candidate.long_short_pool_busy_surcharge_guarded = true;
        if (FLAGS_enable_hybrid_prefill_long_short_pool_continuous_surcharge) {
          candidate.long_short_pool_continuous_surcharge_enabled = true;
          candidate.long_short_pool_busy_surcharge =
              FLAGS_hybrid_prefill_long_short_pool_busy_surcharge_alpha *
              std::max(
                  0.0,
                  static_cast<double>(candidate.combined_load) -
                      static_cast<double>(
                          FLAGS_hybrid_prefill_long_short_pool_busy_combined_load_threshold));
        } else {
          candidate.long_short_pool_busy_surcharge =
              FLAGS_hybrid_prefill_long_short_pool_busy_surcharge;
        }
      }
      if (candidate_is_long_to_short_pool &&
          has_affine_candidates &&
          FLAGS_enable_hybrid_prefill_long_short_pool_affine_gap_penalty) {
        const uint64_t required_gap = static_cast<uint64_t>(
            FLAGS_hybrid_prefill_long_short_pool_affine_gap_threshold);
        const bool lacks_clear_load_advantage =
            candidate.combined_load + required_gap >=
            affine_pool_min_combined_load;
        if (lacks_clear_load_advantage) {
          candidate.long_short_pool_affine_gap_penalty_guarded = true;
          candidate.long_short_pool_affine_gap_penalty =
              FLAGS_hybrid_prefill_long_short_pool_affine_gap_penalty;
        }
      }
      if (candidate.is_short_to_long_lane) {
        const bool early_time_guard =
            FLAGS_hybrid_prefill_short_long_lane_guard_window_ms > 0 &&
            hybrid_route_elapsed_ms <
                static_cast<uint64_t>(
                    FLAGS_hybrid_prefill_short_long_lane_guard_window_ms);
        const bool early_count_guard =
            FLAGS_hybrid_prefill_short_long_lane_guard_request_count > 0 &&
            hybrid_route_decision_ordinal <=
                static_cast<uint64_t>(
                    FLAGS_hybrid_prefill_short_long_lane_guard_request_count);
        const bool long_lane_busy_guard =
            candidate.combined_load >
            static_cast<uint64_t>(
                FLAGS_hybrid_prefill_short_long_lane_guard_max_long_lane_combined_load);
        const bool primary_not_busy_guard =
            FLAGS_hybrid_prefill_short_long_lane_guard_primary_min_load_threshold >
                0 &&
            has_affine_candidates &&
            affine_pool_min_combined_load <
                static_cast<uint64_t>(
                    FLAGS_hybrid_prefill_short_long_lane_guard_primary_min_load_threshold);
        const bool prompt_too_large_guard =
            FLAGS_hybrid_prefill_short_long_lane_guard_max_prompt_tokens > 0 &&
            request_tokens >
                FLAGS_hybrid_prefill_short_long_lane_guard_max_prompt_tokens;
        const bool consecutive_borrow_cap_guard =
            FLAGS_hybrid_prefill_short_long_lane_guard_max_consecutive_borrows >
                0 &&
            consecutive_short_long_borrows_before_route >=
                FLAGS_hybrid_prefill_short_long_lane_guard_max_consecutive_borrows;
        candidate.consecutive_borrow_cap_guard = consecutive_borrow_cap_guard;
        if (FLAGS_enable_hybrid_prefill_short_long_lane_guard &&
            (early_time_guard || early_count_guard || long_lane_busy_guard ||
             primary_not_busy_guard || prompt_too_large_guard ||
             consecutive_borrow_cap_guard)) {
          candidate.short_long_lane_guarded = true;
          std::vector<std::string> reasons;
          if (early_time_guard) {
            reasons.emplace_back("early_time");
          }
          if (early_count_guard) {
            reasons.emplace_back("early_count");
          }
          if (long_lane_busy_guard) {
            reasons.emplace_back("long_lane_busy");
          }
          if (primary_not_busy_guard) {
            reasons.emplace_back("primary_short_lane_available");
          }
          if (prompt_too_large_guard) {
            reasons.emplace_back("prompt_too_large");
          }
          candidate.short_long_lane_guard_reason = absl::StrJoin(reasons, ",");
        }
        candidate.short_long_lane_penalty =
            FLAGS_hybrid_prefill_short_long_lane_penalty;
      }
      candidate.request_penalty =
          static_cast<double>(candidate.active_requests) *
          FLAGS_hybrid_prefill_request_num_weight;
      candidate.waiting_penalty =
          static_cast<double>(candidate.waiting_requests) *
          FLAGS_hybrid_prefill_waiting_requests_weight;
      // P6: grace period for freshly-activated elastic instances. Within
      // pool_elastic_grace_period_s after the last IDLE->ACTIVE transition,
      // scale the waiting_penalty by pool_elastic_grace_waiting_factor (default
      // 0.5). This prevents the "first request lands -> waiting+1 -> next
      // request avoids it -> waiting stays high vs other ACTIVE -> never
      // selected again" feedback trap that we saw in P2 r1 with
      // waiting_requests_weight=28.
      if (FLAGS_pool_elastic_grace_period_s > 0) {
        const auto inst_it = instances_.find(prefill_instance);
        if (inst_it != instances_.end() &&
            inst_it->second.last_activated_ts_ms != 0) {
          const uint64_t now_ms_grace = current_time_ms();
          const uint64_t age_ms =
              now_ms_grace - inst_it->second.last_activated_ts_ms;
          const uint64_t grace_ms = static_cast<uint64_t>(std::max<int32_t>(
              0, FLAGS_pool_elastic_grace_period_s)) * 1000ULL;
          if (age_ms < grace_ms) {
            candidate.waiting_penalty *=
                FLAGS_pool_elastic_grace_waiting_factor;
          }
        }
      }
      candidate.prefill_time =
          request_metrics_[prefill_instance].estimated_prefill_time;
      candidate.predicted_request_ttft =
          get_time_predictor(prefill_instance).predict_ttft(request_tokens);
      candidate.projected_prefill_tokens =
          candidate.current_prefill_tokens + request_tokens;
      candidate.projected_prefill_time =
          candidate.prefill_time + candidate.predicted_request_ttft;
      candidate.normalized_projected_prefill_time_penalty =
          lane_normalized_projected_prefill_time_penalty(
              candidate.projected_prefill_time, candidate.is_long_affine,
              candidate.is_short_affine);
      if (request_is_long &&
          FLAGS_hybrid_prefill_dynamic_long_affinity_gain_enabled &&
          candidate.is_affine) {
        const double load_gain_window = static_cast<double>(
            FLAGS_hybrid_prefill_dynamic_load_gain_window);
        if (load_gain_window > 0.0 &&
            FLAGS_hybrid_prefill_overload_threshold > 0) {
          candidate.dynamic_load_gain = clamp_unit_interval(
              (static_cast<double>(affine_pool_min_combined_load) -
               static_cast<double>(FLAGS_hybrid_prefill_overload_threshold)) /
              load_gain_window);
        }
        candidate.long_affinity_base_bonus =
            FLAGS_hybrid_prefill_long_affinity_base_bonus;
        candidate.effective_long_affinity_bonus =
            candidate.long_affinity_base_bonus;
      }
      candidate.request_token_cost_multiplier_v0_2 =
          request_is_long ? dynamic_long_request_token_cost_multiplier : 1.0;
      candidate.token_penalty_v0_2 =
          static_cast<double>(candidate.current_prefill_tokens) *
          FLAGS_hybrid_prefill_token_num_weight *
          candidate.request_token_cost_multiplier_v0_2;
      candidate.prefill_time_penalty_v0_2 =
          candidate.normalized_projected_prefill_time_penalty +
          (FLAGS_enable_lane_specific_projected_prefill_time_route_signal
              ? static_cast<double>(candidate.projected_prefill_time) *
                    (candidate.is_long_affine
                         ? FLAGS_hybrid_prefill_long_lane_projected_prefill_time_weight
                         : (candidate.is_short_affine
                                ? FLAGS_hybrid_prefill_short_pool_projected_prefill_time_weight
                                : FLAGS_prefill_time_route_weight))
              : (FLAGS_enable_prefill_time_route_signal
                     ? static_cast<double>(candidate.prefill_time) *
                           FLAGS_prefill_time_route_weight
                     : 0.0));
      if (candidate.is_affine) {
        if (request_is_long) {
          candidate.affinity_bonus_v0_2 =
              FLAGS_hybrid_prefill_dynamic_long_affinity_gain_enabled
                  ? candidate.long_affinity_base_bonus
                  : FLAGS_hybrid_prefill_long_affinity_bonus;
        } else {
          candidate.affinity_bonus_v0_2 =
              FLAGS_hybrid_prefill_short_affinity_bonus;
        }
      }
      candidate.score_v0_2 = candidate.affinity_bonus_v0_2 +
                             candidate.rescue_bonus -
                             candidate.long_short_pool_busy_surcharge -
                             candidate.long_short_pool_affine_gap_penalty -
                             candidate.long_affine_busy_penalty -
                             candidate.request_penalty -
                             candidate.waiting_penalty -
                             candidate.token_penalty_v0_2 -
                             candidate.prefill_time_penalty_v0_2;

      candidate.v0_3_long_request_scale = v0_3_long_request_scale;
      candidate.v0_3_short_request_scale = v0_3_short_request_scale;
      candidate.request_token_cost_multiplier_v0_3 =
          1.0 + candidate.v0_3_long_request_scale *
                    (FLAGS_long_request_token_cost_multiplier - 1.0);
      if (use_v0_3a_scoring || use_v0_3b_scoring || use_v0_3c_scoring) {
        // v0_3a keeps the proven hard affinity prior from v0_2 and only swaps
        // in projected-cost penalties, so we can isolate whether soft affinity
        // itself caused gray-band regressions.
        candidate.affinity_bonus_v0_3 = candidate.affinity_bonus_v0_2;
      } else if (short_filtered_instances.empty() &&
                 long_filtered_instances.empty()) {
        candidate.affinity_bonus_v0_3 =
            candidate.v0_3_short_request_scale *
                FLAGS_hybrid_prefill_short_affinity_bonus +
            candidate.v0_3_long_request_scale *
                FLAGS_hybrid_prefill_long_affinity_bonus;
      } else {
        candidate.affinity_bonus_v0_3 =
            (candidate.is_short_affine
                 ? candidate.v0_3_short_request_scale *
                       FLAGS_hybrid_prefill_short_affinity_bonus
                 : 0.0) +
            (candidate.is_long_affine
                 ? candidate.v0_3_long_request_scale *
                       FLAGS_hybrid_prefill_long_affinity_bonus
                 : 0.0);
      }
      candidate.token_penalty_v0_3 =
          static_cast<double>(candidate.projected_prefill_tokens) *
          FLAGS_hybrid_prefill_token_num_weight *
          candidate.request_token_cost_multiplier_v0_3;
      if (use_v0_3b_scoring && request_is_long &&
          candidate.v0_3_long_request_scale > 0.0 &&
          candidate.v0_3_long_request_scale < 1.0) {
        candidate.token_penalty_v0_3 *=
            FLAGS_hybrid_prefill_v0_3b_boundary_long_projected_token_penalty_scale;
      }
      const bool boundary_long_request =
          request_is_long && candidate.v0_3_long_request_scale > 0.0 &&
          candidate.v0_3_long_request_scale < 1.0;
      const bool boundary_long_escape_allowed =
          use_v0_3c_scoring && boundary_long_request && !candidate.is_affine &&
          has_affine_candidates && rescue_load_advantaged &&
          pool_min_combined_load_gap >=
              FLAGS_hybrid_prefill_v0_3c_boundary_long_escape_load_gap &&
          affine_pool_min_combined_load >=
              static_cast<uint64_t>(
                  FLAGS_hybrid_prefill_v0_3c_boundary_long_escape_min_affine_load);
      if (boundary_long_escape_allowed) {
        candidate.boundary_long_escape_bonus_v0_3 =
            FLAGS_hybrid_prefill_v0_3c_boundary_long_escape_bonus;
      }
      candidate.prefill_time_penalty_v0_3 =
          candidate.normalized_projected_prefill_time_penalty +
          (FLAGS_enable_lane_specific_projected_prefill_time_route_signal
              ? static_cast<double>(candidate.projected_prefill_time) *
                    (candidate.is_long_affine
                         ? FLAGS_hybrid_prefill_long_lane_projected_prefill_time_weight
                         : (candidate.is_short_affine
                                ? FLAGS_hybrid_prefill_short_pool_projected_prefill_time_weight
                                : FLAGS_prefill_time_route_weight))
              : (FLAGS_enable_prefill_time_route_signal
                     ? static_cast<double>(candidate.projected_prefill_time) *
                           FLAGS_prefill_time_route_weight
                     : 0.0));
      candidate.score_v0_3 = candidate.affinity_bonus_v0_3 +
                             candidate.rescue_bonus -
                             candidate.long_short_pool_busy_surcharge -
                             candidate.long_short_pool_affine_gap_penalty -
                             candidate.long_affine_busy_penalty -
                             candidate.request_penalty -
                             candidate.waiting_penalty +
                             candidate.boundary_long_escape_bonus_v0_3 -
                             candidate.token_penalty_v0_3 -
                             candidate.prefill_time_penalty_v0_3;

      candidate.score_before_short_long_lane_penalty =
          use_v0_3_family_scoring ? candidate.score_v0_3 : candidate.score_v0_2;
      if (candidate.short_long_lane_penalty > 0.0) {
        candidate.score_v0_2 -= candidate.short_long_lane_penalty;
        candidate.score_v0_3 -= candidate.short_long_lane_penalty;
      }
      candidate.score =
          use_v0_3_family_scoring ? candidate.score_v0_3 : candidate.score_v0_2;
      candidate.decode_pressure_route_cost =
          compute_decode_pressure_route_cost(candidate.projected_prefill_time);
      candidate.score_before_decode_pressure_route_cost = candidate.score;
      if (candidate.decode_pressure_route_cost > 0.0) {
        candidate.score_v0_2 -= candidate.decode_pressure_route_cost;
        candidate.score_v0_3 -= candidate.decode_pressure_route_cost;
        candidate.score -= candidate.decode_pressure_route_cost;
      }
      if (candidate.short_long_lane_guarded &&
          FLAGS_hybrid_prefill_short_long_lane_guard_hard_reject) {
        candidate.score = std::numeric_limits<double>::lowest();
      }
      candidate.affinity_bonus =
          use_v0_3_family_scoring ? candidate.affinity_bonus_v0_3
                                  : candidate.affinity_bonus_v0_2;
      candidate.token_penalty =
          use_v0_3_family_scoring ? candidate.token_penalty_v0_3
                                  : candidate.token_penalty_v0_2;
      candidate.prefill_time_penalty =
          use_v0_3_family_scoring ? candidate.prefill_time_penalty_v0_3
                                  : candidate.prefill_time_penalty_v0_2;
      return candidate;
    };

    auto format_hybrid_candidate =
        [&](const std::string& prefill_instance,
            const HybridCandidateScore& candidate_score) -> std::string {
      std::ostringstream oss;
      oss << prefill_instance << "{"
          << "score=" << candidate_score.score
          << ",score_v0_2=" << candidate_score.score_v0_2
          << ",score_v0_3=" << candidate_score.score_v0_3
          << ",affine=" << candidate_score.is_affine
          << ",short_affine=" << candidate_score.is_short_affine
          << ",long_affine=" << candidate_score.is_long_affine
          << ",short_to_long_lane=" << candidate_score.is_short_to_long_lane
          << ",long_affine_busy_guarded="
          << candidate_score.long_affine_busy_admission_guarded
          << ",long_affine_busy_penalty="
          << candidate_score.long_affine_busy_penalty
          << ",long_short_pool_busy_surcharge_guarded="
          << candidate_score.long_short_pool_busy_surcharge_guarded
          << ",long_short_pool_busy_surcharge="
          << candidate_score.long_short_pool_busy_surcharge
          << ",long_short_pool_continuous_surcharge="
          << candidate_score.long_short_pool_continuous_surcharge_enabled
          << ",long_short_pool_affine_gap_penalty_guarded="
          << candidate_score.long_short_pool_affine_gap_penalty_guarded
          << ",long_short_pool_affine_gap_penalty="
          << candidate_score.long_short_pool_affine_gap_penalty
          << ",short_long_guarded=" << candidate_score.short_long_lane_guarded
          << ",short_long_guard_reason="
          << candidate_score.short_long_lane_guard_reason
          << ",prefill_time=" << candidate_score.prefill_time
          << ",predicted_request_ttft="
          << candidate_score.predicted_request_ttft
          << ",projected_prefill_time="
          << candidate_score.projected_prefill_time
          << ",active=" << candidate_score.active_requests
          << ",waiting=" << candidate_score.waiting_requests
          << ",combined=" << candidate_score.combined_load
          << ",prefill_tokens=" << candidate_score.current_prefill_tokens
          << ",projected_prefill_tokens="
          << candidate_score.projected_prefill_tokens
          << ",aff_bonus=" << candidate_score.affinity_bonus
          << ",aff_bonus_v0_2=" << candidate_score.affinity_bonus_v0_2
          << ",aff_bonus_v0_3=" << candidate_score.affinity_bonus_v0_3
          << ",long_aff_base_bonus="
          << candidate_score.long_affinity_base_bonus
          << ",dynamic_long_aff_bonus="
          << candidate_score.dynamic_long_affinity_bonus
          << ",effective_long_aff_bonus="
          << candidate_score.effective_long_affinity_bonus
          << ",dynamic_load_gain=" << candidate_score.dynamic_load_gain
          << ",dynamic_time_gain=" << candidate_score.dynamic_time_gain
          << ",rescue_bonus=" << candidate_score.rescue_bonus
          << ",short_long_lane_penalty="
          << candidate_score.short_long_lane_penalty
          << ",score_before_short_long_lane_penalty="
          << candidate_score.score_before_short_long_lane_penalty
          << ",decode_pressure_route_cost="
          << candidate_score.decode_pressure_route_cost
          << ",score_before_decode_pressure_route_cost="
          << candidate_score.score_before_decode_pressure_route_cost
          << ",boundary_long_escape_bonus="
          << candidate_score.boundary_long_escape_bonus_v0_3
          << ",req_pen=" << candidate_score.request_penalty
          << ",wait_pen=" << candidate_score.waiting_penalty
          << ",token_pen=" << candidate_score.token_penalty
          << ",token_pen_v0_2=" << candidate_score.token_penalty_v0_2
          << ",token_pen_v0_3=" << candidate_score.token_penalty_v0_3
          << ",prefill_pen=" << candidate_score.prefill_time_penalty
          << ",prefill_pen_v0_2=" << candidate_score.prefill_time_penalty_v0_2
          << ",prefill_pen_v0_3=" << candidate_score.prefill_time_penalty_v0_3
          << ",normalized_projected_prefill_pen="
          << candidate_score.normalized_projected_prefill_time_penalty
          << ",dynamic_long_req_token_cost_load_gain="
          << dynamic_long_request_token_cost_load_gain
          << ",dynamic_long_req_token_cost_time_gain="
          << dynamic_long_request_token_cost_time_gain
          << ",dynamic_long_req_token_cost_gap_gain="
          << dynamic_long_request_token_cost_gap_gain
          << ",dynamic_long_req_token_cost_mult="
          << dynamic_long_request_token_cost_multiplier
          << ",token_cost_mult_v0_2="
          << candidate_score.request_token_cost_multiplier_v0_2
          << ",token_cost_mult_v0_3="
          << candidate_score.request_token_cost_multiplier_v0_3
          << ",v0_3_short_scale="
          << candidate_score.v0_3_short_request_scale
          << ",v0_3_long_scale="
          << candidate_score.v0_3_long_request_scale
          << "}";
      return oss.str();
    };

    auto route_trace_candidate_to_json =
        [&](const std::string& prefill_instance,
            const HybridCandidateScore& candidate_score) -> nlohmann::json {
      return nlohmann::json{
          {"instance", prefill_instance},
          {"score", candidate_score.score},
          {"score_v0_2", candidate_score.score_v0_2},
          {"score_v0_3", candidate_score.score_v0_3},
          {"is_affine", candidate_score.is_affine},
          {"is_short_affine", candidate_score.is_short_affine},
          {"is_long_affine", candidate_score.is_long_affine},
          {"is_short_to_long_lane", candidate_score.is_short_to_long_lane},
          {"short_long_lane_guarded",
           candidate_score.short_long_lane_guarded},
          {"long_affine_busy_admission_guarded",
           candidate_score.long_affine_busy_admission_guarded},
          {"long_affine_busy_penalty",
           candidate_score.long_affine_busy_penalty},
          {"long_short_pool_busy_surcharge_guarded",
           candidate_score.long_short_pool_busy_surcharge_guarded},
          {"long_short_pool_busy_surcharge",
           candidate_score.long_short_pool_busy_surcharge},
          {"long_short_pool_continuous_surcharge_enabled",
           candidate_score.long_short_pool_continuous_surcharge_enabled},
          {"long_short_pool_affine_gap_penalty_guarded",
           candidate_score.long_short_pool_affine_gap_penalty_guarded},
          {"long_short_pool_affine_gap_penalty",
           candidate_score.long_short_pool_affine_gap_penalty},
          {"short_long_lane_guard_reason",
           candidate_score.short_long_lane_guard_reason},
          {"consecutive_borrow_cap_guard",
           candidate_score.consecutive_borrow_cap_guard},
          {"prefill_time_ms", candidate_score.prefill_time},
          {"predicted_request_ttft_ms",
           candidate_score.predicted_request_ttft},
          {"projected_prefill_time_ms",
           candidate_score.projected_prefill_time},
          {"active_requests", candidate_score.active_requests},
          {"waiting_requests", candidate_score.waiting_requests},
          {"combined_load", candidate_score.combined_load},
          {"current_prefill_tokens",
           candidate_score.current_prefill_tokens},
          {"projected_prefill_tokens",
           candidate_score.projected_prefill_tokens},
          {"affinity_bonus", candidate_score.affinity_bonus},
          {"affinity_bonus_v0_2", candidate_score.affinity_bonus_v0_2},
          {"affinity_bonus_v0_3", candidate_score.affinity_bonus_v0_3},
          {"long_affinity_base_bonus",
           candidate_score.long_affinity_base_bonus},
          {"dynamic_long_affinity_bonus",
           candidate_score.dynamic_long_affinity_bonus},
          {"effective_long_affinity_bonus",
           candidate_score.effective_long_affinity_bonus},
          {"dynamic_load_gain", candidate_score.dynamic_load_gain},
          {"dynamic_time_gain", candidate_score.dynamic_time_gain},
          {"rescue_bonus", candidate_score.rescue_bonus},
          {"short_long_lane_penalty",
           candidate_score.short_long_lane_penalty},
          {"score_before_short_long_lane_penalty",
           candidate_score.score_before_short_long_lane_penalty},
          {"decode_pressure_route_cost",
           candidate_score.decode_pressure_route_cost},
          {"score_before_decode_pressure_route_cost",
           candidate_score.score_before_decode_pressure_route_cost},
          {"boundary_long_escape_bonus_v0_3",
           candidate_score.boundary_long_escape_bonus_v0_3},
          {"request_penalty", candidate_score.request_penalty},
          {"waiting_penalty", candidate_score.waiting_penalty},
          {"token_penalty", candidate_score.token_penalty},
          {"token_penalty_v0_2", candidate_score.token_penalty_v0_2},
          {"token_penalty_v0_3", candidate_score.token_penalty_v0_3},
          {"prefill_time_penalty", candidate_score.prefill_time_penalty},
          {"prefill_time_penalty_v0_2",
           candidate_score.prefill_time_penalty_v0_2},
          {"prefill_time_penalty_v0_3",
           candidate_score.prefill_time_penalty_v0_3},
          {"normalized_projected_prefill_time_penalty",
           candidate_score.normalized_projected_prefill_time_penalty},
          {"dynamic_long_request_token_cost_load_gain",
           dynamic_long_request_token_cost_load_gain},
          {"dynamic_long_request_token_cost_time_gain",
           dynamic_long_request_token_cost_time_gain},
          {"dynamic_long_request_token_cost_gap_gain",
           dynamic_long_request_token_cost_gap_gain},
          {"dynamic_long_request_token_cost_multiplier",
           dynamic_long_request_token_cost_multiplier},
          {"request_token_cost_multiplier_v0_2",
           candidate_score.request_token_cost_multiplier_v0_2},
          {"request_token_cost_multiplier_v0_3",
           candidate_score.request_token_cost_multiplier_v0_3},
          {"v0_3_short_request_scale",
           candidate_score.v0_3_short_request_scale},
          {"v0_3_long_request_scale",
           candidate_score.v0_3_long_request_scale}};
    };

    std::vector<std::string> hybrid_best_candidates;
    HybridCandidateScore best_candidate_score;
    std::vector<std::string> candidate_summaries;
    std::vector<std::pair<std::string, HybridCandidateScore>> candidate_scores;
    candidate_scores.reserve(all_prefill_instances.size());
    for (const auto& prefill_instance : all_prefill_instances) {
      HybridCandidateScore candidate_score =
          evaluate_hybrid_candidate(prefill_instance);
      if (candidate_score.is_affine) {
        if (!has_affine_pool_projected_prefill_time ||
            candidate_score.projected_prefill_time <
                affine_pool_min_projected_prefill_time) {
          affine_pool_min_projected_prefill_time =
              candidate_score.projected_prefill_time;
          has_affine_pool_projected_prefill_time = true;
        }
      } else {
        if (!has_opposite_pool_projected_prefill_time ||
            candidate_score.projected_prefill_time <
                opposite_pool_min_projected_prefill_time) {
          opposite_pool_min_projected_prefill_time =
              candidate_score.projected_prefill_time;
          has_opposite_pool_projected_prefill_time = true;
        }
      }
      candidate_scores.emplace_back(prefill_instance, candidate_score);
    }

    if (request_is_long && has_affine_candidates &&
        FLAGS_dynamic_long_request_token_cost_enabled) {
      if (!has_opposite_pool_projected_prefill_time) {
        dynamic_long_request_token_cost_time_gain = 1.0;
      } else {
        const double margin_ms = static_cast<double>(
            FLAGS_hybrid_prefill_dynamic_projected_prefill_time_margin_ms);
        if (margin_ms <= 0.0) {
          dynamic_long_request_token_cost_time_gain =
              affine_pool_min_projected_prefill_time <=
                      opposite_pool_min_projected_prefill_time
                  ? 1.0
                  : 0.0;
        } else {
          dynamic_long_request_token_cost_time_gain = clamp_unit_interval(
              (static_cast<double>(opposite_pool_min_projected_prefill_time) +
               margin_ms -
               static_cast<double>(affine_pool_min_projected_prefill_time)) /
              margin_ms);
        }
      }
      dynamic_long_request_token_cost_multiplier =
          FLAGS_long_request_token_cost_multiplier +
          FLAGS_dynamic_long_request_token_cost_extra_max *
              dynamic_long_request_token_cost_load_gain *
              std::max(dynamic_long_request_token_cost_time_gain,
                       dynamic_long_request_token_cost_gap_gain);
    }

    for (auto& candidate_entry : candidate_scores) {
      const std::string& prefill_instance = candidate_entry.first;
      HybridCandidateScore& candidate_score = candidate_entry.second;
      if (request_is_long &&
          FLAGS_hybrid_prefill_dynamic_long_affinity_gain_enabled &&
          candidate_score.is_affine) {
        if (!has_opposite_pool_projected_prefill_time) {
          candidate_score.dynamic_time_gain = 1.0;
        } else {
          const double margin_ms = static_cast<double>(
              FLAGS_hybrid_prefill_dynamic_projected_prefill_time_margin_ms);
          if (margin_ms <= 0.0) {
            candidate_score.dynamic_time_gain =
                candidate_score.projected_prefill_time <=
                        opposite_pool_min_projected_prefill_time
                    ? 1.0
                    : 0.0;
          } else {
            candidate_score.dynamic_time_gain = clamp_unit_interval(
                (static_cast<double>(opposite_pool_min_projected_prefill_time) +
                 margin_ms -
                 static_cast<double>(candidate_score.projected_prefill_time)) /
                margin_ms);
          }
        }
        candidate_score.dynamic_long_affinity_bonus =
            FLAGS_hybrid_prefill_long_affinity_dynamic_bonus_max *
            candidate_score.dynamic_load_gain * candidate_score.dynamic_time_gain;
        candidate_score.effective_long_affinity_bonus =
            candidate_score.long_affinity_base_bonus +
            candidate_score.dynamic_long_affinity_bonus;
        candidate_score.affinity_bonus_v0_2 =
            candidate_score.effective_long_affinity_bonus;
        candidate_score.score_v0_2 = candidate_score.affinity_bonus_v0_2 +
                                     candidate_score.rescue_bonus -
                                     candidate_score.long_short_pool_busy_surcharge -
                                     candidate_score.long_short_pool_affine_gap_penalty -
                                     candidate_score.long_affine_busy_penalty -
                                     candidate_score.request_penalty -
                                     candidate_score.waiting_penalty -
                                     candidate_score.token_penalty_v0_2 -
                                     candidate_score.prefill_time_penalty_v0_2;
        if (use_v0_3a_scoring || use_v0_3b_scoring || use_v0_3c_scoring) {
          candidate_score.affinity_bonus_v0_3 =
              candidate_score.affinity_bonus_v0_2;
          candidate_score.score_v0_3 =
              candidate_score.affinity_bonus_v0_3 +
              candidate_score.rescue_bonus -
              candidate_score.long_short_pool_busy_surcharge -
              candidate_score.long_short_pool_affine_gap_penalty -
              candidate_score.long_affine_busy_penalty -
              candidate_score.request_penalty -
              candidate_score.waiting_penalty +
              candidate_score.boundary_long_escape_bonus_v0_3 -
              candidate_score.token_penalty_v0_3 -
              candidate_score.prefill_time_penalty_v0_3;
        }
        candidate_score.score = use_v0_3_family_scoring
                                    ? candidate_score.score_v0_3
                                    : candidate_score.score_v0_2;
        candidate_score.affinity_bonus = use_v0_3_family_scoring
                                             ? candidate_score.affinity_bonus_v0_3
                                             : candidate_score.affinity_bonus_v0_2;
      }
      if (request_is_long && FLAGS_dynamic_long_request_token_cost_enabled) {
        candidate_score.request_token_cost_multiplier_v0_2 =
            dynamic_long_request_token_cost_multiplier;
        candidate_score.token_penalty_v0_2 =
            static_cast<double>(candidate_score.current_prefill_tokens) *
            FLAGS_hybrid_prefill_token_num_weight *
            candidate_score.request_token_cost_multiplier_v0_2;
        candidate_score.score_v0_2 = candidate_score.affinity_bonus_v0_2 +
                                     candidate_score.rescue_bonus -
                                     candidate_score.long_short_pool_busy_surcharge -
                                     candidate_score.long_short_pool_affine_gap_penalty -
                                     candidate_score.long_affine_busy_penalty -
                                     candidate_score.request_penalty -
                                     candidate_score.waiting_penalty -
                                     candidate_score.token_penalty_v0_2 -
                                     candidate_score.prefill_time_penalty_v0_2;
        candidate_score.score = use_v0_3_family_scoring
                                    ? candidate_score.score_v0_3
                                    : candidate_score.score_v0_2;
        candidate_score.token_penalty =
            use_v0_3_family_scoring ? candidate_score.token_penalty_v0_3
                                    : candidate_score.token_penalty_v0_2;
      }
      candidate_summaries.emplace_back(
          format_hybrid_candidate(prefill_instance, candidate_score));
      if (route_trace_sampled) {
        route_trace_candidate_breakdown.emplace_back(
            route_trace_candidate_to_json(prefill_instance, candidate_score));
      }
      if (candidate_score.short_long_lane_guarded &&
          FLAGS_hybrid_prefill_short_long_lane_guard_hard_reject) {
        continue;
      }
      const bool better_score =
          candidate_score.score > best_candidate_score.score;
      const bool same_score =
          candidate_score.score == best_candidate_score.score;
      const bool better_combined_load =
          same_score &&
          candidate_score.combined_load < best_candidate_score.combined_load;
      const bool same_combined_load =
          same_score &&
          candidate_score.combined_load == best_candidate_score.combined_load;
      const bool better_waiting =
          same_combined_load &&
          candidate_score.waiting_requests <
              best_candidate_score.waiting_requests;
      const bool same_waiting =
          same_combined_load &&
          candidate_score.waiting_requests ==
              best_candidate_score.waiting_requests;
      const bool better_active =
          same_waiting &&
          candidate_score.active_requests < best_candidate_score.active_requests;
      const bool same_active =
          same_waiting &&
          candidate_score.active_requests == best_candidate_score.active_requests;
      const bool better_prefill_time =
          same_active &&
          candidate_score.prefill_time < best_candidate_score.prefill_time;
      const bool same_prefill_time =
          same_active &&
          candidate_score.prefill_time == best_candidate_score.prefill_time;

      if (hybrid_best_candidates.empty() || better_score ||
          better_combined_load || better_waiting || better_active ||
          better_prefill_time) {
        hybrid_best_candidates.clear();
        hybrid_best_candidates.emplace_back(prefill_instance);
        best_candidate_score = candidate_score;
      } else if (same_prefill_time) {
        hybrid_best_candidates.emplace_back(prefill_instance);
      }
    }

    if (!hybrid_best_candidates.empty()) {
      uint64_t* next_tie_break_index =
          request_is_long ? &next_static_long_prefill_tie_break_index_
                          : &next_static_short_prefill_tie_break_index_;
      const uint64_t selected_index =
          *next_tie_break_index % hybrid_best_candidates.size();
      selected_prefill_instance = hybrid_best_candidates[selected_index];
      *next_tie_break_index = selected_index + 1;

      const HybridCandidateScore selected_candidate_score =
          evaluate_hybrid_candidate(selected_prefill_instance);
      selected_prefill_score = selected_candidate_score.score;
      selected_prefill_score_v0_2 = selected_candidate_score.score_v0_2;
      selected_prefill_score_v0_3 = selected_candidate_score.score_v0_3;
      selected_affinity_bonus = selected_candidate_score.affinity_bonus;
      selected_rescue_bonus = selected_candidate_score.rescue_bonus;
      selected_boundary_long_escape_bonus =
          selected_candidate_score.boundary_long_escape_bonus_v0_3;
      selected_long_affine_busy_penalty =
          selected_candidate_score.long_affine_busy_penalty;
      selected_long_affine_busy_admission_guarded =
          selected_candidate_score.long_affine_busy_admission_guarded;
      selected_long_affinity_base_bonus =
          selected_candidate_score.long_affinity_base_bonus;
      selected_dynamic_long_affinity_bonus =
          selected_candidate_score.dynamic_long_affinity_bonus;
      selected_effective_long_affinity_bonus =
          selected_candidate_score.effective_long_affinity_bonus;
      selected_dynamic_load_gain = selected_candidate_score.dynamic_load_gain;
      selected_dynamic_time_gain = selected_candidate_score.dynamic_time_gain;
      selected_request_penalty = selected_candidate_score.request_penalty;
      selected_waiting_penalty = selected_candidate_score.waiting_penalty;
      selected_token_penalty = selected_candidate_score.token_penalty;
      selected_prefill_time_penalty =
          selected_candidate_score.prefill_time_penalty;
      selected_combined_load = selected_candidate_score.combined_load;
      selected_prefill_affine = selected_candidate_score.is_affine;
      selected_prefill_short_to_long_lane =
          selected_candidate_score.is_short_to_long_lane;
      selected_candidate_summary =
          format_hybrid_candidate(selected_prefill_instance,
                                  selected_candidate_score);
    }
    hybrid_best_candidates_for_trace = hybrid_best_candidates;
    hybrid_candidate_summaries = absl::StrJoin(candidate_summaries, ";");
  }

  if (dynamic_cp_shadow_enabled && !opposite_filtered_instances.empty() &&
      !has_opposite_pool_projected_prefill_time) {
    for (const auto& prefill_instance : opposite_filtered_instances) {
      const int64_t projected_prefill_time =
          request_metrics_[prefill_instance].estimated_prefill_time +
          get_time_predictor(prefill_instance).predict_ttft(request_tokens);
      if (!has_opposite_pool_projected_prefill_time ||
          projected_prefill_time < opposite_pool_min_projected_prefill_time) {
        opposite_pool_min_projected_prefill_time = projected_prefill_time;
        has_opposite_pool_projected_prefill_time = true;
      }
    }
  }

  if (dynamic_cp_shadow_enabled) {
    for (const auto& prefill_instance : all_prefill_instances) {
      DynamicCpShadowCandidate candidate =
          evaluate_dynamic_cp_shadow_candidate(prefill_instance);
      if (route_trace_sampled) {
        dynamic_cp_shadow_candidate_breakdown.emplace_back(
            dynamic_cp_shadow_candidate_to_json(candidate));
      }
      const bool better_cost =
          candidate.cost < dynamic_cp_shadow_selected_candidate.cost;
      const bool same_cost =
          candidate.cost == dynamic_cp_shadow_selected_candidate.cost;
      const bool better_combined_load =
          same_cost &&
          candidate.combined_load <
              dynamic_cp_shadow_selected_candidate.combined_load;
      const bool same_combined_load =
          same_cost &&
          candidate.combined_load ==
              dynamic_cp_shadow_selected_candidate.combined_load;
      const bool better_waiting =
          same_combined_load &&
          candidate.waiting_requests <
              dynamic_cp_shadow_selected_candidate.waiting_requests;
      const bool same_waiting =
          same_combined_load &&
          candidate.waiting_requests ==
              dynamic_cp_shadow_selected_candidate.waiting_requests;
      const bool better_active =
          same_waiting &&
          candidate.active_requests <
              dynamic_cp_shadow_selected_candidate.active_requests;
      if (!dynamic_cp_shadow_has_selected || better_cost ||
          better_combined_load || better_waiting || better_active) {
        dynamic_cp_shadow_selected_candidate = candidate;
        dynamic_cp_shadow_has_selected = true;
      }
    }
  }

  if (FLAGS_enable_static_prefill_instance_split &&
      !enable_affinity_routing &&
      min_prefill_candidates.size() > 1) {
    std::vector<std::string> final_tie_candidates = min_prefill_candidates;
    if (FLAGS_enable_static_prefill_load_aware_tie_break) {
      std::vector<std::string> min_token_candidates;
      int64_t min_prefill_token_num = std::numeric_limits<int64_t>::max();
      for (const auto& prefill_instance : min_prefill_candidates) {
        const int64_t prefill_token_num =
            get_prefill_token_num(request_metrics_, prefill_instance);
        if (prefill_token_num < min_prefill_token_num) {
          min_prefill_token_num = prefill_token_num;
          min_token_candidates.clear();
          min_token_candidates.emplace_back(prefill_instance);
        } else if (prefill_token_num == min_prefill_token_num) {
          min_token_candidates.emplace_back(prefill_instance);
        }
      }

      min_prefill_token_tie_count = min_token_candidates.size();
      std::vector<std::string> min_waiting_candidates;
      uint64_t min_waiting_requests_num = std::numeric_limits<uint64_t>::max();
      for (const auto& prefill_instance : min_token_candidates) {
        const uint64_t waiting_requests_num =
            get_waiting_requests_num(load_metrics_, prefill_instance);
        if (waiting_requests_num < min_waiting_requests_num) {
          min_waiting_requests_num = waiting_requests_num;
          min_waiting_candidates.clear();
          min_waiting_candidates.emplace_back(prefill_instance);
        } else if (waiting_requests_num == min_waiting_requests_num) {
          min_waiting_candidates.emplace_back(prefill_instance);
        }
      }

      min_prefill_waiting_tie_count = min_waiting_candidates.size();
      std::vector<std::string> min_request_candidates;
      int64_t min_prefill_request_num = std::numeric_limits<int64_t>::max();
      for (const auto& prefill_instance : min_waiting_candidates) {
        const int64_t prefill_request_num =
            get_prefill_request_num(request_metrics_, prefill_instance);
        if (prefill_request_num < min_prefill_request_num) {
          min_prefill_request_num = prefill_request_num;
          min_request_candidates.clear();
          min_request_candidates.emplace_back(prefill_instance);
        } else if (prefill_request_num == min_prefill_request_num) {
          min_request_candidates.emplace_back(prefill_instance);
        }
      }
      min_prefill_request_tie_count = min_request_candidates.size();
      final_tie_candidates = std::move(min_request_candidates);
    } else {
      std::vector<std::string> min_request_candidates;
      int64_t min_prefill_request_num = std::numeric_limits<int64_t>::max();
      for (const auto& prefill_instance : min_prefill_candidates) {
        const int64_t prefill_request_num =
            get_prefill_request_num(request_metrics_, prefill_instance);
        if (prefill_request_num < min_prefill_request_num) {
          min_prefill_request_num = prefill_request_num;
          min_request_candidates.clear();
          min_request_candidates.emplace_back(prefill_instance);
        } else if (prefill_request_num == min_prefill_request_num) {
          min_request_candidates.emplace_back(prefill_instance);
        }
      }

      min_prefill_request_tie_count = min_request_candidates.size();
      std::vector<std::string> min_waiting_candidates;
      uint64_t min_waiting_requests_num = std::numeric_limits<uint64_t>::max();
      for (const auto& prefill_instance : min_request_candidates) {
        const uint64_t waiting_requests_num =
            get_waiting_requests_num(load_metrics_, prefill_instance);
        if (waiting_requests_num < min_waiting_requests_num) {
          min_waiting_requests_num = waiting_requests_num;
          min_waiting_candidates.clear();
          min_waiting_candidates.emplace_back(prefill_instance);
        } else if (waiting_requests_num == min_waiting_requests_num) {
          min_waiting_candidates.emplace_back(prefill_instance);
        }
      }
      min_prefill_waiting_tie_count = min_waiting_candidates.size();
      final_tie_candidates = std::move(min_waiting_candidates);
    }

    uint64_t* next_tie_break_index =
        request_is_long ? &next_static_long_prefill_tie_break_index_
                        : &next_static_short_prefill_tie_break_index_;
    if (!final_tie_candidates.empty()) {
      const uint64_t selected_index =
          *next_tie_break_index % final_tie_candidates.size();
      selected_prefill_instance = final_tie_candidates[selected_index];
      *next_tie_break_index = selected_index + 1;
    }
  }

  if (enable_affinity_routing &&
      FLAGS_enable_hybrid_prefill_short_long_lane_guard && !request_is_long) {
    if (selected_prefill_short_to_long_lane) {
      consecutive_short_long_borrows_after_route =
          consecutive_short_long_borrow_counter.fetch_add(
              1, std::memory_order_relaxed) +
          1;
    } else {
      consecutive_short_long_borrow_counter.store(0, std::memory_order_relaxed);
      consecutive_short_long_borrows_after_route = 0;
    }
  }

  // select prefill instance
  float tpot_threshold =
      (schedulable_decode_count - 1.0f) / schedulable_decode_count;
  // When the prefill instances are already overloaded and there are other
  // instances with lower loads in the decode group, we will dispatch the
  // prefill requests to those instances to alleviate the pressure on the
  // prefill instances.
  if (min_prefill_time > FLAGS_target_ttft &&
      target_decode_instance != min_decode_instance &&
      min_estimated_tpot < FLAGS_target_tpot * tpot_threshold &&
      request_metrics_[min_decode_instance].estimated_prefill_time <
          min_prefill_time) {
    request->routing.prefill_name = min_decode_instance;
    // update estimated ttft
    auto& time_predictor = get_time_predictor(min_decode_instance);
    request->estimated_ttft =
        time_predictor.predict_ttft(request->token_ids.size());
    request_metrics_[min_decode_instance].estimated_prefill_time +=
        request->estimated_ttft;
  } else {
    request->routing.prefill_name = selected_prefill_instance;
    // update estimated ttft
    auto& time_predictor = get_time_predictor(selected_prefill_instance);
    request->estimated_ttft =
        time_predictor.predict_ttft(request->token_ids.size());
    request_metrics_[selected_prefill_instance].estimated_prefill_time +=
        request->estimated_ttft;
  }

  if (dynamic_cp_takeover_enabled && dynamic_cp_shadow_has_selected &&
      !dynamic_cp_shadow_selected_candidate.prefill_instance.empty() &&
      dynamic_cp_shadow_selected_candidate.prefill_instance !=
          request->routing.prefill_name) {
    const std::string original_prefill_instance = request->routing.prefill_name;
    request_metrics_[original_prefill_instance].estimated_prefill_time -=
        request->estimated_ttft;
    request->routing.prefill_name =
        dynamic_cp_shadow_selected_candidate.prefill_instance;
    auto& time_predictor = get_time_predictor(request->routing.prefill_name);
    request->estimated_ttft =
        time_predictor.predict_ttft(request->token_ids.size());
    request_metrics_[request->routing.prefill_name].estimated_prefill_time +=
        request->estimated_ttft;
    request->stage_timing_trace.dynamic_cp_takeover_applied_prefill = true;
  }

  if (dynamic_cp_decode_takeover_enabled && dynamic_cp_shadow_has_selected &&
      !dynamic_cp_shadow_selected_candidate.decode_instance.empty() &&
      dynamic_cp_shadow_selected_candidate.decode_instance !=
          request->routing.decode_name) {
    request->routing.decode_name =
        dynamic_cp_shadow_selected_candidate.decode_instance;
    request->stage_timing_trace.dynamic_cp_takeover_applied_decode = true;
  }

  request->stage_timing_trace.decode_admission_preferred_decode_matched =
      !request->stage_timing_trace.decode_admission_preferred_decode_instance
           .empty() &&
      request->stage_timing_trace.decode_admission_preferred_decode_instance ==
          request->routing.decode_name;
  request->stage_timing_trace.dynamic_cp_shadow_enabled =
      dynamic_cp_shadow_enabled;
  request->stage_timing_trace.dynamic_cp_takeover_enabled =
      dynamic_cp_takeover_enabled;
  request->stage_timing_trace.dynamic_cp_decode_takeover_enabled =
      dynamic_cp_decode_takeover_enabled;
  if (dynamic_cp_shadow_has_selected) {
    request->stage_timing_trace.dynamic_cp_shadow_selected_prefill_instance =
        dynamic_cp_shadow_selected_candidate.prefill_instance;
    request->stage_timing_trace.dynamic_cp_shadow_selected_decode_instance =
        dynamic_cp_shadow_selected_candidate.decode_instance;
    request->stage_timing_trace.dynamic_cp_shadow_selected_lane_type =
        dynamic_cp_shadow_selected_candidate.lane_type;
    request->stage_timing_trace.dynamic_cp_shadow_selected_cost =
        dynamic_cp_shadow_selected_candidate.cost;
    request->stage_timing_trace.dynamic_cp_shadow_selected_prefill_cost =
        dynamic_cp_shadow_selected_candidate.prefill_cost;
    request->stage_timing_trace
        .dynamic_cp_shadow_selected_decode_pressure_cost =
        dynamic_cp_shadow_selected_candidate.decode_pressure_cost;
    request->stage_timing_trace.dynamic_cp_shadow_selected_cp_benefit =
        dynamic_cp_shadow_selected_candidate.cp_benefit;
    request->stage_timing_trace.dynamic_cp_shadow_selected_interference_cost =
        dynamic_cp_shadow_selected_candidate.interference_cost;
    request->stage_timing_trace.dynamic_cp_shadow_would_change_prefill =
        dynamic_cp_shadow_selected_candidate.prefill_instance !=
        request->routing.prefill_name;
    request->stage_timing_trace.dynamic_cp_shadow_would_change_decode =
        !dynamic_cp_shadow_selected_candidate.decode_instance.empty() &&
        dynamic_cp_shadow_selected_candidate.decode_instance !=
            request->routing.decode_name;
  }

  if (!enable_affinity_routing && route_trace_sampled) {
    auto candidate_route_to_json = [&](const std::string& prefill_instance) {
      return nlohmann::json{
          {"instance", prefill_instance},
          {"active_requests",
           get_prefill_request_num(request_metrics_, prefill_instance)},
          {"waiting_requests",
           get_waiting_requests_num(load_metrics_, prefill_instance)},
          {"combined_load",
           get_prefill_combined_load(
               request_metrics_, load_metrics_, prefill_instance)},
          {"current_prefill_tokens",
           get_prefill_token_num(request_metrics_, prefill_instance)},
          {"estimated_prefill_time_ms",
           request_metrics_[prefill_instance].estimated_prefill_time}};
    };
    for (const auto& prefill_instance : candidate_prefill_instances) {
      route_trace_candidate_breakdown.emplace_back(
          candidate_route_to_json(prefill_instance));
    }
  }

  if (route_trace_sampled) {
    nlohmann::json route_trace_entry{
        {"timestamp_ms", current_time_ms()},
        {"route_path", "slo"},
        {"sample_ordinal", route_trace_sample_ordinal},
        {"service_request_id", request->service_request_id},
        {"client_request_id", get_client_request_id(request)},
        {"request_tokens", request_tokens},
        {"request_type", request_is_long ? "long" : "short"},
        {"long_request_threshold_tokens",
         FLAGS_long_request_threshold_tokens},
        {"scoring_version", FLAGS_hybrid_prefill_scoring_version},
        {"hybrid_route_decision_ordinal", hybrid_route_decision_ordinal},
        {"hybrid_route_elapsed_ms", hybrid_route_elapsed_ms},
        {"short_long_lane_guard_enabled",
         FLAGS_enable_hybrid_prefill_short_long_lane_guard},
        {"short_long_lane_guard_hard_reject",
         FLAGS_hybrid_prefill_short_long_lane_guard_hard_reject},
        {"short_long_lane_guard_window_ms",
         FLAGS_hybrid_prefill_short_long_lane_guard_window_ms},
        {"short_long_lane_guard_request_count",
         FLAGS_hybrid_prefill_short_long_lane_guard_request_count},
        {"short_long_lane_guard_max_long_lane_combined_load",
         FLAGS_hybrid_prefill_short_long_lane_guard_max_long_lane_combined_load},
        {"short_long_lane_guard_primary_min_load_threshold",
         FLAGS_hybrid_prefill_short_long_lane_guard_primary_min_load_threshold},
        {"short_long_lane_guard_max_prompt_tokens",
         FLAGS_hybrid_prefill_short_long_lane_guard_max_prompt_tokens},
        {"short_long_lane_guard_max_consecutive_borrows",
         FLAGS_hybrid_prefill_short_long_lane_guard_max_consecutive_borrows},
        {"short_long_lane_penalty",
         FLAGS_hybrid_prefill_short_long_lane_penalty},
        {"consecutive_short_long_borrows_before_route",
         consecutive_short_long_borrows_before_route},
        {"consecutive_short_long_borrows_after_route",
         consecutive_short_long_borrows_after_route},
        {"selected_prefill_short_to_long_lane",
         selected_prefill_short_to_long_lane},
        {"selected_prefill_instance",
         request->routing.prefill_name},
        {"selected_prefill_instance_by_hybrid_score",
         selected_prefill_instance},
        {"selected_prefill_instance_final",
         request->routing.prefill_name},
        {"selected_decode_instance", request->routing.decode_name},
        {"decode_admission_preferred_decode_instance",
         request->stage_timing_trace
             .decode_admission_preferred_decode_instance},
        {"decode_admission_preferred_decode_matched",
         request->stage_timing_trace
             .decode_admission_preferred_decode_matched},
        {"decode_admission_active_requests",
         request->stage_timing_trace.decode_admission_active_requests},
        {"decode_admission_waiting_requests",
         request->stage_timing_trace.decode_admission_waiting_requests},
        {"decode_admission_estimated_tpot_ms",
         request->stage_timing_trace.decode_admission_estimated_tpot_ms},
        {"decode_admission_pressure_tier",
         request->stage_timing_trace.decode_admission_pressure_tier},
        {"decode_pressure_soft_signal_only",
         FLAGS_enable_decode_pressure_soft_signal_only},
        {"decode_pressure_use_output_work",
         FLAGS_decode_pressure_admission_use_output_work},
        {"decode_pressure_route_score_enabled",
         FLAGS_enable_decode_pressure_route_score},
        {"decode_pressure_route_score_weight",
         FLAGS_decode_pressure_route_score_weight},
        {"dynamic_cp_shadow_enabled", dynamic_cp_shadow_enabled},
        {"dynamic_cp_takeover_enabled", dynamic_cp_takeover_enabled},
        {"dynamic_cp_decode_takeover_enabled",
         dynamic_cp_decode_takeover_enabled},
        {"dynamic_cp_takeover_applied_prefill",
         request->stage_timing_trace.dynamic_cp_takeover_applied_prefill},
        {"dynamic_cp_takeover_applied_decode",
         request->stage_timing_trace.dynamic_cp_takeover_applied_decode},
        {"dynamic_cp_shadow_selected_prefill_instance",
         request->stage_timing_trace
             .dynamic_cp_shadow_selected_prefill_instance},
        {"dynamic_cp_shadow_selected_decode_instance",
         request->stage_timing_trace
             .dynamic_cp_shadow_selected_decode_instance},
        {"dynamic_cp_shadow_selected_lane_type",
         request->stage_timing_trace.dynamic_cp_shadow_selected_lane_type},
        {"dynamic_cp_shadow_selected_cost",
         request->stage_timing_trace.dynamic_cp_shadow_selected_cost},
        {"dynamic_cp_shadow_selected_prefill_cost",
         request->stage_timing_trace
             .dynamic_cp_shadow_selected_prefill_cost},
        {"dynamic_cp_shadow_selected_decode_pressure_cost",
         request->stage_timing_trace
             .dynamic_cp_shadow_selected_decode_pressure_cost},
        {"dynamic_cp_shadow_selected_cp_benefit",
         request->stage_timing_trace.dynamic_cp_shadow_selected_cp_benefit},
        {"dynamic_cp_shadow_selected_interference_cost",
         request->stage_timing_trace
             .dynamic_cp_shadow_selected_interference_cost},
        {"dynamic_cp_shadow_would_change_prefill",
         request->stage_timing_trace.dynamic_cp_shadow_would_change_prefill},
        {"dynamic_cp_shadow_would_change_decode",
         request->stage_timing_trace.dynamic_cp_shadow_would_change_decode},
        {"enable_affinity_routing", enable_affinity_routing},
        {"static_prefill_split_enabled",
         FLAGS_enable_static_prefill_instance_split},
        {"selected_prefill_affine", selected_prefill_affine},
        {"final_route_used_decode_relief",
         request->routing.prefill_name != selected_prefill_instance},
        {"all_candidates", all_prefill_instances},
        {"affine_candidates", primary_filtered_instances},
        {"opposite_candidates", opposite_filtered_instances},
        {"hybrid_best_candidates", hybrid_best_candidates_for_trace},
        {"affine_candidate_count", primary_filtered_instances.size()},
        {"opposite_candidate_count", opposite_filtered_instances.size()},
        {"affine_pool_overloaded", affine_pool_overloaded},
        {"rescue_token_eligible", rescue_token_eligible},
        {"affine_pool_min_combined_load",
         affine_pool_min_combined_load},
        {"opposite_pool_min_combined_load",
         opposite_pool_min_combined_load},
        {"pool_min_combined_load_gap", pool_min_combined_load_gap},
        {"affine_pool_min_projected_prefill_time_ms",
         affine_pool_min_projected_prefill_time},
        {"opposite_pool_min_projected_prefill_time_ms",
         opposite_pool_min_projected_prefill_time},
        {"selected_combined_load", selected_combined_load},
        {"selected_score", selected_prefill_score},
        {"selected_score_v0_2", selected_prefill_score_v0_2},
        {"selected_score_v0_3", selected_prefill_score_v0_3},
        {"selected_affinity_bonus", selected_affinity_bonus},
        {"selected_rescue_bonus", selected_rescue_bonus},
        {"selected_boundary_long_escape_bonus",
         selected_boundary_long_escape_bonus},
        {"selected_long_affine_busy_penalty",
         selected_long_affine_busy_penalty},
        {"selected_long_affine_busy_admission_guarded",
         selected_long_affine_busy_admission_guarded},
        {"selected_long_affinity_base_bonus",
         selected_long_affinity_base_bonus},
        {"selected_dynamic_long_affinity_bonus",
         selected_dynamic_long_affinity_bonus},
        {"selected_effective_long_affinity_bonus",
         selected_effective_long_affinity_bonus},
        {"selected_dynamic_load_gain", selected_dynamic_load_gain},
        {"selected_dynamic_time_gain", selected_dynamic_time_gain},
        {"selected_request_penalty", selected_request_penalty},
        {"selected_waiting_penalty", selected_waiting_penalty},
        {"selected_token_penalty", selected_token_penalty},
        {"selected_prefill_time_penalty",
         selected_prefill_time_penalty},
        {"dynamic_long_request_token_cost_load_gain",
         dynamic_long_request_token_cost_load_gain},
        {"dynamic_long_request_token_cost_time_gain",
         dynamic_long_request_token_cost_time_gain},
        {"dynamic_long_request_token_cost_gap_gain",
         dynamic_long_request_token_cost_gap_gain},
        {"dynamic_long_request_token_cost_multiplier",
         dynamic_long_request_token_cost_multiplier},
        {"estimated_ttft_ms", request->estimated_ttft},
        {"selected_candidate_summary", selected_candidate_summary},
        {"pool_long_ratio_30s",
         PoolElasticStats::instance().compute_long_ratio(
             FLAGS_pool_elastic_window_s, current_time_ms())},
        {"pool_active_set", snapshot_active_prefill_instances_locked()},
        {"pool_pressure_now", compute_pool_pressure_locked()},
        {"pool_pressure_long_lane", compute_pool_pressure_lane_locked(1)},
        {"pool_pressure_short_lane", compute_pool_pressure_lane_locked(2)},
        {"pool_elastic_use_pressure_signal",
         FLAGS_pool_elastic_use_pressure_signal},
        {"pool_state_change_event",
         build_pool_state_change_event_json(
             FLAGS_pool_elastic_window_s * 1000)},
        {"candidate_breakdown", route_trace_candidate_breakdown},
        {"dynamic_cp_shadow_candidate_breakdown",
         dynamic_cp_shadow_candidate_breakdown}};
    append_route_trace_jsonl(route_trace_entry);
  }

  if (FLAGS_enable_static_prefill_instance_split) {
    LOG(INFO) << "StaticSplitRoute tokens=" << request->token_ids.size()
              << " long=" << request_is_long
              << " candidates="
              << absl::StrJoin(candidate_prefill_instances, ",")
              << " min_prefill_time=" << min_prefill_time
              << " min_prefill_tie_count=" << min_prefill_candidates.size()
              << " token_tie_count=" << min_prefill_token_tie_count
              << " request_tie_count=" << min_prefill_request_tie_count
              << " waiting_tie_count=" << min_prefill_waiting_tie_count
              << " soft_fallback=" << static_soft_fallback_triggered
              << " primary_min_prefill_request_num="
              << static_primary_min_prefill_request_num
              << " opposite_min_prefill_request_num="
              << static_opposite_min_prefill_request_num
              << " primary_min_prefill_token_num="
              << static_primary_min_prefill_token_num
              << " opposite_min_prefill_token_num="
              << static_opposite_min_prefill_token_num
              << " selected_prefill=" << request->routing.prefill_name
              << " selected_decode=" << request->routing.decode_name;
  }
  if (enable_affinity_routing) {
    LOG(INFO) << "HybridAffinityRoute tokens=" << request->token_ids.size()
              << " long=" << request_is_long
              << " scoring_version="
              << FLAGS_hybrid_prefill_scoring_version
              << " v0_3_soft_long_lower_tokens="
              << v0_3_soft_long_lower_tokens
              << " v0_3_soft_long_upper_tokens="
              << v0_3_soft_long_upper_tokens
              << " v0_3_short_scale=" << v0_3_short_request_scale
              << " v0_3_long_scale=" << v0_3_long_request_scale
              << " all_candidates=" << absl::StrJoin(all_prefill_instances, ",")
              << " affine_candidates="
              << absl::StrJoin(primary_filtered_instances, ",")
              << " opposite_candidates="
              << absl::StrJoin(opposite_filtered_instances, ",")
              << " selected_prefill=" << request->routing.prefill_name
              << " selected_decode=" << request->routing.decode_name
              << " selected_affine=" << selected_prefill_affine
              << " short_long_guard_hard_reject="
              << FLAGS_hybrid_prefill_short_long_lane_guard_hard_reject
              << " affine_candidate_count="
              << primary_filtered_instances.size()
              << " opposite_candidate_count="
              << opposite_filtered_instances.size()
              << " affine_pool_overloaded=" << affine_pool_overloaded
              << " rescue_token_eligible=" << rescue_token_eligible
              << " affine_pool_min_combined_load="
              << affine_pool_min_combined_load
              << " opposite_pool_min_combined_load="
              << opposite_pool_min_combined_load
              << " pool_min_combined_load_gap="
              << pool_min_combined_load_gap
              << " affine_pool_min_projected_prefill_time="
              << affine_pool_min_projected_prefill_time
              << " opposite_pool_min_projected_prefill_time="
              << opposite_pool_min_projected_prefill_time
              << " selected_combined_load=" << selected_combined_load
              << " score=" << selected_prefill_score
              << " score_v0_2=" << selected_prefill_score_v0_2
              << " score_v0_3=" << selected_prefill_score_v0_3
              << " affinity_bonus=" << selected_affinity_bonus
              << " long_affinity_base_bonus="
              << selected_long_affinity_base_bonus
              << " dynamic_long_affinity_bonus="
              << selected_dynamic_long_affinity_bonus
              << " effective_long_affinity_bonus="
              << selected_effective_long_affinity_bonus
              << " dynamic_load_gain=" << selected_dynamic_load_gain
              << " dynamic_time_gain=" << selected_dynamic_time_gain
              << " rescue_bonus=" << selected_rescue_bonus
              << " boundary_long_escape_bonus="
              << selected_boundary_long_escape_bonus
              << " request_penalty=" << selected_request_penalty
              << " waiting_penalty=" << selected_waiting_penalty
              << " token_penalty=" << selected_token_penalty
              << " prefill_time_penalty=" << selected_prefill_time_penalty
              << " selected_candidate=" << selected_candidate_summary
              << " candidate_breakdown=" << hybrid_candidate_summaries;
  }

  // If there are no decode instances that meet the requirements, switch a
  // prefill instance to decode if the number of instances allows. Since the
  // current disaggregated PD mode does not support prefill and decode using the
  // same instance, we only switch the instance here, without dispatching the
  // decode request to this instance.
  float ttft_threshold =
      (schedulable_prefill_count - 1.0f) / schedulable_prefill_count;
  if (target_decode_instance.empty() &&
      (avg_prefill_time < FLAGS_target_ttft * ttft_threshold ||
       schedulable_decode_count < schedulable_prefill_count)) {
    flip_prefill_to_decode(request->routing.prefill_name);
  }

  return true;
}

void InstanceMgr::flip_prefill_to_decode(std::string& instance_name) {
  if (count_schedulable_instances(instances_, prefill_index_) <= 1) {
    // Ensure there is at least one prefill instance.
    return;
  }

  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }

  // delete instance name from prefill_index_
  remove_instance_from_index(instance_name, instances_[instance_name]);

  // insert instance name to decode_index_
  instances_[instance_name].current_type = InstanceType::DECODE;
  add_instance_to_index(instance_name, instances_[instance_name]);

  LOG(INFO) << "Flip prefill to decode, instance name : " << instance_name;
}

void InstanceMgr::flip_decode_to_prefill(std::string& instance_name) {
  if (count_schedulable_instances(instances_, decode_index_) <= 1) {
    // Ensure there is at least one decode instance.
    return;
  }

  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }

  // delete instance name from decode_index_
  remove_instance_from_index(instance_name, instances_[instance_name]);

  // insert instance name to prefill_index
  instances_[instance_name].current_type = InstanceType::PREFILL;
  add_instance_to_index(instance_name, instances_[instance_name]);

  LOG(INFO) << "Flip decode to prefill, instance name : " << instance_name;
}

TimePredictor& InstanceMgr::get_time_predictor(
    const std::string& instance_name) {
  auto it = time_predictors_.find(instance_name);
  if (it == time_predictors_.end()) {
    LOG(FATAL) << "Find TimePredictor failed, instance name : "
               << instance_name;
  }
  return it->second;
}

bool InstanceMgr::call_link_instance(const std::string& target_rpc_addr,
                                     const InstanceMetaInfo& peer_info) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = options_.timeout_ms();
  options.max_retry = 3;
  if (channel.Init(target_rpc_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Fail to initialize channel for LinkInstance to "
               << target_rpc_addr;
    return false;
  }
  xllm::proto::DisaggPDService_Stub stub(&channel);
  brpc::Controller cntl;
  xllm::proto::InstanceClusterInfo req;
  req.set_instance_name(peer_info.name);
  for (auto& cluster_id : peer_info.cluster_ids) {
    req.add_cluster_ids(cluster_id);
  }
  for (auto& addr : peer_info.addrs) {
    req.add_addrs(addr);
  }
  for (auto& port : peer_info.ports) {
    req.add_ports(port);
  }
  req.set_dp_size(peer_info.dp_size);
  req.set_kv_split_size(peer_info.kv_split_size);
  xllm::proto::Status res;
  stub.LinkInstance(&cntl, &req, &res, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "LinkInstance failed, target: " << target_rpc_addr
               << ", peer: " << peer_info.name
               << ", error: " << cntl.ErrorText();
    return false;
  }
  return res.ok();
}

bool InstanceMgr::call_unlink_instance(const std::string& target_rpc_addr,
                                       const InstanceMetaInfo& peer_info) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = options_.timeout_ms();
  options.max_retry = 3;
  if (channel.Init(target_rpc_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Fail to initialize channel for UnlinkInstance to "
               << target_rpc_addr;
    return false;
  }
  xllm::proto::DisaggPDService_Stub stub(&channel);
  brpc::Controller cntl;
  xllm::proto::InstanceClusterInfo req;
  req.set_instance_name(peer_info.name);
  for (auto& cluster_id : peer_info.cluster_ids) {
    req.add_cluster_ids(cluster_id);
  }
  for (auto& addr : peer_info.addrs) {
    req.add_addrs(addr);
  }
  for (auto& port : peer_info.ports) {
    req.add_ports(port);
  }
  req.set_dp_size(peer_info.dp_size);
  req.set_kv_split_size(peer_info.kv_split_size);
  xllm::proto::Status res;
  stub.UnlinkInstance(&cntl, &req, &res, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "UnlinkInstance failed, target: " << target_rpc_addr
               << ", peer: " << peer_info.name
               << ", error: " << cntl.ErrorText();
    return false;
  }
  return res.ok();
}

bool InstanceMgr::register_instance(const std::string& name,
                                    InstanceMetaInfo& info) {
  LOG(INFO) << "Warmtrace register_instance begin name=" << name
            << ", type=" << static_cast<int>(info.type)
            << ", rpc_address=" << info.rpc_address
            << ", incarnation_id=" << info.incarnation_id
            << ", prefill_index_size=" << prefill_index_.size()
            << ", decode_index_size=" << decode_index_.size();
  info.runtime_state = InstanceRuntimeState::REGISTERING;
  info.latest_timestamp = current_time_ms();
  info.name = name;

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    if (instances_.find(name) != instances_.end() ||
        cached_channels_.find(name) != cached_channels_.end()) {
      LOG(ERROR) << "Instance is already registered, instance_name: " << name;
      return false;
    }
  }

  std::shared_ptr<brpc::Channel> channel;
  if (!init_brpc_channel(name, &channel)) {
    LOG(ERROR) << "create channel fail: " << name;
    return false;
  }

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    if (instances_.find(name) != instances_.end() ||
        cached_channels_.find(name) != cached_channels_.end()) {
      LOG(WARNING) << "Instance registered concurrently during channel init: "
                   << name;
      return false;
    }
    cached_channels_[name] = std::move(channel);
    instances_.insert_or_assign(name, info);
  }

  add_instance_resources(name, info);

  std::vector<std::pair<std::string, InstanceMetaInfo>> link_ops;
  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    if (!gather_link_operations(info, &link_ops)) {
      LOG(ERROR) << "Warmtrace gather_link_operations failed name=" << name
                 << ", type=" << static_cast<int>(info.type);
      {
        auto it = instances_.find(name);
        if (it != instances_.end()) {
          instances_.erase(it);
        }
      }
      remove_instance_resources(name);
      return false;
    }
    LOG(INFO) << "Warmtrace register_instance gathered link ops name=" << name
              << ", ops=" << link_ops.size();
  }

  LOG(INFO) << "Warmtrace register_instance run_link_operations name=" << name << ", ops=" << link_ops.size();
  if (!run_link_operations(link_ops)) {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    auto it = instances_.find(name);
    if (it != instances_.end()) {
      instances_.erase(it);
    }
    remove_instance_resources(name);
    return false;
  }

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    auto it = instances_.find(name);
    if (it == instances_.end()) {
      LOG(ERROR) << "Instance disappeared during registration, instance_name: "
                 << name;
      remove_instance_resources(name);
      return false;
    }
    it->second.runtime_state = resolve_initial_runtime_state(name, it->second);
    it->second.draining_since_ms = 0;
    it->second.deactivate_condition_since_ms = 0;
    add_instance_to_index(name, it->second);
    LOG(INFO) << "Warmtrace register_instance inserted name=" << name
              << ", type=" << static_cast<int>(it->second.type)
              << ", rpc_address=" << it->second.rpc_address
              << ", runtime_state="
              << runtime_state_name(it->second.runtime_state)
              << ", prefill_index_size=" << prefill_index_.size()
              << ", decode_index_size=" << decode_index_.size();

    if (it->second.runtime_state == InstanceRuntimeState::IDLE) {
      if (FLAGS_pool_elastic_auto_mode) {
        LOG(INFO) << "Pool elastic init: " << name
                  << " set to IDLE (auto-mode, current ACTIVE elastic count "
                  << "already meets min_active="
                  << FLAGS_pool_elastic_min_active << ")";
      } else {
        LOG(INFO) << "Pool elastic init: " << name
                  << " set to IDLE (matches elastic selector list '"
                  << FLAGS_pool_elastic_idle_default_instances << "')";
      }
    } else if (FLAGS_pool_elastic_auto_mode &&
               (it->second.type == InstanceType::PREFILL ||
                it->second.type == InstanceType::MIX)) {
      LOG(INFO) << "Pool elastic init: " << name
                << " set to ACTIVE (auto-mode, ACTIVE elastic count below "
                << "min_active=" << FLAGS_pool_elastic_min_active << ")";
    }
  }
  return true;
}

void InstanceMgr::deregister_instance(
    const std::string& name,
    const std::string& expected_incarnation_id) {
  InstanceMetaInfo info;
  std::vector<std::pair<std::string, InstanceMetaInfo>> unlink_ops;
  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    auto it = instances_.find(name);
    if (it == instances_.end()) {
      LOG(ERROR) << "Instance is not registered, instance_name: " << name;
      return;
    }

    if (!expected_incarnation_id.empty() &&
        it->second.incarnation_id != expected_incarnation_id) {
      LOG(INFO) << "Skip deregistering stale incarnation, instance_name: "
                << name
                << ", current incarnation_id: " << it->second.incarnation_id
                << ", expected incarnation_id: " << expected_incarnation_id;
      return;
    }

    info = it->second;
    clear_suspect_instance(name, info.incarnation_id);
    gather_unlink_operations(name, info, &unlink_ops);
  }

  for (const auto& op : unlink_ops) {
    call_unlink_instance(op.first, op.second);
  }

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    auto it = instances_.find(name);
    if (it == instances_.end()) {
      return;
    }
    remove_instance_from_index(name, it->second);
  }

  scheduler_->clear_requests_on_failed_instance(
      name, info.incarnation_id, get_cleanup_type(info));

  {
    std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
    auto it = instances_.find(name);
    if (it == instances_.end()) {
      return;
    }
    remove_instance_resources(name);
    instances_.erase(it);
  }
  LOG(INFO) << "delete instance: " << name;

  // P4-NEW: in auto mode, deregistering an ACTIVE elastic instance can drop
  // the live ACTIVE count below min_active. Trigger an immediate elasticity
  // tick so an IDLE replacement can be promoted right away, without waiting
  // for the next 1Hz reconcile_instance_states loop. Bypass cool-down because
  // a dropped ACTIVE is an emergency, not a normal scale-up.
  if (FLAGS_pool_elastic_auto_mode &&
      info.runtime_state == InstanceRuntimeState::ACTIVE &&
      (info.type == InstanceType::PREFILL ||
       info.type == InstanceType::MIX)) {
    {
      std::unique_lock<std::shared_mutex> lock(cluster_mutex_);
      last_pool_activation_ts_ms_ = 0;  // bypass cool-down for emergency
    }
    tick_pool_elasticity(current_time_ms());
  }
}

void InstanceMgr::add_instance_resources(const std::string& name,
                                         const InstanceMetaInfo& info) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

  time_predictors_.insert_or_assign(
      name, TimePredictor(info.ttft_profiling_data, info.tpot_profiling_data));

  request_metrics_.insert_or_assign(name, RequestMetrics());
}

void InstanceMgr::remove_instance_resources(const std::string& name) {
  // Caller must hold cluster_mutex_ (cached_channels_ is L1).
  cached_channels_.erase(name);
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
  time_predictors_.erase(name);
  request_metrics_.erase(name);
  latency_metrics_.erase(name);
  updated_metrics_.erase(name);
  removed_instance_.insert(name);
  load_metrics_.erase(name);
}

bool InstanceMgr::gather_link_operations(
    const InstanceMetaInfo& info,
    std::vector<std::pair<std::string, InstanceMetaInfo>>* out_ops) {
  out_ops->clear();
  switch (info.type) {
    case InstanceType::DEFAULT:
      break;
    case InstanceType::PREFILL: {
      for (auto& d_name : decode_index_) {
        out_ops->emplace_back(instances_[d_name].rpc_address, info);
      }
      break;
    }
    case InstanceType::DECODE: {
      for (auto& p_name : prefill_index_) {
        out_ops->emplace_back(info.rpc_address, instances_[p_name]);
      }
      break;
    }
    case InstanceType::MIX: {
      for (const auto& [peer_name, peer_info] : instances_) {
        if (peer_name == info.name) {
          continue;
        }
        out_ops->emplace_back(info.rpc_address, peer_info);
      }
      break;
    }
    default:
      LOG(WARNING) << "Unknown InstanceType: " << int(info.type);
      return false;
  }
  return true;
}

bool InstanceMgr::run_link_operations(
    const std::vector<std::pair<std::string, InstanceMetaInfo>>& ops) {
  for (size_t i = 0; i < ops.size(); ++i) {
    if (!call_link_instance(ops[i].first, ops[i].second)) {
      LOG(ERROR) << "Fail to link instance during registration, op index " << i;
      for (size_t j = 0; j < i; ++j) {
        call_unlink_instance(ops[j].first, ops[j].second);
      }
      return false;
    }
  }
  return true;
}

void InstanceMgr::gather_unlink_operations(
    const std::string& name,
    const InstanceMetaInfo& info,
    std::vector<std::pair<std::string, InstanceMetaInfo>>* out_ops) {
  out_ops->clear();
  if (info.type == InstanceType::PREFILL) {
    for (auto& d_name : decode_index_) {
      out_ops->emplace_back(instances_[d_name].rpc_address, info);
    }
  } else if (info.type == InstanceType::DECODE) {
    for (auto& p_name : prefill_index_) {
      out_ops->emplace_back(instances_[p_name].rpc_address, info);
    }
  } else if (info.type == InstanceType::MIX) {
    for (const auto& [peer_name, peer_info] : instances_) {
      if (peer_name == name) {
        continue;
      }
      out_ops->emplace_back(peer_info.rpc_address, info);
    }
  }
}

void InstanceMgr::add_instance_to_index(const std::string& name,
                                        InstanceMetaInfo& info) {
  switch (info.type) {
    case InstanceType::DEFAULT:
      info.instance_index = prefill_index_.size();
      prefill_index_.emplace_back(name);
      LOG(INFO) << "Register a new default instance, instance name : " << name;
      break;
    case InstanceType::PREFILL:
      info.instance_index = prefill_index_.size();
      prefill_index_.emplace_back(name);
      LOG(INFO) << "Register a new prefill instance, instance name : " << name;
      break;
    case InstanceType::DECODE:
      info.instance_index = decode_index_.size();
      decode_index_.emplace_back(name);
      LOG(INFO) << "Register a new decode instance, instance name : " << name;
      break;
    case InstanceType::MIX:
      if (decode_index_.size() > 0) {
        info.instance_index = prefill_index_.size();
        info.current_type = InstanceType::PREFILL;
        prefill_index_.emplace_back(name);
        LOG(INFO) << "Register a new prefill instance, instance name : "
                  << name;
      } else {
        info.instance_index = decode_index_.size();
        info.current_type = InstanceType::DECODE;
        decode_index_.emplace_back(name);
        LOG(INFO) << "Register a new decode instance, instance name : " << name;
      }
      break;
    default:
      break;
  }
}

void InstanceMgr::remove_instance_from_index(const std::string& name,
                                             const InstanceMetaInfo& info) {
  uint64_t index = info.instance_index;
  if (index == -1) return;

  auto remove_from_vec = [&](std::vector<std::string>& vec) {
    if (index >= vec.size()) return;
    std::swap(vec[index], vec.back());
    instances_[vec[index]].instance_index = index;
    vec.pop_back();
  };

  switch (info.type) {
    case InstanceType::DEFAULT:
    case InstanceType::PREFILL:
      remove_from_vec(prefill_index_);
      break;
    case InstanceType::DECODE:
      remove_from_vec(decode_index_);
      break;
    case InstanceType::MIX:
      if (info.current_type == InstanceType::PREFILL) {
        remove_from_vec(prefill_index_);
      } else {
        remove_from_vec(decode_index_);
      }
      break;
    default:
      break;
  }
}

bool InstanceMgr::has_available_instances() const {
  std::shared_lock<std::shared_mutex> lock(cluster_mutex_);

  bool has_default = false;
  bool has_prefill = false;
  bool has_decode = false;
  bool has_mix_as_prefill = false;
  bool has_mix_as_decode = false;

  for (const auto& [name, info] : instances_) {
    if (!is_instance_schedulable(info)) continue;

    switch (info.type) {
      case InstanceType::DEFAULT:
        has_default = true;
        break;
      case InstanceType::PREFILL:
        has_prefill = true;
        break;
      case InstanceType::DECODE:
        has_decode = true;
        break;
      case InstanceType::MIX:
        if (info.current_type == InstanceType::PREFILL) {
          has_mix_as_prefill = true;
        } else if (info.current_type == InstanceType::DECODE) {
          has_mix_as_decode = true;
        }
        break;
      default:
        break;
    }

    // Early exit: any satisfied condition is enough
    if (has_default || (has_prefill && has_decode) ||
        (has_mix_as_prefill && has_mix_as_decode)) {
      return true;
    }
  }

  return has_default || (has_prefill && has_decode) ||
         (has_mix_as_prefill && has_mix_as_decode);
}

}  // namespace xllm_service
