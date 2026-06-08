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

#include "common/global_gflags.h"

#include "brpc/reloadable_flags.h"

namespace {

bool ValidatePositiveDouble(const char*, double value) { return value > 0.0; }
bool ValidateNonNegativeDouble(const char*, double value) {
  return value >= 0.0;
}
bool ValidatePositiveInt32(const char*, int32_t value) { return value > 0; }

}  // namespace

DEFINE_string(server_host,
              "",
              "Server listen address, may be IPV4/IPV6/UDS."
              " If this is set, the flag port will be ignored");

DEFINE_int32(http_server_port, 8888, "Port for xllm http service to listen on");

DEFINE_int32(http_server_idle_timeout_s,
             -1,
             "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");

DEFINE_int32(http_server_num_threads, 32, "Maximum number of threads to use");

DEFINE_int32(http_server_max_concurrency,
             128,
             "Limit number of requests processed in parallel");

DEFINE_int32(rpc_server_port, 8889, "Port for xllm rpc service to listen on");

DEFINE_int32(rpc_server_idle_timeout_s,
             -1,
             "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");

DEFINE_int32(rpc_server_num_threads, 32, "Maximum number of threads to use");

DEFINE_int32(rpc_server_max_concurrency,
             128,
             "Limit number of requests processed in parallel");

DEFINE_string(etcd_addr,
              "0.0.0.0:2379",
              "etcd adderss for save instance meta info");

DEFINE_string(
    etcd_namespace,
    "",
    "Optional etcd namespace prefix for all xllm-service keys, e.g. prod-a.");

DEFINE_uint32(xxh3_128bits_seed, 1024, "default XXH3 128bits Hash seed");

DEFINE_int32(port, 8888, "Port for xllm service to listen on");

DEFINE_int32(num_threads, 32, "Number of threads to process requests");

DEFINE_int32(max_concurrency,
             128,
             "Limit number of requests processed in parallel");

DEFINE_int32(
    timeout_ms,
    -1,
    "Max duration (millisecond) of bRPC Channel. -1 means wait indefinitely.");

DEFINE_int32(connect_timeout_ms,
             -1,
             "Max duration (millisecond) of bRPC to establish connections. -1 "
             "means wait "
             "indefinitely.");

DEFINE_string(listen_addr,
              "",
              "Server listen address, may be IPV4/IPV6/UDS."
              " If this is set, the flag port will be ignored");

DEFINE_int32(idle_timeout_s,
             -1,
             "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");

DEFINE_string(load_balance_policy,
              "RR",
              "Disaggregated prefill-decode policy.");

DEFINE_bool(enable_static_prefill_instance_split,
            false,
            "Whether to route prefill requests to different prefill instance "
            "subsets based on prompt length.");

DEFINE_string(static_prefill_short_instance_selectors,
              "",
              "Comma-separated list of selectors for short-request prefill "
              "instances. Each selector can be a full instance name, "
              "a ':port' suffix, or a bare port number.");

DEFINE_string(static_prefill_long_instance_selectors,
              "",
              "Comma-separated list of selectors for long-request prefill "
              "instances. Each selector can be a full instance name, "
              "a ':port' suffix, or a bare port number.");

DEFINE_bool(enable_static_prefill_soft_fallback,
            false,
            "Whether static prefill split can temporarily include the opposite "
            "pool when the primary pool is significantly busier.");

DEFINE_int32(static_prefill_soft_fallback_request_num_gap,
             1,
             "Minimum prefill request-num gap (primary - opposite) required "
             "to trigger static prefill soft fallback.");

BRPC_VALIDATE_GFLAG(static_prefill_soft_fallback_request_num_gap,
                    brpc::NonNegativeInteger);

DEFINE_int32(static_prefill_soft_fallback_token_gap,
             2048,
             "Minimum prefill token-num gap (primary - opposite) required "
             "to trigger static prefill soft fallback.");

BRPC_VALIDATE_GFLAG(static_prefill_soft_fallback_token_gap,
                    brpc::NonNegativeInteger);

DEFINE_bool(enable_static_prefill_load_aware_tie_break,
            false,
            "Whether static prefill tie-break prioritizes lower prompt-token "
            "load before waiting/request counts.");

DEFINE_bool(enable_hybrid_prefill_affinity_routing,
            false,
            "Whether to score all schedulable prefill instances with a "
            "hybrid affinity model instead of hard static split routing.");

DEFINE_double(hybrid_prefill_long_affinity_bonus,
              100.0,
              "Score bonus applied when a long request is routed to a long/"
              "CP-affine prefill instance.");

DEFINE_double(hybrid_prefill_short_affinity_bonus,
              100.0,
              "Score bonus applied when a short request is routed to a short/"
              "non-CP-affine prefill instance.");

DEFINE_bool(
    hybrid_prefill_dynamic_long_affinity_gain_enabled,
    false,
    "Whether to dynamically raise long-request affine bonus based on runtime "
    "pool pressure and projected prefill-time competitiveness.");

DEFINE_double(
    hybrid_prefill_long_affinity_base_bonus,
    80.0,
    "Base score bonus applied when a long request is routed to a long/CP-"
    "affine prefill instance while dynamic long-affinity gain is enabled.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_long_affinity_base_bonus,
                    ValidateNonNegativeDouble);

DEFINE_double(
    hybrid_prefill_long_affinity_dynamic_bonus_max,
    10.0,
    "Maximum extra score bonus that dynamic long-affinity gain may add on "
    "top of hybrid_prefill_long_affinity_base_bonus.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_long_affinity_dynamic_bonus_max,
                    ValidateNonNegativeDouble);

DEFINE_int32(hybrid_prefill_dynamic_load_gain_window,
             2,
             "Combined-load window used by dynamic long-affinity gain. "
             "When affine pool minimum combined load rises from overload "
             "threshold to overload threshold + this window, load_gain rises "
             "from 0 to 1.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_dynamic_load_gain_window,
                    brpc::NonNegativeInteger);

DEFINE_int32(
    hybrid_prefill_dynamic_projected_prefill_time_margin_ms,
    1500,
    "Projected prefill-time slack used by dynamic long-affinity gain. "
    "If affine pool projected prefill time is no worse than opposite pool "
    "plus this margin, time_gain stays near 1; otherwise it decays toward 0.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_dynamic_projected_prefill_time_margin_ms,
                    brpc::NonNegativeInteger);

DEFINE_double(hybrid_prefill_request_num_weight,
              1.0,
              "Weight of prefill_request_num in hybrid prefill affinity "
              "routing score.");

DEFINE_double(hybrid_prefill_waiting_requests_weight,
              1.0,
              "Weight of waiting_requests_num in hybrid prefill affinity "
              "routing score.");

DEFINE_double(hybrid_prefill_token_num_weight,
              0.001,
              "Weight of prefill_token_num in hybrid prefill affinity "
              "routing score.");

DEFINE_int32(hybrid_prefill_overload_threshold,
             0,
             "Combined prefill load threshold (waiting + active requests) "
             "for enabling non-affine rescue routing.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_overload_threshold,
                    brpc::NonNegativeInteger);

DEFINE_double(hybrid_prefill_rescue_bonus,
              0.0,
              "Additional score bonus granted to non-affine instances when "
              "the affine pool is overloaded.");

DEFINE_int32(hybrid_prefill_rescue_min_tokens,
             0,
             "Minimum prompt token count eligible for non-affine rescue "
             "routing.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_rescue_min_tokens,
                    brpc::NonNegativeInteger);

DEFINE_int32(hybrid_prefill_rescue_max_tokens,
             2147483647,
             "Maximum prompt token count eligible for non-affine rescue "
             "routing.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_rescue_max_tokens,
                    brpc::NonNegativeInteger);

DEFINE_int32(hybrid_prefill_rescue_load_gap,
             0,
             "Minimum combined-load advantage that a non-affine instance "
             "must have before rescue routing is allowed.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_rescue_load_gap,
                    brpc::NonNegativeInteger);

DEFINE_bool(
    enable_hybrid_prefill_short_long_lane_guard,
    false,
    "Whether short requests are protected from borrowing long-affine prefill "
    "lanes unless the configured guard conditions allow it.");

DEFINE_bool(
    hybrid_prefill_short_long_lane_guard_hard_reject,
    true,
    "Whether short-long lane guard conditions remove long-affine candidates "
    "entirely. When false, guarded candidates remain eligible with the "
    "configured short-long lane score penalty, enabling soft admission shaping.");

DEFINE_int32(
    hybrid_prefill_short_long_lane_guard_window_ms,
    3000,
    "Startup window in milliseconds during which short requests cannot borrow "
    "long-affine prefill lanes when the short-long lane guard is enabled. 0 "
    "disables the time window.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_short_long_lane_guard_window_ms,
                    brpc::NonNegativeInteger);

DEFINE_int32(
    hybrid_prefill_short_long_lane_guard_request_count,
    16,
    "Initial hybrid route-decision count during which short requests cannot "
    "borrow long-affine prefill lanes when the short-long lane guard is "
    "enabled. 0 disables the request-count window.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_short_long_lane_guard_request_count,
                    brpc::NonNegativeInteger);

DEFINE_int32(
    hybrid_prefill_short_long_lane_guard_max_long_lane_combined_load,
    0,
    "Maximum combined load allowed on a long-affine lane before a short "
    "request is prevented from borrowing it. 0 means the long lane must be "
    "idle when the guard is enabled.");

BRPC_VALIDATE_GFLAG(
    hybrid_prefill_short_long_lane_guard_max_long_lane_combined_load,
    brpc::NonNegativeInteger);

DEFINE_int32(
    hybrid_prefill_short_long_lane_guard_primary_min_load_threshold,
    1,
    "Minimum primary short-lane combined load required before a short request "
    "may borrow a long-affine lane. This keeps borrowing disabled while at "
    "least one short lane is still clearly available. 0 disables this check.");

BRPC_VALIDATE_GFLAG(
    hybrid_prefill_short_long_lane_guard_primary_min_load_threshold,
    brpc::NonNegativeInteger);

DEFINE_int32(
    hybrid_prefill_short_long_lane_guard_max_prompt_tokens,
    0,
    "Maximum short-request prompt tokens eligible to borrow a long-affine "
    "prefill lane. 0 disables this prompt-size check.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_short_long_lane_guard_max_prompt_tokens,
                    brpc::NonNegativeInteger);

DEFINE_int32(
    hybrid_prefill_short_long_lane_guard_max_consecutive_borrows,
    0,
    "Maximum consecutive short requests allowed to borrow long-affine "
    "prefill lanes. 0 disables the consecutive-borrow cap.");

BRPC_VALIDATE_GFLAG(
    hybrid_prefill_short_long_lane_guard_max_consecutive_borrows,
    brpc::NonNegativeInteger);

DEFINE_double(
    hybrid_prefill_short_long_lane_penalty,
    0.0,
    "Additional score penalty applied when a short request considers a "
    "long-affine prefill lane under hybrid affinity routing.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_short_long_lane_penalty,
                    ValidateNonNegativeDouble);

DEFINE_bool(
    enable_hybrid_prefill_long_short_pool_busy_surcharge,
    false,
    "Whether long requests considering short-pool prefill lanes should receive "
    "an extra score penalty when the candidate short-pool lane is already "
    "high-load.");

DEFINE_int32(
    hybrid_prefill_long_short_pool_busy_combined_load_threshold,
    0,
    "Combined-load threshold above which a short-pool candidate receives the "
    "configured long-request busy surcharge. 0 means any non-idle short-pool "
    "candidate is surcharge-eligible when enabled.");

BRPC_VALIDATE_GFLAG(
    hybrid_prefill_long_short_pool_busy_combined_load_threshold,
    brpc::NonNegativeInteger);

DEFINE_double(
    hybrid_prefill_long_short_pool_busy_surcharge,
    0.0,
    "Additional score penalty applied when a long request considers a busy "
    "short-pool prefill candidate.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_long_short_pool_busy_surcharge,
                    ValidateNonNegativeDouble);

DEFINE_bool(
    enable_hybrid_prefill_long_short_pool_continuous_surcharge,
    false,
    "Whether to replace the fixed long-to-short-pool busy surcharge with a "
    "continuous alpha * max(0, combined_load - threshold) penalty. This keeps "
    "low-load short-pool escape cheap while increasingly discouraging long "
    "requests from leaking into already busy short-pool lanes.");

DEFINE_double(
    hybrid_prefill_long_short_pool_busy_surcharge_alpha,
    0.0,
    "Continuous surcharge slope used when "
    "enable_hybrid_prefill_long_short_pool_continuous_surcharge is enabled.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_long_short_pool_busy_surcharge_alpha,
                    ValidateNonNegativeDouble);

DEFINE_bool(
    enable_hybrid_prefill_long_short_pool_affine_gap_penalty,
    false,
    "Whether long requests considering short-pool lanes should receive an "
    "extra penalty when the candidate short-pool lane does not have a clear "
    "combined-load advantage over the long-affine pool.");

DEFINE_int32(
    hybrid_prefill_long_short_pool_affine_gap_threshold,
    0,
    "Minimum combined-load advantage that a short-pool candidate should have "
    "versus the affine long pool before the extra affine-gap penalty is "
    "waived. 0 means the candidate must be strictly better than the affine "
    "pool to avoid the penalty.");

BRPC_VALIDATE_GFLAG(
    hybrid_prefill_long_short_pool_affine_gap_threshold,
    brpc::NonNegativeInteger);

DEFINE_double(
    hybrid_prefill_long_short_pool_affine_gap_penalty,
    0.0,
    "Additional score penalty applied when a long request leaks into a "
    "short-pool lane without a sufficient combined-load advantage over the "
    "long-affine pool.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_long_short_pool_affine_gap_penalty,
                    ValidateNonNegativeDouble);

DEFINE_string(hybrid_prefill_scoring_version,
              "v0_2",
              "Hybrid prefill scoring architecture version. Supported values: "
              "v0_2, v0_3, v0_3a, v0_3b, v0_3c.");

DEFINE_int32(hybrid_prefill_v0_3_soft_long_lower_tokens,
             0,
             "For hybrid prefill scoring v0_3, request lengths at or below "
             "this token count use the full short prior. 0 means use "
             "long_request_threshold_tokens.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_v0_3_soft_long_lower_tokens,
                    brpc::NonNegativeInteger);

DEFINE_int32(hybrid_prefill_v0_3_soft_long_upper_tokens,
             0,
             "For hybrid prefill scoring v0_3, request lengths at or above "
             "this token count use the full long prior. 0 means use "
             "2 * long_request_threshold_tokens.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_v0_3_soft_long_upper_tokens,
                    brpc::NonNegativeInteger);

DEFINE_double(
    hybrid_prefill_v0_3b_boundary_long_projected_token_penalty_scale,
    1.0,
    "For hybrid prefill scoring v0_3b, extra multiplier applied to the "
    "projected token penalty when a request is in the boundary-long band "
    "(hard long, but still within the soft-long interpolation range).");

BRPC_VALIDATE_GFLAG(
    hybrid_prefill_v0_3b_boundary_long_projected_token_penalty_scale,
    ValidatePositiveDouble);

DEFINE_double(
    hybrid_prefill_v0_3c_boundary_long_escape_bonus,
    0.0,
    "For hybrid prefill scoring v0_3c, extra bonus granted to a non-affine "
    "candidate when a boundary-long request is allowed to escape into the "
    "short pool.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_v0_3c_boundary_long_escape_bonus,
                    ValidateNonNegativeDouble);

DEFINE_int32(
    hybrid_prefill_v0_3c_boundary_long_escape_load_gap,
    0,
    "For hybrid prefill scoring v0_3c, minimum affine-vs-opposite combined "
    "load gap required before a boundary-long request may escape to the "
    "opposite pool.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_v0_3c_boundary_long_escape_load_gap,
                    brpc::NonNegativeInteger);

DEFINE_int32(
    hybrid_prefill_v0_3c_boundary_long_escape_min_affine_load,
    0,
    "For hybrid prefill scoring v0_3c, minimum affine pool combined load "
    "required before boundary-long escape is considered.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_v0_3c_boundary_long_escape_min_affine_load,
                    brpc::NonNegativeInteger);

DEFINE_bool(
    enable_hybrid_prefill_long_affine_busy_admission,
    false,
    "Whether long-affine prefill lanes receive a score penalty when they are "
    "already busy, allowing long requests to spill to lighter non-affine lanes.");

DEFINE_int32(
    hybrid_prefill_long_affine_busy_combined_load_threshold,
    2,
    "Combined-load threshold at or above which a long-affine prefill lane is "
    "considered busy for long-request admission shaping. 0 disables it.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_long_affine_busy_combined_load_threshold,
                    brpc::NonNegativeInteger);

DEFINE_double(
    hybrid_prefill_long_affine_busy_penalty,
    0.0,
    "Additional score penalty applied to busy long-affine candidates for long "
    "requests. This is soft admission shaping, not a hard rejection.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_long_affine_busy_penalty,
                    ValidateNonNegativeDouble);

DEFINE_bool(enable_prefill_time_route_signal,
            false,
            "Whether to include projected prefill time in SLO-aware route "
            "cost calculation.");

DEFINE_double(prefill_time_route_weight,
              0.5,
              "Weight of projected prefill time in SLO-aware route cost when "
              "enable_prefill_time_route_signal is enabled.");

DEFINE_bool(
    enable_lane_specific_projected_prefill_time_route_signal,
    false,
    "Whether to use lane-specific projected prefill-time weights under hybrid "
    "affinity routing.");

DEFINE_double(
    hybrid_prefill_long_lane_projected_prefill_time_weight,
    0.0,
    "Projected prefill-time weight applied to long-affine candidates when "
    "lane-specific projected prefill-time routing is enabled.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_long_lane_projected_prefill_time_weight,
                    ValidateNonNegativeDouble);

DEFINE_double(
    hybrid_prefill_short_pool_projected_prefill_time_weight,
    0.0,
    "Projected prefill-time weight applied to short-pool candidates when "
    "lane-specific projected prefill-time routing is enabled.");

BRPC_VALIDATE_GFLAG(hybrid_prefill_short_pool_projected_prefill_time_weight,
                    ValidateNonNegativeDouble);

DEFINE_bool(
    enable_lane_normalized_projected_prefill_time_route_signal,
    false,
    "Whether to add a lane-class-normalized projected prefill-time score "
    "penalty under hybrid affinity routing. This should be used as a "
    "tie-breaker scale, not as a replacement for lane-class affinity.");

DEFINE_double(
    hybrid_prefill_long_lane_projected_prefill_time_baseline_ms,
    1.0,
    "Baseline projected prefill time for long-affine lanes used by normalized "
    "projected-prefill scoring.");

BRPC_VALIDATE_GFLAG(
    hybrid_prefill_long_lane_projected_prefill_time_baseline_ms,
    ValidatePositiveDouble);

DEFINE_double(
    hybrid_prefill_short_pool_projected_prefill_time_baseline_ms,
    1.0,
    "Baseline projected prefill time for short-pool lanes used by normalized "
    "projected-prefill scoring.");

BRPC_VALIDATE_GFLAG(
    hybrid_prefill_short_pool_projected_prefill_time_baseline_ms,
    ValidatePositiveDouble);

DEFINE_double(
    hybrid_prefill_projected_prefill_time_normalized_weight,
    0.0,
    "Score penalty weight applied to lane-normalized projected prefill time.");

BRPC_VALIDATE_GFLAG(
    hybrid_prefill_projected_prefill_time_normalized_weight,
    ValidateNonNegativeDouble);

DEFINE_double(
    hybrid_prefill_projected_prefill_time_normalized_cap,
    0.0,
    "Optional cap for lane-normalized projected-prefill penalty input. 0 means "
    "uncapped.");

BRPC_VALIDATE_GFLAG(
    hybrid_prefill_projected_prefill_time_normalized_cap,
    ValidateNonNegativeDouble);

DEFINE_double(long_request_token_cost_multiplier,
              1.0,
              "Multiplier applied to long-request prompt token cost during "
              "SLO-aware prefill routing.");

DEFINE_bool(dynamic_long_request_token_cost_enabled,
            false,
            "Whether to dynamically raise long-request token-cost multiplier "
            "based on affine-pool pressure.");

DEFINE_double(
    dynamic_long_request_token_cost_extra_max,
    0.75,
    "Maximum extra multiplier added on top of "
    "long_request_token_cost_multiplier when dynamic token-cost scaling is "
    "enabled.");

BRPC_VALIDATE_GFLAG(dynamic_long_request_token_cost_extra_max,
                    ValidateNonNegativeDouble);

DEFINE_int32(dynamic_long_request_token_cost_overload_threshold,
             2,
             "Combined-load threshold at which dynamic long-request token-cost "
             "scaling begins.");

BRPC_VALIDATE_GFLAG(dynamic_long_request_token_cost_overload_threshold,
                    brpc::NonNegativeInteger);

DEFINE_int32(dynamic_long_request_token_cost_load_gain_window,
             2,
             "Combined-load window used by dynamic long-request token-cost "
             "scaling. Once affine-pool minimum combined load rises from "
             "overload_threshold to overload_threshold + this window, the "
             "extra multiplier rises from 0 to "
             "dynamic_long_request_token_cost_extra_max.");

BRPC_VALIDATE_GFLAG(dynamic_long_request_token_cost_load_gain_window,
                    brpc::NonNegativeInteger);

DEFINE_int32(detect_disconnected_instance_interval,
             15,
             "The interval that server detect the disconnected instance.");

DEFINE_int32(instance_delete_probe_timeout_ms,
             1000,
             "Timeout in milliseconds for the initial health probe after an "
             "instance lease delete event.");

DEFINE_int32(instance_delete_probe_attempts,
             2,
             "The total number of health probe attempts after an instance "
             "lease delete event.");

DEFINE_int32(lease_lost_heartbeat_timeout_ms,
             3000,
             "Heartbeat silence timeout in milliseconds before a "
             "LEASE_LOST instance enters SUSPECT.");

DEFINE_int32(block_size,
             128,
             "Number of slots per kv cache block. Default is 128.");

DEFINE_string(tokenizer_path, "", "tokenizer config path.");

DEFINE_bool(enable_request_trace, false, "Whether to enable request trace");

DEFINE_bool(enable_route_trace,
            false,
            "Whether to emit sampled route-decision trace jsonl records.");

DEFINE_string(route_trace_log_path,
              "trace/route_trace.jsonl",
              "Path of the sampled route trace jsonl output.");

DEFINE_int32(route_trace_sample_every_n,
             20,
             "Sample one eligible request out of every N route decisions.");

BRPC_VALIDATE_GFLAG(route_trace_sample_every_n, ValidatePositiveInt32);

DEFINE_bool(route_trace_long_requests_only,
            true,
            "Whether route trace sampling only records long requests.");

DEFINE_bool(enable_dynamic_cp_pricing_shadow,
            false,
            "Whether to compute and trace dynamic CP pricing decisions without "
            "changing the selected prefill/decode route.");

DEFINE_bool(enable_dynamic_cp_pricing_takeover,
            false,
            "Whether dynamic CP pricing should take over prefill routing. "
            "Requires enable_dynamic_cp_pricing_shadow to compute candidates.");

DEFINE_bool(enable_dynamic_cp_pricing_decode_takeover,
            false,
            "Whether dynamic CP pricing should also take over decode routing "
            "using the admission preferred decode hint when available.");

DEFINE_double(dynamic_cp_pricing_prefill_time_weight,
              1.0,
              "Weight of projected prefill time in dynamic CP pricing shadow.");

BRPC_VALIDATE_GFLAG(dynamic_cp_pricing_prefill_time_weight,
                    ValidateNonNegativeDouble);

DEFINE_double(dynamic_cp_pricing_token_weight,
              0.001,
              "Weight of projected prefill tokens in dynamic CP pricing shadow.");

BRPC_VALIDATE_GFLAG(dynamic_cp_pricing_token_weight,
                    ValidateNonNegativeDouble);

DEFINE_double(dynamic_cp_pricing_active_weight,
              1.0,
              "Weight of active request count in dynamic CP pricing shadow.");

BRPC_VALIDATE_GFLAG(dynamic_cp_pricing_active_weight,
                    ValidateNonNegativeDouble);

DEFINE_double(dynamic_cp_pricing_waiting_weight,
              4.0,
              "Weight of waiting request count in dynamic CP pricing shadow.");

BRPC_VALIDATE_GFLAG(dynamic_cp_pricing_waiting_weight,
                    ValidateNonNegativeDouble);

DEFINE_double(dynamic_cp_pricing_decode_tpot_weight,
              1.0,
              "Weight of decode TPOT overflow in dynamic CP pricing shadow.");

BRPC_VALIDATE_GFLAG(dynamic_cp_pricing_decode_tpot_weight,
                    ValidateNonNegativeDouble);

DEFINE_double(dynamic_cp_pricing_cp_benefit_weight,
              1.0,
              "Weight of long-lane CP benefit credit in dynamic CP pricing "
              "shadow.");

BRPC_VALIDATE_GFLAG(dynamic_cp_pricing_cp_benefit_weight,
                    ValidateNonNegativeDouble);

DEFINE_double(dynamic_cp_pricing_interference_weight,
              1.0,
              "Weight of cross-lane interference surcharge in dynamic CP "
              "pricing shadow.");

BRPC_VALIDATE_GFLAG(dynamic_cp_pricing_interference_weight,
                    ValidateNonNegativeDouble);

DEFINE_bool(enable_stage_timing_trace,
            false,
            "Whether to emit request lifecycle stage timing trace jsonl.");

DEFINE_string(stage_timing_trace_log_path,
              "trace/stage_trace.jsonl",
              "Path of the stage timing trace jsonl output.");

DEFINE_int32(long_request_threshold_tokens,
             4096,
             "Prompt token threshold used to classify a request as long.");

DEFINE_bool(enable_decode_pressure_admission_control,
            false,
            "Whether xllm_service should delay request scheduling when the "
            "decode pool is already near the configured pressure threshold.");

DEFINE_bool(enable_decode_pressure_soft_signal_only,
            false,
            "Whether decode-pressure admission should only record the preferred "
            "decode lane and pressure tier without blocking scheduling.");

DEFINE_int32(decode_pressure_admission_max_active_requests,
             0,
             "Maximum active decode requests allowed before admission waits. "
             "A value of 0 disables this active-request cap.");

BRPC_VALIDATE_GFLAG(decode_pressure_admission_max_active_requests,
                    brpc::NonNegativeInteger);

DEFINE_bool(decode_pressure_admission_use_estimated_tpot,
            false,
            "Whether decode admission should also wait when the decode TPOT "
            "predictor estimates the next request would exceed target.");

DEFINE_int32(decode_pressure_admission_target_tpot_ms,
             0,
             "TPOT target used by decode admission when estimated-TPOT gating "
             "is enabled. A value of 0 reuses --target_tpot.");

BRPC_VALIDATE_GFLAG(decode_pressure_admission_target_tpot_ms,
                    brpc::NonNegativeInteger);

DEFINE_bool(decode_pressure_admission_use_output_work,
            false,
            "Whether decode pressure prediction should include expected output "
            "tokens instead of only prompt tokens.");

DEFINE_int32(decode_pressure_admission_default_short_output_tokens,
             0,
             "Default expected output tokens for short requests when the client "
             "does not provide max_tokens and output-work admission is enabled.");

BRPC_VALIDATE_GFLAG(decode_pressure_admission_default_short_output_tokens,
                    brpc::NonNegativeInteger);

DEFINE_int32(decode_pressure_admission_default_long_output_tokens,
             0,
             "Default expected output tokens for long requests when the client "
             "does not provide max_tokens and output-work admission is enabled.");

BRPC_VALIDATE_GFLAG(decode_pressure_admission_default_long_output_tokens,
                    brpc::NonNegativeInteger);

DEFINE_int32(decode_pressure_admission_long_rescue_wait_ms,
             0,
             "Maximum soft-signal wait credit for long requests used by decode "
             "route-score aging. A value of 0 disables the credit.");

BRPC_VALIDATE_GFLAG(decode_pressure_admission_long_rescue_wait_ms,
                    brpc::NonNegativeInteger);

DEFINE_int32(decode_pressure_admission_medium_waiting_requests,
             1,
             "When decode waiting requests reach this level, long requests "
             "begin to yield under decode-pressure admission.");

BRPC_VALIDATE_GFLAG(decode_pressure_admission_medium_waiting_requests,
                    brpc::NonNegativeInteger);

DEFINE_int32(decode_pressure_admission_hard_waiting_requests,
             3,
             "When decode waiting requests reach this level, all requests "
             "must wait under decode-pressure admission.");

BRPC_VALIDATE_GFLAG(decode_pressure_admission_hard_waiting_requests,
                    brpc::NonNegativeInteger);

DEFINE_int32(decode_pressure_admission_short_active_burst,
             2,
             "Extra decode active-request headroom short requests may use "
             "under medium pressure before decode admission blocks them.");

BRPC_VALIDATE_GFLAG(decode_pressure_admission_short_active_burst,
                    brpc::NonNegativeInteger);

DEFINE_int32(decode_pressure_admission_hard_active_burst,
             4,
             "Extra decode active-request headroom allowed above the base "
             "active cap before decode admission blocks all requests.");

BRPC_VALIDATE_GFLAG(decode_pressure_admission_hard_active_burst,
                    brpc::NonNegativeInteger);

DEFINE_double(decode_pressure_admission_soft_tpot_ratio,
              1.15,
              "Soft TPOT pressure ratio for decode admission. Long requests "
              "yield when predicted TPOT exceeds target * this ratio.");

BRPC_VALIDATE_GFLAG(decode_pressure_admission_soft_tpot_ratio,
                    ValidatePositiveDouble);

DEFINE_double(decode_pressure_admission_hard_tpot_ratio,
              1.35,
              "Hard TPOT pressure ratio for decode admission. All requests "
              "wait when predicted TPOT exceeds target * this ratio.");

BRPC_VALIDATE_GFLAG(decode_pressure_admission_hard_tpot_ratio,
                    ValidatePositiveDouble);

DEFINE_int32(decode_pressure_admission_check_interval_ms,
             20,
             "Sleep interval between decode admission pressure checks.");

BRPC_VALIDATE_GFLAG(decode_pressure_admission_check_interval_ms,
                    ValidatePositiveInt32);

DEFINE_int32(decode_pressure_admission_timeout_ms,
             600000,
             "Maximum time a request may wait for decode-pressure admission "
             "before scheduling fails.");

BRPC_VALIDATE_GFLAG(decode_pressure_admission_timeout_ms,
                    ValidatePositiveInt32);

DEFINE_int32(decode_pressure_admission_log_every_n,
             100,
             "Log every N blocked decode-pressure admission checks.");

BRPC_VALIDATE_GFLAG(decode_pressure_admission_log_every_n,
                    ValidatePositiveInt32);

DEFINE_bool(enable_decode_pressure_route_score,
            false,
            "Whether prefill routing should add a soft decode-pressure cost "
            "based on the decode lane selected by admission.");

DEFINE_double(decode_pressure_route_score_weight,
              0.0,
              "Global multiplier for decode-pressure route cost.");

BRPC_VALIDATE_GFLAG(decode_pressure_route_score_weight,
                    ValidateNonNegativeDouble);

DEFINE_double(decode_pressure_route_active_weight,
              1.0,
              "Weight of active decode requests in decode-pressure route cost.");

BRPC_VALIDATE_GFLAG(decode_pressure_route_active_weight,
                    ValidateNonNegativeDouble);

DEFINE_double(decode_pressure_route_waiting_weight,
              4.0,
              "Weight of waiting decode requests in decode-pressure route cost.");

BRPC_VALIDATE_GFLAG(decode_pressure_route_waiting_weight,
                    ValidateNonNegativeDouble);

DEFINE_double(decode_pressure_route_estimated_tpot_weight,
              0.0,
              "Weight of estimated TPOT overflow in decode-pressure route cost.");

BRPC_VALIDATE_GFLAG(decode_pressure_route_estimated_tpot_weight,
                    ValidateNonNegativeDouble);

DEFINE_int32(decode_pressure_route_prefill_delay_decay_ms,
             0,
             "Decay window that weakens decode-pressure route cost as projected "
             "prefill delay grows. A value of 0 disables decay.");

BRPC_VALIDATE_GFLAG(decode_pressure_route_prefill_delay_decay_ms,
                    brpc::NonNegativeInteger);

DEFINE_double(decode_pressure_route_long_aging_credit_weight,
              0.0,
              "Credit weight that offsets decode-pressure route cost for long "
              "requests that have waited in decode admission.");

BRPC_VALIDATE_GFLAG(decode_pressure_route_long_aging_credit_weight,
                    ValidateNonNegativeDouble);

DEFINE_double(decode_pressure_route_long_aging_credit_cap,
              0.0,
              "Maximum long-request aging credit applied to decode-pressure "
              "route cost.");

BRPC_VALIDATE_GFLAG(decode_pressure_route_long_aging_credit_cap,
                    ValidateNonNegativeDouble);

DEFINE_int32(target_ttft,
             1000,
             "Target Time to First Token (TTFT), in milliseconds.");

BRPC_VALIDATE_GFLAG(target_ttft, brpc::NonNegativeInteger);

DEFINE_int32(target_tpot,
             50,
             "Target Time Per Output Token (TPOT), in milliseconds.");

BRPC_VALIDATE_GFLAG(target_tpot, brpc::NonNegativeInteger);

DEFINE_string(reasoning_parser,
              "",
              "Specify the reasoning parser for handling reasoning "
              "interactions(e.g. auto, glm45, glm47, qwen3, deepseek-r1).");

DEFINE_string(tool_call_parser,
              "",
              "Specify the parser for handling tool-call interactions(e.g. "
              "auto, qwen25, qwen3, kimi_k2, deepseekv3, glm45, glm47).");

DEFINE_int32(readiness_check_interval_s,
             3,
             "Interval in seconds to check for available instance groups "
             "before starting and during runtime of the HTTP service.");

BRPC_VALIDATE_GFLAG(readiness_check_interval_s, brpc::PositiveInteger);
