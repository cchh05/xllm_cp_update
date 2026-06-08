# Historical-Best + Softwork Decode Pressure

This branch keeps the historical-best prefill policy as the main routing
shape, and adds decode-pressure softwork signals as a conservative overlay.

## Mainline Behavior

- Prefill routing still enumerates all schedulable prefill instances registered
  in etcd. The scoring loop is not hardcoded to a 3P topology.
- Decode admission enumerates all schedulable decode instances registered in
  etcd and records the preferred decode lane in stage trace.
- Decode-pressure soft-signal mode records the pressure tier and preferred
  decode instance without blocking scheduling.
- Output-work admission uses expected decode output tokens for pressure
  prediction. If `max_tokens` is provided by the client, it is used directly;
  otherwise the short/long defaults are selected by request type.
- Decode-pressure route score is an additive cost on top of the prefill score.
  It is disabled by default and only affects routing when explicitly enabled.

## 4P / 5P Adaptation Rules

The core scorer can automatically include P4/P5 once the instances are started
and registered as schedulable prefill instances. The policy does not require a
code change for a larger prefill pool.

The topology and lane-role configuration are still explicit:

- Add the P4/P5 start wrappers or env entries so the extra prefill instances
  actually register in etcd.
- Add their ports or instance-name selectors to
  `STATIC_PREFILL_SHORT_INSTANCE_SELECTORS` or
  `STATIC_PREFILL_LONG_INSTANCE_SELECTORS` according to their lane role.
- Keep short-only lanes out of the long selector if the goal is to protect
  short decode latency.
- Put extra CP/long-capable lanes into the long selector if they should absorb
  boundary-long or very-long requests.

If a new prefill instance is registered but omitted from both selectors, the
core scorer may still see it as a schedulable candidate, but its lane affinity
will be ambiguous. That is acceptable for shared lanes, but unsafe for a strict
short/long split.

## Push Checklist

Before pushing this branch:

- Rebuild `build_softwork/xllm_service/xllm_master_serving`.
- Confirm `--help` contains the softwork flags:
  `enable_decode_pressure_soft_signal_only`,
  `decode_pressure_admission_use_output_work`,
  `enable_decode_pressure_route_score`, and the related route-score weights.
- Review `git diff --cached` and make sure only strategy/source/docs files are
  staged. Do not stage local `.bak` files or generated benchmark artifacts.
- Use a real remote repository for push; the current lab `origin` may point to a
  local backup path.
