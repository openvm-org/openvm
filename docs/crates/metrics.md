# Metrics

We use the [`metrics`](https://docs.rs/metrics/latest/metrics/) crate to collect metrics for the STARK prover. We refer to [reth docs](https://github.com/paradigmxyz/reth/blob/main/docs/design/metrics.md) for more guidelines on how to use metrics.

Metrics will only be collected if the `metrics` feature is enabled.
We describe the metrics that are collected for a single VM circuit proof, which corresponds to a single execution segment.

To scope metrics from different proofs, we use the [`metrics_tracing_context`](https://docs.rs/metrics-tracing-context/latest/metrics_tracing_context/) crate to provide context-dependent labels. With the exception of the `segment` label, all other labels must be set by the caller.

For a segment proof, the following metrics are collected:

- `execute_metered_time_ms` (gauge): The metered execution time of the segment in milliseconds. This is timed across **all** segments in the group.
- `execute_preflight_time_ms` (gauge): The preflight execution time of the segment in milliseconds.
  - If this is a segment in a VM with continuations enabled, a `segment: segment_idx` label is added to the metric.
  - `memory_finalize_time_ms` (gauge): The time at the end of preflight execution spent on memory finalization.
- `trace_gen_time_ms` (gauge): The time to generate non-cached trace matrices from execution records.
  - If this is a segment in a VM with continuations enabled, a `segment: segment_idx` label is added to the metric.
- All metrics collected by [`openvm-stark-backend`](https://github.com/openvm-org/stark-backend/blob/main/docs/metrics.md), in particular `stark_prove_excluding_trace_time_ms` (gauge).
- The `total_proof_time_ms` of the proof is instrumented directly when possible. Otherwise, it is calculated as:
  - The sum `execute_preflight_time_ms + trace_gen_time_ms + stark_prove_excluding_trace_time_ms`. The `execute_metered_time_ms` is excluded for app proofs because it is not run on a per-segment basis.
- `insns` (counter): The total number of instructions executed in the segment.
- `main_cells_used` (counter): The total number of main trace cells used by all chips in the segment. This does not include cells needed to pad rows to power-of-two matrix heights. Only main trace cells, not preprocessed or permutation trace cells, are counted.
- `total_cells_used` (counter): The total number of preprocessed, main, and permutation trace cells used by all chips in the segment. This does not include cells needed to pad rows to power-of-two matrix heights.

## Scoping

As mentioned above, different proofs must be scoped for metrics post-processing. We currently use labels which are added within a scoped span using the [`metrics_tracing_context`](https://docs.rs/metrics-tracing-context/latest/metrics_tracing_context/) crate. To make post-processing easier, we have the following conventions:

- The `group` label should be the top level scope for all proofs which can be proven in parallel in an aggregation tree.

The `openvm-sdk` crate applies the following additional labeling conventions:

- For App proofs, the `group` label is set to `app_proof` or the `program_name: String` set in the `AppProver`.
  - App proofs are distinguished by the `segment` label, which is set to the segment index.
- The leaf aggregation layer has `group = leaf`.
  - Leaf proofs (each without continuations) are distinguished by the `idx` label, which is set to the leaf node index.
- The internal aggregation layers have `group = internal.{hgt}` where `hgt` is the height within the aggregation tree (`hgt = 0` is the furthest from the root).
  - Internal proofs (each without continuations) are distinguished by the `idx` label, which is set to the internal node index. The internal node index is not reset across internal layers, but it is separate from the leaf node index.
- The root aggregation layer has `group = root`.
  - There is only a single root proof, but we add `idx = 0` for uniformity.
- The STARK-to-SNARK outer aggregation proof has `group = halo2_outer`.
  - The halo2 metrics are different. Only `total_proof_time_ms` (gauge) and `main_cells_used` (counter) are collected, where `main_cells_used` is the trace cells from advice columns and constants, excluding lookup table fixed cells, and virtual columns from permutation or lookup arguments.
- The final SNARK-to-SNARK wrapper proof has `group = halo2_wrapper`.
  - The only metric collected is `total_proof_time_ms` (gauge).
