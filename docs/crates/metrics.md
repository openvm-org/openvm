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
- `compile_pure_time_ms`, `compile_metered_time_ms`, `compile_metered_segment_time_ms`, `compile_metered_cost_time_ms`, `compile_preflight_time_ms` (gauge): Time to build an execution instance in milliseconds. The metric name identifies the execution mode, and the `backend` label identifies the backend.
- `prepare_preflight_time_ms` (gauge): One-time preparation of a fixed-program compiled preflight prover, including its metered and preflight executors and immutable GPU program.
- `upload_preflight_program_time_ms` (gauge): The immutable GPU replay-program upload within preflight preparation.
- `app_prove_time_ms` (gauge): Reusable app proving time. It excludes generated-code compilation and immutable program upload.
- `postflight_time_ms` (gauge): GPU replay and read-only index construction for one segment. Its subphases are `postflight_replay_count_time_ms`, `postflight_replay_emit_time_ms`, `postflight_memory_chronology_time_ms`, and `postflight_program_index_time_ms`.
- `trace_gen_time_ms` (gauge): The time to generate non-cached trace matrices
  from the read-only preflight data prepared by postflight.
  - If this is a segment in a VM with continuations enabled, a `segment: segment_idx` label is added to the metric.
- All metrics collected by [`openvm-stark-backend`](https://github.com/openvm-org/stark-backend/blob/main/docs/metrics.md), in particular `stark_prove_excluding_trace_time_ms` (gauge).
- The `total_proof_time_ms` of the proof is instrumented directly when possible. Otherwise, it is calculated as:
  - The sum `execute_preflight_time_ms + trace_gen_time_ms + stark_prove_excluding_trace_time_ms`. The `execute_metered_time_ms` is excluded for app proofs because it is not run on a per-segment basis.
- `execute_pure_insns` (counter): The total number of instructions executed in pure execution mode.
- `execute_metered_insns` (counter): The total number of instructions executed in metered execution mode.
- `execute_preflight_insns` (counter): The number of instructions executed by
  preflight in each segment of a reusable app proof. It carries the `segment` label and
  is emitted only after the whole proof succeeds; summing the series gives the proof-level total.
- `execute_preflight_intervals`, `execute_preflight_residuals`, and
  `execute_preflight_transcript_bytes` (counters): The compact authoritative transcript
  size across all segments in one reusable app proof. Transcript bytes measure the logical
  initialized checkpoint and residual payload, not vector capacity or allocator overhead. These
  are emitted once after a successful proof rather than once per segment.
- `main_cells_used` (counter): The total number of main trace cells used by all chips in the segment. This does not include cells needed to pad rows to power-of-two matrix heights. Only main trace cells, not preprocessed or permutation trace cells, are counted.
- `total_cells_used` (counter): The total number of preprocessed, main, and permutation trace cells used by all chips in the segment. This does not include cells needed to pad rows to power-of-two matrix heights.

## Scoping

As mentioned above, different proofs must be scoped for metrics post-processing. We currently use labels which are added within a scoped span using the [`metrics_tracing_context`](https://docs.rs/metrics-tracing-context/latest/metrics_tracing_context/) crate. To make post-processing easier, we have the following conventions:

- The `group` label should be the top level scope for all proofs which can be proven in parallel in an aggregation tree.

The `openvm-sdk` crate applies the following additional labeling conventions:

- App proofs always use `group = app_proof`. `program_name` identifies the
  program to the application, but does not create a separate proof group.
  - App proofs are distinguished by the `segment` label, which is set to the segment index.
- The leaf aggregation layer has `group = leaf`.
  - Leaf proofs (each without continuations) are distinguished by the `idx` label, which is set to the leaf node index.
- The first internal aggregation layer has `group = internal_for_leaf`. Subsequent internal recursive layers have `group = internal_recursive.{i}` where `i` starts at `0` and increments for each recursive layer.
  - Internal proofs (each without continuations) are distinguished by the `idx` label, which is set to the internal node index. The internal node index is not reset across internal layers, but it is separate from the leaf node index.
- The root aggregation layer has `group = root`.
- The STARK-to-SNARK outer aggregation proof has `group = halo2_outer`.
  - The halo2 metrics are different. Only `total_proof_time_ms` (gauge) and `main_cells_used` (counter) are collected, where `main_cells_used` is the trace cells from advice columns and constants, excluding lookup table fixed cells, and virtual columns from permutation or lookup arguments.
- The final SNARK-to-SNARK wrapper proof has `group = halo2_wrapper`.
  - The only metric collected is `total_proof_time_ms` (gauge).
