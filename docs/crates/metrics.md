# Metrics

We use the [`metrics`](https://docs.rs/metrics/latest/metrics/) crate to collect metrics for the STARK prover. We refer to [reth docs](https://github.com/paradigmxyz/reth/blob/main/docs/design/metrics.md) for more guidelines on how to use metrics.

Metrics will only be collected if the `bench-metrics` feature is enabled.
We describe the metrics that are collected for a single VM circuit proof, which corresponds to a single execution segment.

To scope metrics from different proofs, we use the [`metrics_tracing_context`](https://docs.rs/metrics-tracing-context/latest/metrics_tracing_context/) crate to provide context-dependent labels. With the exception of the `segment` label, all other labels must be set by the caller.

For a single segment proof, the following metrics are collected:

- `execute_time_ms` (gauge): The runtime execution time of the segment in milliseconds.
  - If this is a segment in a VM with continuations enabled, a `segment: segment_idx` label is added to the metric.
- `trace_gen_time_ms` (gauge): The time to generate non-cached trace matrices from execution records.
  - If this is a segment in a VM with continuations enabled, a `segment: segment_idx` label is added to the metric.
- All metrics collected by [`openvm-stark-backend`](https://github.com/openvm-org/stark-backend/blob/main/docs/metrics.md), in particular `stark_prove_excluding_trace_time_ms` (gauge).
  - The total proving time of the proof is the sum of `execute_time_ms + trace_gen_time_ms + stark_prove_excluding_trace_time_ms`.
- `total_cycles` (counter): The total number of cycles in the segment.
- `total_cells_used` (counter): The total number of main trace cells used by all chips in the segment. This does not include cells needed to pad rows to power-of-two matrix heights. Only main trace cells, not preprocessed or permutation trace cells, are counted.
