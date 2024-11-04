# Benchmarks

## Metric Labels

On a line like

```rust
info_span!("Fibonacci Program", group = "fibonacci_program").in_scope(|| {
```

The `"Fibonacci Program"` is the label for tracing logs, only for display purposes. The `group = "fibonacci_program"` adds a label `group -> "fibonacci_program"` to any metrics within the span.

Different labels can be added to provide more granularity on the metrics, but the `group` label should always be the top level label used to distinguish different proof workloads.

## Criterion Benchmarks

Most benchmarks are binaries that run once since proving benchmarks take longer. For smaller benchmarks, such as to benchmark VM runtime, we use Criterion. These are in the `benches` directory.

### Usage

```bash
cargo bench --bench fibonacci_execute
```

will run the normal criterion benchmark.

```bash
cargo bench --bench fibonacci_execute -- --profile-time=30
```

will generate a flamegraph report without running any criterion analysis.

Flamegraph reports can be found in `target/criterion/fibonacci/execute/profile/flamegraph.svg` of the repo root directory.
