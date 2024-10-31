# Benchmarks

## Metric Labels

On a line like

```rust
info_span!("Fibonacci Program", group = "fibonacci_program").in_scope(|| {
```

The `"Fibonacci Program"` is the label for tracing logs, only for display purposes. The `group = "fibonacci_program"` adds a label `group -> "fibonacci_program"` to any metrics within the span.

Different labels can be added to provide more granularity on the metrics, but the `group` label should always be the top level label used to distinguish different proof workloads.
