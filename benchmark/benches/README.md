We use criterion with flamegraph profiler. [Here](https://www.worthe-it.co.za/blog/2021-06-19-rust-performance-optimization-tools.html) is a useful resource.

Usage:

```bash
cargo bench --bench single_rw -- --profile-time 60 main_trace_gen
cargo bench --bench single_rw -- --profile-time 60 perm_trace_gen
cargo bench --bench single_rw -- --profile-time 60 calc_quot_values
```

You can find flamegraph output in `../target/criterion/**/**/profile/flamegraph.svg`.
