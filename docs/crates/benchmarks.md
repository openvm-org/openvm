# Benchmarks

To run benchmarks, install python3 and run (from root of repo):

```bash
python ci/scripts/bench.py <name> --instance_type <string> --memory_allocator <mimalloc | jemalloc>
```

where `<name>` is a benchmark implemented as a rust binary (located in `src/bin` in the `openvm-benchmarks` crate).
For local benchmarking, the `--instance_type` flag can take an arbitrary string.
The benchmark outputs a JSON of metrics. You can process this into markdown with:

```bash
python ci/scripts/metric_unify/main.py <path to json> --aggregation-json ci/scripts/metric_unify/aggregation.json <path to metric json>
```

Currently the processing is done automatically at the end of `bench.py`. The script automatically detects if you have a previously saved metric file for the same benchmark and includes the diff report in the output.

### Flamegraphs

Flamegraphs to visualize the metrics collected by the VM cycle tracker can be generated if you have [inferno-flamegraph](https://crates.io/crates/inferno) installed. Install via

```bash
cargo install inferno
```

Then run

```bash
python ci/scripts/metric_unify/flamegraph.py <path to json>
```

The flamegraphs will be written to `*.svg` files in `.bench_metrics/flamegraphs` with respect to the repo root.

### Latest Benchmark Results

Latest benchmark results can be found [here](https://github.com/openvm-org/openvm/blob/benchmark-results/index.md).
These are run via [github workflows](./.github/workflows/benchmark-call.yml) and should always be up to date with the latest `main` branch.
