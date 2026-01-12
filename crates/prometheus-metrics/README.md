# OpenVM Prometheus Metrics

A Rust crate for exporting OpenVM benchmark metrics to Prometheus.

## Features

- **Live metrics during benchmarks**: Stream metrics to Prometheus in real-time via HTTP scrape endpoint
- **Import existing metrics.json**: Load metrics from JSON files (local or S3) into Prometheus
- **Run ID labeling**: Each benchmark run gets a unique identifier for filtering in Grafana
- **Metric name sanitization**: Automatically converts invalid characters (dots, hyphens, slashes) to underscores

## Usage

### Run benchmark with Prometheus metrics

Add the `--prometheus` flag to stream metrics to Prometheus:

```bash
cargo run --release -p openvm-benchmarks-prove --bin regex -- --prometheus
```

The metrics server runs on port 9091 by default (configurable with `--prometheus-port`).
After the benchmark completes, the server continues running until you press Ctrl+C.

### Import metrics.json to Prometheus

Use the `export-to-prometheus` CLI to load an existing metrics.json file:

```bash
# From local file
cargo run --release -p openvm-prometheus-metrics --bin export-to-prometheus -- \
  --source ./metrics.json

# With custom run ID
cargo run --release -p openvm-prometheus-metrics --bin export-to-prometheus -- \
  --source ./metrics.json \
  --run-id "my-benchmark-run"

# From S3 (requires --features s3)
cargo run --release -p openvm-prometheus-metrics --features s3 --bin export-to-prometheus -- \
  --source s3://my-bucket/benchmarks/metrics.json
```

### Programmatic usage

```rust
use openvm_prometheus_metrics::run_with_prometheus_metrics;

fn main() {
    run_with_prometheus_metrics(9091, 0, || {
        // Your benchmark code here
    });
}
```

## Viewing Metrics in Grafana

1. Start the Prometheus and Grafana stack:
   ```bash
   cd crates/grafana
   docker-compose up -d
   ```

2. Run your benchmark with `--prometheus` or use `export-to-prometheus`

3. Open Grafana at http://localhost:3000 (login: admin/admin)

4. Navigate to the "OpenVM Benchmark Metrics" dashboard

5. Select your **Run ID** and **Group** from the dropdowns

## CLI Options

### export-to-prometheus

```
Options:
      --source <SOURCE>      Source: path to local JSON file or S3 URI (s3://bucket/key)
      --run-id <RUN_ID>      Run ID to use for metrics labeling (defaults to filename)
      --port <PORT>          Port to expose metrics on [default: 9091]
      --duration <DURATION>  Duration in seconds to keep server alive (0 = indefinite) [default: 0]
      --list-s3              List available runs in S3 bucket
      --s3-bucket <BUCKET>   S3 bucket name (for --list-s3)
      --s3-prefix <PREFIX>   S3 key prefix (for --list-s3) [default: ""]
```

## Metric Name Sanitization

Prometheus metric names must match `[a-zA-Z_:][a-zA-Z0-9_:]*`. This crate automatically sanitizes metric names:

| Original | Sanitized |
|----------|-----------|
| `client.execute_time_ms` | `client_execute_time_ms` |
| `execute_e1_insn_mi/s` | `execute_e1_insn_mi_s` |
| `reth-block_time_ms` | `reth_block_time_ms` |

## S3 Support

S3 support is optional and requires the `s3` feature:

```bash
cargo build -p openvm-prometheus-metrics --features s3
```

This enables:
- Loading metrics.json from S3 URIs
- Listing available benchmark runs in S3
- Uploading metrics to S3 (programmatic API)
