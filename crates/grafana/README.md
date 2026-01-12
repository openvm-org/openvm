# OpenVM Grafana Dashboard

This directory contains configuration files for running Prometheus and Grafana to visualize OpenVM benchmark metrics.

## Quick Start

1. Start the Prometheus and Grafana stack:
   ```bash
   cd crates/grafana
   docker-compose up -d
   ```

2. Run a benchmark with Prometheus metrics enabled:
   ```bash
   cargo run --bin regex --release -- --prometheus
   ```

3. Access Grafana at http://localhost:3000
   - Default credentials: admin/admin
   - The OpenVM Benchmark dashboard is auto-provisioned

## Architecture

- **Prometheus** (port 9090): Scrapes metrics from the benchmark process
- **Grafana** (port 3000): Visualizes metrics with pre-configured dashboards

## Dashboard Panels

The OpenVM Benchmark dashboard includes:

1. **Summary Stats Row**: Total proof time, cells used, instructions, and cycles
2. **Cells by Segment Bar Chart**: Stacked bar chart showing cells per segment, colored by AIR name
3. **Summary by Group Table**: Breakdown of metrics by group (app, leaf, internal, root, etc.)
4. **Instructions by Segment**: Bar chart showing instruction count per segment

## Configuration

### Prometheus

The Prometheus configuration (`prometheus/prometheus.yml`) is set to scrape metrics from `host.docker.internal:9091` which is the default port for the benchmark's Prometheus endpoint.

To change the target port, modify:
```yaml
scrape_configs:
  - job_name: 'openvm-benchmark'
    static_configs:
      - targets: ['host.docker.internal:YOUR_PORT']
```

### Custom Dashboard

To customize the dashboard, you can either:
1. Edit `dashboards/openvm-benchmark.json` directly
2. Make changes in the Grafana UI and export the updated JSON

## Troubleshooting

### No data showing in Grafana

1. Check that the benchmark is running with `--prometheus` flag
2. Verify Prometheus can reach the benchmark:
   ```bash
   curl http://localhost:9091/metrics
   ```
3. Check Prometheus targets at http://localhost:9090/targets

### Docker networking issues (macOS/Linux)

If `host.docker.internal` doesn't work, try:
- macOS: Should work out of the box
- Linux: Add `--add-host=host.docker.internal:host-gateway` to docker run, or use `host` network mode

## Stopping the Stack

```bash
docker-compose down
```

To also remove the data volumes:
```bash
docker-compose down -v
```
