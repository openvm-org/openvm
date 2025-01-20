use std::{sync::LazyLock, thread, time::Duration};

use metrics_exporter_prometheus::{BuildError, PrometheusBuilder};
use metrics_process::Collector;

/// Installs the Prometheus recorder as the global recorder.
pub fn install_prometheus_gateway() -> &'static PrometheusMetrics {
    &PROMETHEUS_METRICS_HANDLE
}

/// The default Prometheus recorder handle. We use a global static to ensure that it is only
/// installed once.
static PROMETHEUS_METRICS_HANDLE: LazyLock<PrometheusMetrics> =
    LazyLock::new(|| PrometheusMetrics::install().unwrap());

/// This is intended to be used as the global recorder.
#[derive(Debug)]
pub struct PrometheusMetrics;

impl PrometheusMetrics {
    fn install() -> eyre::Result<Self, BuildError> {
        PrometheusBuilder::new()
            .with_push_gateway(
                "http://127.0.0.1:9091/metrics/job/openvm",
                Duration::from_secs(10),
                None,
                None,
            )?
            .install()?; // Install the configured builder

        println!("prometheus-metrics: push gateway installed. Make sure you are running the prometheus nodes");

        PrometheusMetrics::spawn_thread();

        Ok(Self)
    }

    fn spawn_thread() {
        thread::spawn(move || {
            let collector = Collector::default();
            collector.describe();

            loop {
                collector.collect();
                thread::sleep(Duration::from_secs(5));
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prometheus_recorder() {
        let handle = PrometheusBuilder::new().install_recorder().unwrap();
        let collector = Collector::default();
        collector.describe();
        collector.collect();

        let metrics = handle.render();
        println!("{}", metrics);
        assert!(metrics.contains("process_cpu_seconds_total"), "{metrics:?}");
    }
}
