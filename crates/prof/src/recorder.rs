//! Prometheus recorder
use std::{
    sync::{atomic::AtomicBool, Arc, LazyLock},
    thread,
    time::Duration,
};

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use metrics_process::Collector;
use metrics_tracing_context::{MetricsLayer, TracingContextLayer};
use metrics_util::layers::Layer;
use tracing_subscriber::{layer::SubscriberExt, Registry};

/// Installs the Prometheus recorder as the global recorder.
///
/// Caution: This only configures the global recorder and does not spawn the exporter.
pub fn install_prometheus_recorder() -> &'static PrometheusRecorder {
    &PROMETHEUS_RECORDER_HANDLE
}

/// The default Prometheus recorder handle. We use a global static to ensure that it is only
/// installed once.
static PROMETHEUS_RECORDER_HANDLE: LazyLock<PrometheusRecorder> =
    LazyLock::new(PrometheusRecorder::new);

pub struct PrometheusRecorder {
    handle: Arc<PrometheusHandle>,
    spawned: AtomicBool,
}

impl PrometheusRecorder {
    /// Installs Prometheus as the metrics recorder.
    ///
    /// Caution: This only configures the global recorder and does not spawn the exporter.
    /// Callers must run `PrometheusHandle.run_upkeep` manually.
    fn new() -> Self {
        // Set up tracing:
        let subscriber = Registry::default().with(MetricsLayer::new());
        tracing::subscriber::set_global_default(subscriber)
            .expect("Error initializing the tracing subscriber");

        // Prepare metrics.
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = Arc::new(recorder.handle());
        let recorder_with_tracing = TracingContextLayer::all().layer(recorder);
        metrics::set_global_recorder(recorder_with_tracing).unwrap();

        Self {
            handle,
            spawned: AtomicBool::new(false),
        }
    }

    // Creates a thread that regularly collects process metrics.
    // Necessary to avoid the VM starving the process.
    pub fn spawn_process_metrics_thread(&self, interval: Duration) {
        if self
            .spawned
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_err()
        {
            return;
        }

        let handle = self.handle().clone();
        thread::spawn(move || {
            let collector = Collector::default();
            collector.describe();

            loop {
                collector.collect();
                handle.run_upkeep();
                thread::sleep(interval);
            }
        });
    }

    pub fn handle(&self) -> &PrometheusHandle {
        &self.handle
    }
}

#[cfg(test)]
mod tests {
    use metrics::{counter, describe_counter};
    use tracing::{span, Level};

    use super::*;
    // Dependencies using different version of the `metrics` crate (to be exact, 0.21 vs 0.22)
    // may not be able to communicate with each other through the global recorder.
    //
    // This test ensures that `metrics-process` dependency plays well with the current
    // `metrics-exporter-prometheus` dependency version.
    #[test]
    fn test_prometheus_recorder_process_metrics() {
        // initialize the lazy handle
        let handle = &PROMETHEUS_RECORDER_HANDLE.handle;
        let span = span!(Level::TRACE, "my_span", test = "process_metrics");
        let _guard = span.enter();

        let collector = Collector::default();
        collector.collect();
        let metrics = handle.render();
        assert!(
            metrics.contains("process_cpu_seconds_total{test=\"process_metrics\"} 0"),
            "{metrics:?}"
        );
    }

    #[test]
    fn test_prometheus_metrics_tracing_context() {
        let handle = &PROMETHEUS_RECORDER_HANDLE.handle;
        let span = span!(Level::TRACE, "my_span", test = "tracing_context");
        let _guard = span.enter();

        describe_counter!("example_metric", "A counter for demonstration purposes");
        counter!("example_metric").increment(42);

        let metrics = handle.render();
        assert!(
            metrics.contains("example_metric{test=\"tracing_context\"} 42"),
            "{metrics:?}"
        );
    }
}
