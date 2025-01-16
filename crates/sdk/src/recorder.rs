//! Prometheus recorder
//! The code in this file was in most part taken from: https://github.com/paradigmxyz/reth/blob/b4610a04e6a1ceecbeacce92da5e14ea43476e57/crates/node/metrics/src/recorder.rs
use std::{
    sync::{atomic::AtomicBool, LazyLock},
    thread,
    time::Duration,
};

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use metrics_process::Collector;
use metrics_util::layers::{PrefixLayer, Stack};

/// Installs the Prometheus recorder as the global recorder.
///
/// Caution: This only configures the global recorder and does not spawn the exporter.
pub fn install_prometheus_recorder() -> &'static PrometheusRecorder {
    &PROMETHEUS_RECORDER_HANDLE
}

/// The default Prometheus recorder handle. We use a global static to ensure that it is only
/// installed once.
static PROMETHEUS_RECORDER_HANDLE: LazyLock<PrometheusRecorder> =
    LazyLock::new(|| PrometheusRecorder::install().unwrap());

/// A handle to the Prometheus recorder.
///
/// This is intended to be used as the global recorder.
/// Callers must ensure that [`PrometheusRecorder::spawn_upkeep`] is called once.
#[derive(Debug)]
pub struct PrometheusRecorder {
    handle: PrometheusHandle,
    upkeep: AtomicBool,
}

impl PrometheusRecorder {
    /// Installs Prometheus as the metrics recorder.
    ///
    /// Caution: This only configures the global recorder and does not spawn the exporter.
    /// Callers must run [`Self::spawn_upkeep`] manually.
    fn new() -> Self {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();

        // Build metrics stack
        Stack::new(recorder)
            .push(PrefixLayer::new("openvm"))
            .install()
            .expect("Couldn't set metrics recorder.");

        Self {
            handle,
            upkeep: AtomicBool::new(false),
        }
    }

    /// Spawns the upkeep thread if there hasn't been one spawned already.
    /// See also [`PrometheusHandle::run_upkeep`]
    fn spawn_upkeep(&self) {
        if self
            .upkeep
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

        let handle = self.handle.clone();
        let collector = Collector::default();
        collector.describe();
        let mut counter = 0;
        thread::spawn(move || loop {
            counter %= 30;
            if counter == 0 {
                handle.run_upkeep();
            }

            collector.collect();
            counter += 1;
            thread::sleep(Duration::from_secs(10));
        });
    }

    pub fn install() -> eyre::Result<Self> {
        let recorder = Self::new();
        recorder.spawn_upkeep();
        Ok(recorder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Dependencies using different version of the `metrics` crate (to be exact, 0.21 vs 0.22)
    // may not be able to communicate with each other through the global recorder.
    //
    // This test ensures that `metrics-process` dependency plays well with the current
    // `metrics-exporter-prometheus` dependency version.
    #[test]
    fn process_metrics() {
        // initialize the lazy handle
        let _ = &*PROMETHEUS_RECORDER_HANDLE;
        thread::sleep(Duration::from_secs(2));

        let metrics = PROMETHEUS_RECORDER_HANDLE.handle.render();
        println!("{}", metrics);
        assert!(metrics.contains("process_cpu_seconds_total"), "{metrics:?}");
    }
}
