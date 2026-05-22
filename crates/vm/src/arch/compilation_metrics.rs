use std::time::Instant;

pub(crate) struct CompilationTimer {
    metric_name: &'static str,
    backend: &'static str,
    start: Instant,
}

impl CompilationTimer {
    pub(crate) fn start(metric_name: &'static str, backend: &'static str) -> Self {
        Self {
            metric_name,
            backend,
            start: Instant::now(),
        }
    }
}

impl Drop for CompilationTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        // Compilation gauges report milliseconds.
        let ms = elapsed.as_secs_f64() * 1000.0;
        tracing::info!(
            metric = self.metric_name,
            backend = self.backend,
            compilation_time_ms = ms,
            "compilation_time"
        );
        metrics::gauge!(self.metric_name, "backend" => self.backend).set(ms);
    }
}
