use std::path::Path;

use crate::metric::{Metric, MetricsFile};

/// Load summary metrics from a file
pub fn load_summary_metrics<P: AsRef<Path>>(
    metrics_file_path: P,
) -> Result<Vec<Metric>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(metrics_file_path)?;
    let metrics_file: MetricsFile = serde_json::from_reader(file)?;

    let mut metrics: Vec<Metric> = Default::default();
    metrics.extend(metrics_file.counter.into_iter().map(Into::into));
    metrics.extend(metrics_file.gauge.into_iter().map(Into::into));

    Ok(metrics)
}
