use std::{collections::BTreeMap, ffi::OsStr};

use metrics_tracing_context::{MetricsLayer, TracingContextLayer};
use metrics_util::{
    debugging::{DebugValue, DebuggingRecorder, Snapshot},
    layers::Layer,
    CompositeKey, MetricKind,
};
use serde_json::json;
use tracing::Level;
use tracing_forest::ForestLayer;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

/// Run a function with metric collection enabled. The metrics will be written to a file specified
/// by an environment variable which name is `output_path_envar`.
pub fn run_with_metric_collection(output_path_envar: impl AsRef<OsStr>, f: impl FnOnce()) {
    let path = std::env::var(output_path_envar).unwrap();
    let file = std::fs::File::create(path).unwrap();
    // Set up tracing:
    let env_filter = EnvFilter::builder()
        .with_default_directive(Level::INFO.into())
        .from_env_lossy();
    let subscriber = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .with(MetricsLayer::new());
    // Prepare tracing.
    tracing::subscriber::set_global_default(subscriber).unwrap();

    // Prepare metrics.
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let recorder = TracingContextLayer::all().layer(recorder);
    // Install the registry as the global recorder
    metrics::set_global_recorder(recorder).unwrap();
    f();
    serde_json::to_writer_pretty(&file, &serialize_metric_snapshot(snapshotter.snapshot()))
        .unwrap();
}

/// Serialize a gauge metric into a JSON object. The object has the following structure:
/// {
///    "metric": <Metric Name>,
///    "labels": [
///       (<key1>, <value1>),
///       (<key2>, <value2>),
///     ],
///    "value": <float value>
/// }
///
fn serialize_gauge_metric(ckey: CompositeKey, value: DebugValue) -> serde_json::Value {
    let (kind, key) = ckey.into_parts();
    assert_eq!(kind, MetricKind::Gauge, "Unexpected metric kind");
    let (key_name, labels) = key.into_parts();
    let value = match value {
        DebugValue::Gauge(v) => v.into_inner(),
        _ => unreachable!(),
    };
    let labels = labels
        .into_iter()
        .map(|label| {
            let (k, v) = label.into_parts();
            (k.as_ref().to_owned(), v.as_ref().to_owned())
        })
        .collect::<Vec<_>>();

    json!({
        "metric": key_name.as_str(),
        "labels": labels,
        "value": value,
    })
}

/// Serialize a metric snapshot into a JSON object. The object has the following structure:
/// {
///   "gauge": [
///     {
///         "metric": <Metric Name>,
///         "labels": [
///             (<key1>, <value1>),
///             (<key2>, <value2>),
///         ],
///         "value": <float value>
///     },
///     ...
///   ],
///   ...
/// }
///
fn serialize_metric_snapshot(snapshot: Snapshot) -> serde_json::Value {
    let mut ret = BTreeMap::<_, Vec<serde_json::Value>>::new();
    for (ckey, _, _, value) in snapshot.into_vec() {
        match ckey.kind() {
            MetricKind::Gauge => {
                ret.entry("gauge")
                    .or_default()
                    .push(serialize_gauge_metric(ckey, value));
            }
            _ => todo!(),
        }
    }
    json!(ret)
}
