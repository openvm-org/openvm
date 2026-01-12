//! HTTP server for Prometheus metrics scrape endpoint.

use std::{
    net::SocketAddr,
    path::Path,
    sync::Arc,
    time::Duration,
};

use eyre::Result;
use metrics_exporter_prometheus::PrometheusBuilder;
use metrics_tracing_context::{MetricsLayer, TracingContextLayer};
use metrics_util::layers::Layer;
use tokio::sync::Notify;
use tracing_forest::ForestLayer;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

/// Sanitize a metric name to be valid for Prometheus.
/// Prometheus metric names must match [a-zA-Z_:][a-zA-Z0-9_:]*
/// This replaces invalid characters (dots, hyphens, slashes, etc.) with underscores.
fn sanitize_metric_name(name: &str) -> String {
    let mut result = String::with_capacity(name.len());
    for (i, c) in name.chars().enumerate() {
        if c.is_ascii_alphanumeric() || c == '_' || c == ':' {
            result.push(c);
        } else if i == 0 && c.is_ascii_digit() {
            // First character can't be a digit, prefix with underscore
            result.push('_');
            result.push(c);
        } else {
            result.push('_');
        }
    }
    // Ensure the name doesn't start with a digit
    if result.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        result.insert(0, '_');
    }
    result
}

/// Generate a unique run ID based on timestamp and optional git SHA.
pub fn generate_run_id(benchmark_name: &str) -> String {
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");

    // Try to get git SHA
    let git_sha = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());

    format!("{}_{}_{}",  benchmark_name, timestamp, git_sha)
}

/// Configuration for Prometheus metrics export.
#[derive(Clone, Debug)]
pub struct PrometheusConfig {
    /// Port to listen on for the metrics endpoint
    pub port: u16,
    /// How long to keep the server running after the function completes
    pub shutdown_delay_secs: u64,
    /// Unique identifier for this benchmark run
    pub run_id: String,
    /// Optional S3 bucket to upload metrics.json to
    pub s3_bucket: Option<String>,
    /// Optional S3 key prefix for the uploaded file
    pub s3_prefix: Option<String>,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            port: 9091,
            shutdown_delay_secs: 30,
            run_id: generate_run_id("benchmark"),
            s3_bucket: None,
            s3_prefix: None,
        }
    }
}

/// Run a function with Prometheus metrics export enabled.
///
/// This is a drop-in replacement for `run_with_metric_collection` from stark-sdk.
/// Instead of writing metrics to a JSON file, it starts an HTTP server that exposes
/// metrics at `/metrics` for Prometheus to scrape.
///
/// # Arguments
///
/// * `port` - The port to listen on for the metrics endpoint (e.g., 9091)
/// * `shutdown_delay_secs` - How long to keep the server running after the function completes,
///   to allow Prometheus to perform a final scrape
/// * `f` - The function to run (typically your benchmark code)
///
/// # Example
///
/// ```ignore
/// run_with_prometheus_metrics(9091, 30, || {
///     // Your benchmark code here
/// });
/// ```
pub fn run_with_prometheus_metrics<R>(port: u16, shutdown_delay_secs: u64, f: impl FnOnce() -> R) -> R {
    run_with_prometheus_metrics_config(
        PrometheusConfig {
            port,
            shutdown_delay_secs,
            run_id: generate_run_id("benchmark"),
            s3_bucket: None,
            s3_prefix: None,
        },
        f,
    )
}

/// Run a function with Prometheus metrics export enabled, with full configuration.
///
/// This version allows specifying a run_id and S3 upload options.
pub fn run_with_prometheus_metrics_config<R>(config: PrometheusConfig, f: impl FnOnce() -> R) -> R {
    let addr: SocketAddr = ([0, 0, 0, 0], config.port).into();

    // Build the Prometheus exporter with HTTP listener
    let builder = PrometheusBuilder::new().with_http_listener(addr);

    let shutdown_notify = Arc::new(Notify::new());
    let shutdown_notify_clone = shutdown_notify.clone();

    // Create a tokio runtime for the HTTP server
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime");

    let (recorder, exporter) = {
        let _guard = runtime.enter();
        builder.build().expect("Failed to build Prometheus exporter")
    };

    // Spawn the HTTP server
    runtime.spawn(async move {
        tokio::select! {
            result = exporter => {
                if result.is_err() {
                    tracing::error!("Prometheus exporter encountered an error");
                }
            }
            _ = shutdown_notify_clone.notified() => {
                tracing::info!("Shutting down Prometheus exporter");
            }
        }
    });

    // Set up tracing with metrics layer
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,p3_=warn"));

    let subscriber = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .with(MetricsLayer::new());

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set global tracing subscriber");

    // Wrap recorder with tracing context layer to propagate labels from spans
    let recorder = TracingContextLayer::all().layer(recorder);

    // Install as global metrics recorder
    metrics::set_global_recorder(recorder).expect("Failed to set global metrics recorder");

    println!("Run ID: {}", config.run_id);
    println!("Prometheus metrics available at http://0.0.0.0:{}/metrics", config.port);

    // Set the run_id as a span so it gets added to all metrics via TracingContextLayer
    let _run_span = tracing::info_span!("run", run_id = %config.run_id).entered();

    // Run the actual function
    let result = f();

    // Wait for final scrape or Ctrl+C
    if config.shutdown_delay_secs == 0 {
        println!("Benchmark complete. Metrics server running indefinitely. Press Ctrl+C to stop.");
        runtime.block_on(async {
            tokio::signal::ctrl_c().await.ok();
        });
    } else {
        println!(
            "Benchmark complete. Keeping metrics server alive for {} seconds for final scrape...",
            config.shutdown_delay_secs
        );
        runtime.block_on(async {
            tokio::time::sleep(Duration::from_secs(config.shutdown_delay_secs)).await;
        });
    }

    // Signal shutdown
    shutdown_notify.notify_one();

    // Give the server a moment to shut down gracefully
    runtime.block_on(async {
        tokio::time::sleep(Duration::from_millis(100)).await;
    });

    result
}

/// Export metrics from a JSON file or S3 to Prometheus.
///
/// This function reads a metrics JSON file and exposes those metrics via an HTTP endpoint
/// for Prometheus to scrape. The run_id is added as a label to all metrics.
///
/// # Arguments
///
/// * `source` - Path to local JSON file or S3 URI (s3://bucket/key)
/// * `run_id` - Unique identifier for this run (added as label to all metrics)
/// * `port` - The port to listen on for the metrics endpoint
/// * `duration_secs` - How long to keep the server running (0 = indefinite until Ctrl+C)
pub async fn export_metrics_to_prometheus(
    source: &str,
    run_id: &str,
    port: u16,
    duration_secs: u64,
) -> Result<()> {
    export_multiple_metrics_to_prometheus(&[(source.to_string(), run_id.to_string())], port, duration_secs).await
}

/// Export metrics from multiple JSON files or S3 sources to Prometheus.
///
/// This function reads multiple metrics JSON files and exposes all metrics via a single
/// HTTP endpoint for Prometheus to scrape. Each source gets its own run_id label.
///
/// # Arguments
///
/// * `sources` - List of (source_path, run_id) pairs. Source can be local path or S3 URI.
/// * `port` - The port to listen on for the metrics endpoint
/// * `duration_secs` - How long to keep the server running (0 = indefinite until Ctrl+C)
pub async fn export_multiple_metrics_to_prometheus(
    sources: &[(String, String)],
    port: u16,
    duration_secs: u64,
) -> Result<()> {
    use std::collections::BTreeMap;

    let addr: SocketAddr = ([0, 0, 0, 0], port).into();

    // Build the Prometheus exporter
    let builder = PrometheusBuilder::new().with_http_listener(addr);
    let (recorder, exporter) = builder.build()?;

    // Install the recorder
    metrics::set_global_recorder(recorder)?;

    // Spawn the HTTP server
    let server_handle = tokio::spawn(exporter);

    // Load and register metrics from all sources
    for (source, run_id) in sources {
        println!("Loading metrics from: {}", source);
        println!("  Run ID: {}", run_id);

        // Load metrics from source
        let file_contents = if source.starts_with("s3://") {
            load_from_s3(source).await?
        } else {
            std::fs::read_to_string(source)?
        };

        let metrics: BTreeMap<String, Vec<MetricEntry>> = serde_json::from_str(&file_contents)?;

        // Leak the run_id for static lifetime
        let run_id_static: &'static str = Box::leak(run_id.clone().into_boxed_str());

        // Register all metrics from the JSON file with run_id label
        if let Some(gauges) = metrics.get("gauge") {
            for entry in gauges {
                let mut labels: Vec<(&'static str, &'static str)> = entry
                    .labels
                    .iter()
                    .map(|(k, v)| {
                        let k: &'static str = Box::leak(k.clone().into_boxed_str());
                        let v: &'static str = Box::leak(v.clone().into_boxed_str());
                        (k, v)
                    })
                    .collect();

                // Add run_id label
                labels.push(("run_id", run_id_static));

                let value: f64 = entry.value.parse().unwrap_or(0.0);
                let sanitized_name = sanitize_metric_name(&entry.metric);
                let name: &'static str = Box::leak(sanitized_name.into_boxed_str());
                metrics::gauge!(name, &labels).set(value);
            }
        }

        if let Some(counters) = metrics.get("counter") {
            for entry in counters {
                let mut labels: Vec<(&'static str, &'static str)> = entry
                    .labels
                    .iter()
                    .map(|(k, v)| {
                        let k: &'static str = Box::leak(k.clone().into_boxed_str());
                        let v: &'static str = Box::leak(v.clone().into_boxed_str());
                        (k, v)
                    })
                    .collect();

                // Add run_id label
                labels.push(("run_id", run_id_static));

                let value: u64 = entry.value.parse().unwrap_or(0);
                let sanitized_name = sanitize_metric_name(&entry.metric);
                let name: &'static str = Box::leak(sanitized_name.into_boxed_str());
                metrics::counter!(name, &labels).absolute(value);
            }
        }
    }

    println!("\nPrometheus metrics available at http://0.0.0.0:{}/metrics", port);
    println!("Loaded {} source(s)", sources.len());

    if duration_secs == 0 {
        println!("Server running indefinitely. Press Ctrl+C to stop.");
        // Wait for Ctrl+C
        tokio::signal::ctrl_c().await?;
    } else {
        println!("Server will run for {} seconds...", duration_secs);
        tokio::time::sleep(Duration::from_secs(duration_secs)).await;
    }

    // Cancel the server
    server_handle.abort();

    Ok(())
}

/// Legacy function for backwards compatibility
pub async fn export_json_to_prometheus(
    json_path: &Path,
    port: u16,
    duration_secs: u64,
) -> Result<()> {
    let run_id = json_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    export_metrics_to_prometheus(
        json_path.to_str().unwrap_or(""),
        run_id,
        port,
        duration_secs,
    ).await
}

/// Load metrics JSON from S3
#[cfg(feature = "s3")]
async fn load_from_s3(s3_uri: &str) -> Result<String> {
    // Parse s3://bucket/key format
    let uri = s3_uri.strip_prefix("s3://").ok_or_else(|| eyre::eyre!("Invalid S3 URI"))?;
    let (bucket, key) = uri.split_once('/').ok_or_else(|| eyre::eyre!("Invalid S3 URI format"))?;

    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = aws_sdk_s3::Client::new(&config);

    let response = client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await?;

    let bytes = response.body.collect().await?.into_bytes();
    Ok(String::from_utf8(bytes.to_vec())?)
}

#[cfg(not(feature = "s3"))]
async fn load_from_s3(_s3_uri: &str) -> Result<String> {
    Err(eyre::eyre!("S3 support not enabled. Rebuild with --features s3"))
}

/// Upload metrics JSON to S3
#[cfg(feature = "s3")]
pub async fn upload_to_s3(
    content: &str,
    bucket: &str,
    key: &str,
) -> Result<()> {
    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = aws_sdk_s3::Client::new(&config);

    client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(content.as_bytes().to_vec().into())
        .content_type("application/json")
        .send()
        .await?;

    println!("Uploaded metrics to s3://{}/{}", bucket, key);
    Ok(())
}

#[cfg(not(feature = "s3"))]
pub async fn upload_to_s3(
    _content: &str,
    _bucket: &str,
    _key: &str,
) -> Result<()> {
    Err(eyre::eyre!("S3 support not enabled. Rebuild with --features s3"))
}

/// List available runs in an S3 bucket
#[cfg(feature = "s3")]
pub async fn list_runs_in_s3(bucket: &str, prefix: &str) -> Result<Vec<String>> {
    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = aws_sdk_s3::Client::new(&config);

    let mut runs = Vec::new();
    let mut continuation_token: Option<String> = None;

    loop {
        let mut request = client
            .list_objects_v2()
            .bucket(bucket)
            .prefix(prefix);

        if let Some(token) = continuation_token {
            request = request.continuation_token(token);
        }

        let response = request.send().await?;

        if let Some(contents) = response.contents {
            for object in contents {
                if let Some(key) = object.key {
                    if key.ends_with(".json") {
                        runs.push(format!("s3://{}/{}", bucket, key));
                    }
                }
            }
        }

        if response.is_truncated == Some(true) {
            continuation_token = response.next_continuation_token;
        } else {
            break;
        }
    }

    Ok(runs)
}

#[cfg(not(feature = "s3"))]
pub async fn list_runs_in_s3(_bucket: &str, _prefix: &str) -> Result<Vec<String>> {
    Err(eyre::eyre!("S3 support not enabled. Rebuild with --features s3"))
}

/// A single metric entry from the JSON file.
#[derive(Debug, serde::Deserialize)]
struct MetricEntry {
    metric: String,
    labels: Vec<(String, String)>,
    value: String,
}
