//! Prometheus metrics exporter for OpenVM benchmarks.
//!
//! This crate provides a drop-in replacement for `run_with_metric_collection` from stark-sdk
//! that exports metrics to Prometheus via an HTTP scrape endpoint instead of writing to a JSON file.
//!
//! # Usage
//!
//! ## Live metrics during benchmark
//!
//! ```ignore
//! use openvm_prometheus_metrics::run_with_prometheus_metrics;
//!
//! fn main() {
//!     run_with_prometheus_metrics(9091, 30, || {
//!         // Your benchmark code here
//!     });
//! }
//! ```
//!
//! ## With custom configuration and run_id
//!
//! ```ignore
//! use openvm_prometheus_metrics::{run_with_prometheus_metrics_config, PrometheusConfig, generate_run_id};
//!
//! fn main() {
//!     let config = PrometheusConfig {
//!         port: 9091,
//!         shutdown_delay_secs: 30,
//!         run_id: generate_run_id("my_benchmark"),
//!         s3_bucket: Some("my-bucket".to_string()),
//!         s3_prefix: Some("benchmarks/".to_string()),
//!     };
//!     run_with_prometheus_metrics_config(config, || {
//!         // Your benchmark code here
//!     });
//! }
//! ```
//!
//! ## Export existing metrics.json to Prometheus
//!
//! ```ignore
//! use openvm_prometheus_metrics::export_metrics_to_prometheus;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Load from local file or S3
//!     export_metrics_to_prometheus(
//!         "./metrics.json",  // or "s3://bucket/path/metrics.json"
//!         "my_run_id",
//!         9091,
//!         60,
//!     ).await.unwrap();
//! }
//! ```

mod server;

pub use server::{
    export_json_to_prometheus,
    export_metrics_to_prometheus,
    generate_run_id,
    list_runs_in_s3,
    run_with_prometheus_metrics,
    run_with_prometheus_metrics_config,
    upload_to_s3,
    PrometheusConfig,
};
