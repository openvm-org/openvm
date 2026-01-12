//! CLI tool to export metrics from JSON files or S3 to Prometheus.
//!
//! # Usage
//!
//! Export from a single local file:
//! ```bash
//! cargo run --bin export-to-prometheus -- --source ./metrics.json
//! ```
//!
//! Export multiple files for comparison:
//! ```bash
//! cargo run --bin export-to-prometheus -- --source ./run1.json --source ./run2.json
//! ```
//!
//! Export from S3 (requires --features s3):
//! ```bash
//! cargo run --bin export-to-prometheus --features s3 -- --source s3://bucket/path/metrics.json
//! ```
//!
//! List available runs in S3:
//! ```bash
//! cargo run --bin export-to-prometheus --features s3 -- --list-s3 --s3-bucket my-bucket --s3-prefix benchmarks/
//! ```

use clap::Parser;
use eyre::Result;
use openvm_prometheus_metrics::{export_multiple_metrics_to_prometheus, list_runs_in_s3};

#[derive(Parser)]
#[command(name = "export-to-prometheus")]
#[command(about = "Export metrics.json files (from local paths or S3) to Prometheus scrape endpoint")]
struct Args {
    /// Source: path to local JSON file or S3 URI (s3://bucket/key).
    /// Can be specified multiple times to load multiple files for comparison.
    #[arg(long, required_unless_present = "list_s3")]
    source: Vec<String>,

    /// Port to expose metrics on
    #[arg(long, default_value = "9091")]
    port: u16,

    /// Duration in seconds to keep the server alive (0 = indefinite until Ctrl+C)
    #[arg(long, default_value = "0")]
    duration: u64,

    /// List available runs in S3 bucket
    #[arg(long)]
    list_s3: bool,

    /// S3 bucket name (for --list-s3)
    #[arg(long, required_if_eq("list_s3", "true"))]
    s3_bucket: Option<String>,

    /// S3 key prefix (for --list-s3)
    #[arg(long, default_value = "")]
    s3_prefix: String,
}

/// Derive run_id from a source path (filename without extension)
fn derive_run_id(source: &str) -> String {
    if source.starts_with("s3://") {
        // Extract filename from S3 key
        source
            .rsplit('/')
            .next()
            .and_then(|f| f.strip_suffix(".json"))
            .unwrap_or("unknown")
            .to_string()
    } else {
        // Extract filename from local path
        std::path::Path::new(source)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if args.list_s3 {
        let bucket = args.s3_bucket.as_ref().expect("--s3-bucket required with --list-s3");
        println!("Listing runs in s3://{}/{}...", bucket, args.s3_prefix);

        let runs = list_runs_in_s3(bucket, &args.s3_prefix).await?;

        if runs.is_empty() {
            println!("No metrics.json files found.");
        } else {
            println!("Available runs:");
            for run in runs {
                println!("  {}", run);
            }
        }
        return Ok(());
    }

    if args.source.is_empty() {
        eprintln!("Error: at least one --source is required");
        std::process::exit(1);
    }

    // Build list of (source, run_id) pairs
    let sources: Vec<(String, String)> = args
        .source
        .iter()
        .map(|s| {
            let run_id = derive_run_id(s);
            (s.clone(), run_id)
        })
        .collect();

    println!("Port: {}", args.port);
    println!("Loading {} source(s)...\n", sources.len());

    export_multiple_metrics_to_prometheus(&sources, args.port, args.duration).await?;

    println!("Done.");
    Ok(())
}
