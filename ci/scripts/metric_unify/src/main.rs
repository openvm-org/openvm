mod aggregate;
mod diff;
mod markdown;
mod metric;
mod summary;
mod util;

use aggregate::{aggregate_metrics, load_aggregation_metrics};
use clap::{Parser, Subcommand};
use diff::diff_metrics;
use markdown::Tables;
use summary::load_summary_metrics;
use util::to_tables;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate summary tables
    Summary {
        /// Path to the metrics JSON file
        #[arg(long, value_name = "METRICS_JSON", value_parser = clap::value_parser!(std::path::PathBuf))]
        metrics_json: std::path::PathBuf,
    },
    /// Generate aggregated tables
    Aggregate {
        /// Path to the metrics JSON file
        #[arg(long, value_name = "METRICS_JSON", value_parser = clap::value_parser!(std::path::PathBuf))]
        metrics_json: std::path::PathBuf,

        /// Path to the aggregation JSON file
        #[arg(long, value_name = "AGGREGATION_JSON", value_parser = clap::value_parser!(std::path::PathBuf))]
        aggregation_json: std::path::PathBuf,
    },
    /// Compare metrics with previous metrics and show differences
    Diff {
        /// Path to the current metrics JSON file
        #[arg(long, value_name = "METRICS_JSON", value_parser = clap::value_parser!(std::path::PathBuf))]
        metrics_json: std::path::PathBuf,

        /// Path to the previous metrics JSON file
        #[arg(long, value_name = "PREV_METRICS_JSON", value_parser = clap::value_parser!(std::path::PathBuf))]
        prev_metrics_json: std::path::PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.command {
        Commands::Summary { metrics_json } => {
            let summary = load_summary_metrics(&metrics_json)?;
            let tables = to_tables(summary);
            println!("\n<details>\n<summary>Detailed Metrics</summary>\n\n");
            println!("{}", Tables::from(tables));
            println!("</details>\n\n");
        }
        Commands::Aggregate {
            metrics_json,
            aggregation_json,
        } => {
            let summary = load_summary_metrics(&metrics_json)?;
            let aggregations = load_aggregation_metrics(&aggregation_json)?;
            let aggregated_metrics = aggregate_metrics(aggregations, summary);
            let tables = to_tables(aggregated_metrics);
            println!("{}", Tables::from(tables));
        }
        Commands::Diff {
            metrics_json,
            prev_metrics_json,
        } => {
            let summary = load_summary_metrics(&metrics_json)?;
            let prev_summary = load_summary_metrics(&prev_metrics_json)?;
            let diff_matrics = diff_metrics(summary, prev_summary);
            let tables = to_tables(diff_matrics);
            println!("\n<details>\n<summary>Detailed Metrics</summary>\n\n");
            println!("{}", Tables::from(tables));
            println!("</details>\n\n");
        }
    }
    Ok(())
}
