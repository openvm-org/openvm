use std::{
    fs,
    io::{stdout, Write},
    path::PathBuf,
};

use clap::{Parser, Subcommand};
use eyre::Result;
use itertools::Itertools;
use openvm_prof::{
    aggregate::{GroupedMetrics, VM_METRIC_NAMES},
    summary::GithubSummary,
    types::{BenchmarkOutput, MetricDb},
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to the metrics JSON files
    #[arg(long, value_delimiter = ',')]
    json_paths: Vec<PathBuf>,

    /// Previous metric json paths (optional).
    /// If provided, must be same length and in same order as `json-paths`.
    /// Some file paths may be passed in that do not exist to account for new benchmarks.
    #[arg(long, value_delimiter = ',')]
    prev_json_paths: Option<Vec<PathBuf>>,

    /// Display names for each metrics file (optional).
    /// If provided, must be same length and in same order as `json-paths`.
    /// Otherwise, the app program name will be used.
    #[arg(long, value_delimiter = ',')]
    names: Option<Vec<String>>,

    /// Path to write the output JSON in BMF format
    #[arg(long)]
    output_json: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Summary(SummaryCmd),
}

#[derive(Parser, Debug)]
struct SummaryCmd {
    #[arg(long)]
    benchmark_results_link: String,
    #[arg(long)]
    summary_md_path: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let prev_json_paths = if let Some(paths) = args.prev_json_paths {
        paths.into_iter().map(Some).collect()
    } else {
        vec![None; args.json_paths.len()]
    };
    let names = args.names.unwrap_or_default();
    let mut names = if names.len() == args.json_paths.len() {
        names
    } else {
        vec!["".to_string(); args.json_paths.len()]
    };
    let mut aggregated_metrics = Vec::new();
    let mut md_paths = Vec::new();
    let mut output = BenchmarkOutput::default();
    for ((metrics_path, prev_metrics_path), name) in args
        .json_paths
        .into_iter()
        .zip_eq(prev_json_paths)
        .zip_eq(&mut names)
    {
        let db = MetricDb::new(&metrics_path)?;
        let grouped = GroupedMetrics::new(&db, "group")?;
        let mut aggregated = grouped.aggregate();
        let mut prev_aggregated = None;
        if let Some(prev_path) = prev_metrics_path {
            // If this is a new benchmark, prev_path will not exist
            if let Ok(prev_db) = MetricDb::new(&prev_path) {
                let prev_grouped = GroupedMetrics::new(&prev_db, "group")?;
                let prev_grouped_aggregated = prev_grouped.aggregate();
                aggregated.set_diff(&prev_grouped_aggregated);
                prev_aggregated = Some(prev_grouped_aggregated);
            }
        }
        if name.is_empty() {
            *name = aggregated.name();
        }
        output.insert(name, aggregated.to_bencher_metrics());
        let mut writer = Vec::new();
        aggregated.write_markdown(&mut writer, VM_METRIC_NAMES)?;

        let mut markdown_output = String::from_utf8(writer)?;

        // TODO: calculate diffs for detailed metrics
        // Add detailed metrics in a collapsible section
        markdown_output.push_str("\n<details>\n<summary>Detailed Metrics</summary>\n\n");
        markdown_output.push_str(&db.generate_markdown_tables());
        markdown_output.push_str("</details>\n\n");

        let md_path = metrics_path.with_extension("md");
        fs::write(&md_path, markdown_output)?;
        md_paths.push(md_path);
        aggregated_metrics.push((aggregated, prev_aggregated));
    }
    if let Some(path) = args.output_json {
        fs::write(&path, serde_json::to_string_pretty(&output)?)?;
    }
    if let Some(command) = args.command {
        match command {
            Commands::Summary(cmd) => {
                let summary = GithubSummary::new(
                    &names,
                    &aggregated_metrics,
                    &md_paths,
                    &cmd.benchmark_results_link,
                );
                let mut writer = Vec::new();
                summary.write_markdown(&mut writer)?;
                if let Some(path) = cmd.summary_md_path {
                    fs::write(&path, writer)?;
                } else {
                    stdout().write_all(&writer)?;
                }
            }
        }
    }

    Ok(())
}
