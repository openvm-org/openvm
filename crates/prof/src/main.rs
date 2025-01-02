use std::{
    fs,
    io::{stdout, Write},
    path::PathBuf,
};

use clap::{Parser, Subcommand};
use eyre::Result;
use openvm_prof::{
    aggregate::{GroupedMetrics, VM_METRIC_NAMES},
    summary::GithubSummary,
    types::MetricDb,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to the metrics JSON files
    #[arg(long, value_delimiter = ',')]
    json_paths: Vec<PathBuf>,
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

    let mut aggregated_metrics = Vec::new();
    let mut md_paths = Vec::new();
    for metrics_path in args.json_paths {
        let db = MetricDb::new(&metrics_path)?;

        let grouped = GroupedMetrics::new(&db, "group")?;
        let aggregated = grouped.aggregate();
        let mut writer = Vec::new();
        aggregated.write_markdown(&mut writer, VM_METRIC_NAMES)?;
        let mut markdown_output = String::from_utf8(writer)?;

        // Add detailed metrics in a collapsible section
        markdown_output.push_str("\n<details>\n<summary>Detailed Metrics</summary>\n\n");
        markdown_output.push_str(&db.generate_markdown_tables());
        markdown_output.push_str("</details>\n\n");

        let md_path = metrics_path.with_extension("md");
        fs::write(&md_path, markdown_output)?;
        md_paths.push(md_path);
        aggregated_metrics.push(aggregated);
    }
    if let Some(command) = args.command {
        match command {
            Commands::Summary(cmd) => {
                let summary =
                    GithubSummary::new(&aggregated_metrics, &md_paths, &cmd.benchmark_results_link);
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
