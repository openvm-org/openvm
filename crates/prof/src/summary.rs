use std::{io::Write, path::PathBuf};

use eyre::Result;
use itertools::Itertools;

use crate::{
    aggregate::{group_weight, AggregateMetrics, CELLS_USED_LABEL, CYCLES_LABEL, PROOF_TIME_LABEL},
    types::MdTableCell,
};

#[derive(Clone, Debug)]
pub struct GithubSummary {
    pub rows: Vec<SummaryRow>,
    pub benchmark_results_link: String,
}

#[derive(Clone, Debug)]
pub struct SummaryRow {
    pub name: String,
    pub md_filename: String,
    pub metrics: BenchSummaryMetrics,
}

#[derive(Clone, Debug)]
pub struct BenchSummaryMetrics {
    pub app: SingleSummaryMetrics,
    pub leaf: Option<SingleSummaryMetrics>,
    pub internals: Vec<SingleSummaryMetrics>,
    pub root: Option<SingleSummaryMetrics>,
}

#[derive(Clone, Debug)]
pub struct SingleSummaryMetrics {
    pub proof_time_ms: MdTableCell,
    pub cells_used: MdTableCell,
    pub cycles: MdTableCell,
}

impl GithubSummary {
    pub fn new(
        aggregated_metrics: &[AggregateMetrics],
        md_paths: &[PathBuf],
        benchmark_results_link: &str,
    ) -> Self {
        let rows = aggregated_metrics
            .iter()
            .zip_eq(md_paths.iter())
            .map(|(aggregated, md_path)| {
                let md_filename = md_path.file_name().unwrap().to_str().unwrap();
                aggregated.get_summary_row(md_filename)
            })
            .collect();

        Self {
            rows,
            benchmark_results_link: benchmark_results_link.to_string(),
        }
    }

    pub fn write_markdown(&self, writer: &mut impl Write) -> Result<()> {
        writeln!(writer, "| group | app.proof_time_ms | app.cycles | app.cells_used | leaf.proof_time_ms | leaf.cycles | leaf.cells_used |")?;
        write!(writer, "| -- |")?;
        for _ in 0..6 {
            write!(writer, " -- |")?;
        }
        writeln!(writer)?;

        for row in self.rows.iter() {
            write!(
                writer,
                "| [{}]({}/individual/{}) |",
                row.name, self.benchmark_results_link, row.md_filename
            )?;
            row.metrics.write_partial_md_row(writer)?;
            writeln!(writer)?;
        }
        writeln!(writer)?;

        Ok(())
    }
}

impl BenchSummaryMetrics {
    pub fn write_partial_md_row(&self, writer: &mut impl Write) -> Result<()> {
        self.app.write_partial_md_row(writer)?;
        if let Some(leaf) = &self.leaf {
            leaf.write_partial_md_row(writer)?;
        } else {
            // Always write placeholder for leaf
            write!(writer, "- | - | - |")?;
        }
        // Don't print other metrics in summary for now:

        // for internal in &self.internals {
        //     internal.write_partial_md_row(writer)?;
        // }
        // if let Some(root) = &self.root {
        //     root.write_partial_md_row(writer)?;
        // }

        Ok(())
    }
}

impl SingleSummaryMetrics {
    pub fn write_partial_md_row(&self, writer: &mut impl Write) -> Result<()> {
        write!(
            writer,
            "{} | {} | {} |",
            self.proof_time_ms, self.cycles, self.cells_used,
        )?;
        Ok(())
    }
}

impl AggregateMetrics {
    pub fn get_single_summary(&self, name: &str) -> Option<SingleSummaryMetrics> {
        let stats = self.by_group.get(name)?;
        let cells_used = MdTableCell::new(stats.get(CELLS_USED_LABEL)?.sum, None);
        let cycles = MdTableCell::new(stats.get(CYCLES_LABEL)?.sum, None);
        let proof_time_ms = MdTableCell::new(stats.get(PROOF_TIME_LABEL)?.sum, None);
        Some(SingleSummaryMetrics {
            cells_used,
            cycles,
            proof_time_ms,
        })
    }

    pub fn get_summary_row(&self, md_filename: &str) -> SummaryRow {
        // A hacky way to determine the app name
        let app_name = self
            .by_group
            .keys()
            .find(|k| group_weight(k) == 0)
            .expect("cannot find app name");
        let app = self.get_single_summary(app_name).unwrap();
        let leaf = self.get_single_summary("leaf");
        let mut internals = Vec::new();
        let mut hgt = 0;
        while let Some(internal) = self.get_single_summary(&format!("internal.{hgt}")) {
            internals.push(internal);
            hgt += 1;
        }
        let root = self.get_single_summary("root");
        SummaryRow {
            name: app_name.to_string(),
            md_filename: md_filename.to_string(),
            metrics: BenchSummaryMetrics {
                app,
                leaf,
                internals,
                root,
            },
        }
    }
}
