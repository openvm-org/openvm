use std::{io::Write, path::PathBuf};

use eyre::Result;
use itertools::Itertools;

use crate::{
    aggregate::{
        AggregateMetrics, EXECUTE_METERED_TIME_LABEL, EXECUTE_PREFLIGHT_INSNS_LABEL,
        EXECUTE_PREFLIGHT_TIME_LABEL, MAIN_CELLS_USED_LABEL, PROOF_TIME_LABEL,
        PROVE_EXCL_TRACE_TIME_LABEL, TRACE_GEN_TIME_LABEL,
    },
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
    pub halo2_outer: Option<SingleSummaryMetrics>,
    pub halo2_wrapper: Option<SingleSummaryMetrics>,
}

#[derive(Clone, Debug)]
pub struct SingleSummaryMetrics {
    pub proof_time_ms: MdTableCell,
    /// Parallel proof time is approximated as the max of proof times within a group
    pub par_proof_time_ms: MdTableCell,
    pub cells_used: MdTableCell,
    pub insns: MdTableCell,
}

impl GithubSummary {
    pub fn new(
        names: &[String],
        aggregated_metrics: &[(AggregateMetrics, Option<AggregateMetrics>)],
        md_paths: &[PathBuf],
        benchmark_results_link: &str,
    ) -> Self {
        let rows = aggregated_metrics
            .iter()
            .zip_eq(md_paths.iter())
            .zip_eq(names)
            .map(|(((aggregated, prev_aggregated), md_path), name)| {
                let md_filename = md_path
                    .file_name()
                    .expect("Path should have a filename")
                    .to_str()
                    .expect("Filename should be valid UTF-8");
                let mut row = aggregated.get_summary_row(md_filename).unwrap_or_else(|| {
                    panic!("Failed to get summary row for file '{}'", md_filename)
                });
                if let Some(prev_aggregated) = prev_aggregated {
                    // md_filename doesn't matter
                    if let Some(prev_row) = prev_aggregated.get_summary_row(md_filename) {
                        if row.name == prev_row.name {
                            row.metrics.set_diff(&prev_row.metrics);
                        }
                    }
                }
                row.name = name.clone();
                row
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
                "| [{}]({}/{}) |",
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

    pub fn set_diff(&mut self, prev: &Self) {
        self.app.set_diff(&prev.app);
        if let (Some(leaf), Some(prev_leaf)) = (&mut self.leaf, &prev.leaf) {
            leaf.set_diff(prev_leaf);
        }
        for (internal, prev_internal) in self.internals.iter_mut().zip(prev.internals.iter()) {
            internal.set_diff(prev_internal);
        }
        if let (Some(root), Some(prev_root)) = (&mut self.root, &prev.root) {
            root.set_diff(prev_root);
        }
    }
}

impl SingleSummaryMetrics {
    pub fn write_partial_md_row(&self, writer: &mut impl Write) -> Result<()> {
        write!(
            writer,
            "{} | {} | {} |",
            self.proof_time_ms, self.insns, self.cells_used,
        )?;
        Ok(())
    }

    pub fn set_diff(&mut self, prev: &Self) {
        self.cells_used.diff = Some(self.cells_used.val - prev.cells_used.val);
        self.insns.diff = Some(self.insns.val - prev.insns.val);
        self.proof_time_ms.diff = Some(self.proof_time_ms.val - prev.proof_time_ms.val);
    }
}

impl AggregateMetrics {
    pub fn get_single_summary(&self, name: &str) -> Option<SingleSummaryMetrics> {
        let stats = self.by_group.get(name)?;
        // Any group must have proof_time, but may not have cells_used or cycles (e.g., halo2)
        let proof_time_ms = if let Some(proof_stats) = stats.get(PROOF_TIME_LABEL) {
            proof_stats.sum
        } else {
            // Note: execute_metered is outside any segment scope, so it should have sum = max = avg
            let execute_metered = stats
                .get(EXECUTE_METERED_TIME_LABEL)
                .map(|s| s.sum.val)
                .unwrap_or(0.0);
            let execute_preflight = stats
                .get(EXECUTE_PREFLIGHT_TIME_LABEL)
                .map(|s| s.sum.val)
                .unwrap_or(0.0);
            // If total_proof_time_ms is not available, compute it from components
            let trace_gen = stats
                .get(TRACE_GEN_TIME_LABEL)
                .map(|s| s.sum.val)
                .unwrap_or(0.0);
            let stark_prove = stats
                .get(PROVE_EXCL_TRACE_TIME_LABEL)
                .map(|s| s.sum.val)
                .unwrap_or(0.0);
            println!(
                "{} {} {} {}",
                execute_metered, execute_preflight, trace_gen, stark_prove
            );
            MdTableCell::new(
                execute_metered + execute_preflight + trace_gen + stark_prove,
                None,
            )
        };
        println!("{}", self.total_proof_time.val);
        let par_proof_time_ms = if let Some(proof_stats) = stats.get(PROOF_TIME_LABEL) {
            proof_stats.max
        } else {
            // Use the same computation for max
            let execute_metered = stats
                .get(EXECUTE_METERED_TIME_LABEL)
                .map(|s| s.max.val)
                .unwrap_or(0.0);
            let execute_preflight = stats
                .get(EXECUTE_PREFLIGHT_TIME_LABEL)
                .map(|s| s.max.val)
                .unwrap_or(0.0);
            let trace_gen = stats
                .get(TRACE_GEN_TIME_LABEL)
                .map(|s| s.max.val)
                .unwrap_or(0.0);
            let stark_prove = stats
                .get(PROVE_EXCL_TRACE_TIME_LABEL)
                .map(|s| s.max.val)
                .unwrap_or(0.0);
            MdTableCell::new(
                execute_metered + execute_preflight + trace_gen + stark_prove,
                None,
            )
        };
        let cells_used = stats
            .get(MAIN_CELLS_USED_LABEL)
            .map(|s| s.sum)
            .unwrap_or_default();
        let insns = stats
            .get(EXECUTE_PREFLIGHT_INSNS_LABEL)
            .map(|s| s.sum)
            .unwrap_or_default();
        Some(SingleSummaryMetrics {
            cells_used,
            insns,
            proof_time_ms,
            par_proof_time_ms,
        })
    }

    /// Returns `None` if no group for app is found.
    pub fn get_summary_row(&self, md_filename: &str) -> Option<SummaryRow> {
        let app_name = self.name();
        let app = self.get_single_summary(&app_name)?;
        let leaf = self.get_single_summary("leaf");
        let mut internals = Vec::new();
        let mut hgt = 0;
        while let Some(internal) = self.get_single_summary(&format!("internal.{hgt}")) {
            internals.push(internal);
            hgt += 1;
        }
        let root = self.get_single_summary("root");
        let halo2_outer = self.get_single_summary("halo2_outer");
        let halo2_wrapper = self.get_single_summary("halo2_wrapper");
        Some(SummaryRow {
            name: app_name.to_string(),
            md_filename: md_filename.to_string(),
            metrics: BenchSummaryMetrics {
                app,
                leaf,
                internals,
                root,
                halo2_outer,
                halo2_wrapper,
            },
        })
    }
}
