use std::{collections::HashMap, io::Write};

use eyre::Result;
use serde::{Deserialize, Serialize};

use crate::types::{BencherValue, BenchmarkOutput, Labels, MdTableCell, MetricDb};

type MetricName = String;
type MetricsByName = HashMap<MetricName, Vec<(f64, Labels)>>;

#[derive(Clone, Debug, Default)]
pub struct GroupedMetrics {
    /// "group" label => metrics with that "group" label, further grouped by metric name
    pub by_group: HashMap<String, MetricsByName>,
    pub ungrouped: MetricsByName,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AggregateMetrics {
    /// "group" label => metric aggregate statistics
    #[serde(flatten)]
    pub by_group: HashMap<String, HashMap<MetricName, Stats>>,
    /// In seconds
    pub total_proof_time: MdTableCell,
    /// In seconds
    pub total_par_proof_time: MdTableCell,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BencherAggregateMetrics {
    #[serde(flatten)]
    pub by_group: HashMap<String, HashMap<String, BencherValue>>,
    /// In seconds
    pub total_proof_time: BencherValue,
    /// In seconds
    pub total_par_proof_time: BencherValue,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stats {
    pub sum: MdTableCell,
    pub max: MdTableCell,
    pub min: MdTableCell,
    pub avg: MdTableCell,
    #[serde(skip)]
    pub count: usize,
}

impl Default for Stats {
    fn default() -> Self {
        Self::new()
    }
}

impl Stats {
    pub fn new() -> Self {
        Self {
            sum: MdTableCell::default(),
            max: MdTableCell::default(),
            min: MdTableCell::new(f64::MAX, None),
            avg: MdTableCell::default(),
            count: 0,
        }
    }
    pub fn push(&mut self, value: f64) {
        self.sum.val += value;
        self.count += 1;
        if value > self.max.val {
            self.max.val = value;
        }
        if value < self.min.val {
            self.min.val = value;
        }
    }

    pub fn finalize(&mut self) {
        assert!(self.count != 0);
        self.avg.val = self.sum.val / self.count as f64;
    }

    pub fn set_diff(&mut self, prev: &Self) {
        self.sum.diff = Some(self.sum.val - prev.sum.val);
        self.max.diff = Some(self.max.val - prev.max.val);
        self.min.diff = Some(self.min.val - prev.min.val);
        self.avg.diff = Some(self.avg.val - prev.avg.val);
    }
}

impl GroupedMetrics {
    pub fn new(db: &MetricDb, group_label_name: &str) -> Result<Self> {
        let mut by_group = HashMap::<String, MetricsByName>::new();
        let mut ungrouped = MetricsByName::new();
        for (labels, metrics) in db.flat_dict.iter() {
            let group_name = labels.get(group_label_name);
            if let Some(group_name) = group_name {
                let group_entry = by_group.entry(group_name.to_string()).or_default();
                let mut labels = labels.clone();
                labels.remove(group_label_name);
                for metric in metrics {
                    group_entry
                        .entry(metric.name.clone())
                        .or_default()
                        .push((metric.value, labels.clone()));
                }
            } else {
                for metric in metrics {
                    ungrouped
                        .entry(metric.name.clone())
                        .or_default()
                        .push((metric.value, labels.clone()));
                }
            }
        }
        Ok(Self {
            by_group,
            ungrouped,
        })
    }

    /// Validates that E1, metered, and preflight instruction counts all match each other
    fn validate_instruction_counts(group_summaries: &HashMap<MetricName, Stats>) {
        let e1_insns = group_summaries.get(EXECUTE_E1_INSNS_LABEL);
        let metered_insns = group_summaries.get(EXECUTE_METERED_INSNS_LABEL);
        let preflight_insns = group_summaries.get(EXECUTE_PREFLIGHT_INSNS_LABEL);

        if let (Some(e1_insns), Some(preflight_insns)) = (e1_insns, preflight_insns) {
            assert_eq!(e1_insns.sum.val as u64, preflight_insns.sum.val as u64);
        }
        if let (Some(e1_insns), Some(metered_insns)) = (e1_insns, metered_insns) {
            assert_eq!(e1_insns.sum.val as u64, metered_insns.sum.val as u64);
        }
        if let (Some(metered_insns), Some(preflight_insns)) = (metered_insns, preflight_insns) {
            assert_eq!(metered_insns.sum.val as u64, preflight_insns.sum.val as u64);
        }
    }

    pub fn aggregate(&self) -> AggregateMetrics {
        let by_group: HashMap<String, _> = self
            .by_group
            .iter()
            .map(|(group_name, metrics)| {
                let group_summaries: HashMap<MetricName, Stats> = metrics
                    .iter()
                    .map(|(metric_name, metrics)| {
                        let mut summary = Stats::new();
                        for (value, _) in metrics {
                            summary.push(*value);
                        }
                        summary.finalize();
                        (metric_name.clone(), summary)
                    })
                    .collect();

                if !group_name.contains("keygen") {
                    Self::validate_instruction_counts(&group_summaries);
                }

                (group_name.clone(), group_summaries)
            })
            .collect();
        let mut metrics = AggregateMetrics {
            by_group,
            ..Default::default()
        };
        metrics.compute_total();
        metrics
    }
}

// A hacky way to order the groups for display.
pub(crate) fn group_weight(name: &str) -> usize {
    let label_prefix = ["leaf", "internal", "root", "halo2_outer", "halo2_wrapper"];
    if name.contains("keygen") {
        return label_prefix.len() + 1;
    }
    for (i, prefix) in label_prefix.iter().enumerate().rev() {
        if name.starts_with(prefix) {
            return i + 1;
        }
    }
    0
}

impl AggregateMetrics {
    pub fn compute_total(&mut self) {
        let mut total_proof_time = MdTableCell::new(0.0, Some(0.0));
        let mut total_par_proof_time = MdTableCell::new(0.0, Some(0.0));
        for (group_name, metrics) in &self.by_group {
            let stats = metrics.get(PROOF_TIME_LABEL);
            let execute_metered_stats = metrics.get(EXECUTE_METERED_TIME_LABEL);
            let execute_e1_stats = metrics.get(EXECUTE_E1_TIME_LABEL);
            if stats.is_none() {
                continue;
            }
            let stats = stats.unwrap_or_else(|| {
                panic!("Missing proof time statistics for group '{}'", group_name)
            });
            let mut sum = stats.sum;
            let mut max = stats.max;
            // convert ms to s
            sum.val /= 1000.0;
            max.val /= 1000.0;
            if let Some(diff) = &mut sum.diff {
                *diff /= 1000.0;
            }
            if let Some(diff) = &mut max.diff {
                *diff /= 1000.0;
            }
            if !group_name.contains("keygen") {
                // Proving time in keygen group is dummy and not part of total.
                total_proof_time.val += sum.val;
                *total_proof_time
                    .diff
                    .as_mut()
                    .expect("total_proof_time.diff should be initialized") +=
                    sum.diff.unwrap_or(0.0);
                total_par_proof_time.val += max.val;
                *total_par_proof_time
                    .diff
                    .as_mut()
                    .expect("total_par_proof_time.diff should be initialized") +=
                    max.diff.unwrap_or(0.0);

                // Account for the serial execute_metered and execute_e1 for app outside of segments
                if group_name != "leaf"
                    && group_name != "root"
                    && group_name != "halo2_outer"
                    && group_name != "halo2_wrapper"
                    && !group_name.starts_with("internal")
                {
                    if let Some(execute_metered_stats) = execute_metered_stats {
                        // For metered metrics without segment labels, we just use the value
                        // directly Count is 1, so avg = sum = max = min =
                        // value
                        total_proof_time.val += execute_metered_stats.avg.val / 1000.0;
                        total_par_proof_time.val += execute_metered_stats.avg.val / 1000.0;
                        if let Some(diff) = execute_metered_stats.avg.diff {
                            *total_proof_time
                                .diff
                                .as_mut()
                                .expect("total_proof_time.diff should be initialized") +=
                                diff / 1000.0;
                            *total_par_proof_time
                                .diff
                                .as_mut()
                                .expect("total_par_proof_time.diff should be initialized") +=
                                diff / 1000.0;
                        }
                    }

                    if let Some(execute_e1_stats) = execute_e1_stats {
                        total_proof_time.val += execute_e1_stats.avg.val / 1000.0;
                        total_par_proof_time.val += execute_e1_stats.avg.val / 1000.0;
                        if let Some(diff) = execute_e1_stats.avg.diff {
                            *total_proof_time
                                .diff
                                .as_mut()
                                .expect("total_proof_time.diff should be initialized") +=
                                diff / 1000.0;
                            *total_par_proof_time
                                .diff
                                .as_mut()
                                .expect("total_par_proof_time.diff should be initialized") +=
                                diff / 1000.0;
                        }
                    }
                }
            }
        }
        self.total_proof_time = total_proof_time;
        self.total_par_proof_time = total_par_proof_time;
    }

    pub fn set_diff(&mut self, prev: &Self) {
        for (group_name, metrics) in self.by_group.iter_mut() {
            if let Some(prev_metrics) = prev.by_group.get(group_name) {
                for (metric_name, stats) in metrics.iter_mut() {
                    if let Some(prev_stats) = prev_metrics.get(metric_name) {
                        stats.set_diff(prev_stats);
                    }
                }
            }
        }
        self.compute_total();
    }

    pub fn to_vec(&self) -> Vec<(String, HashMap<MetricName, Stats>)> {
        let mut group_names: Vec<_> = self.by_group.keys().collect();
        group_names.sort_by(|a, b| {
            let a_wt = group_weight(a);
            let b_wt = group_weight(b);
            if a_wt == b_wt {
                a.cmp(b)
            } else {
                a_wt.cmp(&b_wt)
            }
        });
        group_names
            .into_iter()
            .map(|group_name| {
                let key = group_name.clone();
                let value = self
                    .by_group
                    .get(group_name)
                    .unwrap_or_else(|| {
                        panic!("Group '{}' should exist in by_group map", group_name)
                    })
                    .clone();
                (key, value)
            })
            .collect()
    }

    pub fn to_bencher_metrics(&self) -> BencherAggregateMetrics {
        let by_group = self
            .by_group
            .iter()
            .map(|(group_name, metrics)| {
                let metrics = metrics
                    .iter()
                    .filter(|(_, stats)| stats.avg.val.is_finite() && stats.sum.val.is_finite())
                    .flat_map(|(metric_name, stats)| {
                        [
                            (format!("{metric_name}::sum"), stats.sum.into()),
                            (
                                metric_name.clone(),
                                BencherValue {
                                    value: stats.avg.val,
                                    lower_value: Some(stats.min.val),
                                    upper_value: Some(stats.max.val),
                                },
                            ),
                        ]
                    })
                    .collect();
                (group_name.clone(), metrics)
            })
            .collect();
        let total_proof_time = self.total_proof_time.into();
        let total_par_proof_time = self.total_par_proof_time.into();
        BencherAggregateMetrics {
            by_group,
            total_proof_time,
            total_par_proof_time,
        }
    }

    pub fn write_markdown(&self, writer: &mut impl Write, metric_names: &[&str]) -> Result<()> {
        self.write_summary_markdown(writer)?;
        writeln!(writer)?;

        let metric_names = metric_names.to_vec();
        for (group_name, summaries) in self.to_vec() {
            writeln!(writer, "| {} |||||", group_name)?;
            writeln!(writer, "|:---|---:|---:|---:|---:|")?;
            writeln!(writer, "|metric|avg|sum|max|min|")?;
            let names = if metric_names.is_empty() {
                summaries.keys().map(|s| s.as_str()).collect()
            } else {
                metric_names.clone()
            };
            for metric_name in names {
                let summary = summaries.get(metric_name);
                if let Some(summary) = summary {
                    // Special handling for execute_metered metrics (not aggregated across segments
                    // in the app proof case)
                    if metric_name == EXECUTE_METERED_TIME_LABEL
                        && group_name != "leaf"
                        && group_name != "root"
                        && group_name != "halo2_outer"
                        && group_name != "halo2_wrapper"
                        && !group_name.starts_with("internal")
                    {
                        writeln!(
                            writer,
                            "| `{:<20}` | {:<10} | {:<10} | {:<10} | {:<10} |",
                            metric_name, summary.avg, "-", "-", "-",
                        )?;
                    } else if metric_name == EXECUTE_E1_INSN_MI_S_LABEL
                        || metric_name == EXECUTE_PREFLIGHT_INSN_MI_S_LABEL
                        || metric_name == EXECUTE_METERED_INSN_MI_S_LABEL
                    {
                        // skip sum because it is misleading
                        writeln!(
                            writer,
                            "| `{:<20}` | {:<10} | {:<10} | {:<10} | {:<10} |",
                            metric_name, summary.avg, "-", summary.max, summary.min,
                        )?;
                    } else {
                        writeln!(
                            writer,
                            "| `{:<20}` | {:<10} | {:<10} | {:<10} | {:<10} |",
                            metric_name, summary.avg, summary.sum, summary.max, summary.min,
                        )?;
                    }
                }
            }
            writeln!(writer)?;
        }
        writeln!(writer)?;

        Ok(())
    }

    fn write_summary_markdown(&self, writer: &mut impl Write) -> Result<()> {
        writeln!(
            writer,
            "| Summary | Proof Time (s) | Parallel Proof Time (s) |"
        )?;
        writeln!(writer, "|:---|---:|---:|")?;
        let mut rows = Vec::new();
        for (group_name, summaries) in self.to_vec() {
            if group_name.contains("keygen") {
                continue;
            }
            let stats = summaries.get(PROOF_TIME_LABEL);
            if stats.is_none() {
                continue;
            }
            let stats = stats.unwrap_or_else(|| {
                panic!("Missing proof time statistics for group '{}'", group_name)
            });
            let mut sum = stats.sum;
            let mut max = stats.max;
            // convert ms to s
            sum.val /= 1000.0;
            max.val /= 1000.0;
            if let Some(diff) = &mut sum.diff {
                *diff /= 1000.0;
            }
            if let Some(diff) = &mut max.diff {
                *diff /= 1000.0;
            }
            rows.push((group_name, sum, max));
        }
        writeln!(
            writer,
            "| Total | {} | {} |",
            self.total_proof_time, self.total_par_proof_time
        )?;
        for (group_name, proof_time, par_proof_time) in rows {
            writeln!(writer, "| {group_name} | {proof_time} | {par_proof_time} |")?;
        }
        writeln!(writer)?;
        Ok(())
    }

    pub fn name(&self) -> String {
        // A hacky way to determine the app name
        self.by_group
            .keys()
            .find(|k| group_weight(k) == 0)
            .unwrap_or_else(|| {
                self.by_group
                    .keys()
                    .next()
                    .expect("by_group should contain at least one group")
            })
            .clone()
    }
}

impl BenchmarkOutput {
    pub fn insert(&mut self, name: &str, metrics: BencherAggregateMetrics) {
        for (group_name, metrics) in metrics.by_group {
            self.by_name
                .entry(format!("{name}::{group_name}"))
                .or_default()
                .extend(metrics);
        }
        if let Some(e) = self.by_name.insert(
            name.to_owned(),
            HashMap::from_iter([
                ("total_proof_time".to_owned(), metrics.total_proof_time),
                (
                    "total_par_proof_time".to_owned(),
                    metrics.total_par_proof_time,
                ),
            ]),
        ) {
            panic!("Duplicate metric: {e:?}");
        }
    }
}

pub const PROOF_TIME_LABEL: &str = "total_proof_time_ms";
pub const MAIN_CELLS_USED_LABEL: &str = "main_cells_used";
pub const TOTAL_CELLS_USED_LABEL: &str = "total_cells_used";
pub const EXECUTE_E1_INSNS_LABEL: &str = "execute_e1_insns";
pub const EXECUTE_METERED_INSNS_LABEL: &str = "execute_metered_insns";
pub const EXECUTE_PREFLIGHT_INSNS_LABEL: &str = "execute_preflight_insns";
pub const EXECUTE_E1_TIME_LABEL: &str = "execute_e1_time_ms";
pub const EXECUTE_E1_INSN_MI_S_LABEL: &str = "execute_e1_insn_mi/s";
pub const EXECUTE_METERED_TIME_LABEL: &str = "execute_metered_time_ms";
pub const EXECUTE_METERED_INSN_MI_S_LABEL: &str = "execute_metered_insn_mi/s";
pub const EXECUTE_PREFLIGHT_TIME_LABEL: &str = "execute_preflight_time_ms";
pub const EXECUTE_PREFLIGHT_INSN_MI_S_LABEL: &str = "execute_preflight_insn_mi/s";
pub const TRACE_GEN_TIME_LABEL: &str = "trace_gen_time_ms";
pub const MEM_FIN_TIME_LABEL: &str = "memory_finalize_time_ms";
pub const BOUNDARY_FIN_TIME_LABEL: &str = "boundary_finalize_time_ms";
pub const MERKLE_FIN_TIME_LABEL: &str = "merkle_finalize_time_ms";
pub const PROVE_EXCL_TRACE_TIME_LABEL: &str = "stark_prove_excluding_trace_time_ms";

pub const VM_METRIC_NAMES: &[&str] = &[
    PROOF_TIME_LABEL,
    MAIN_CELLS_USED_LABEL,
    TOTAL_CELLS_USED_LABEL,
    EXECUTE_E1_TIME_LABEL,
    EXECUTE_E1_INSN_MI_S_LABEL,
    EXECUTE_METERED_TIME_LABEL,
    EXECUTE_METERED_INSN_MI_S_LABEL,
    EXECUTE_PREFLIGHT_INSNS_LABEL,
    EXECUTE_PREFLIGHT_TIME_LABEL,
    EXECUTE_PREFLIGHT_INSN_MI_S_LABEL,
    TRACE_GEN_TIME_LABEL,
    MEM_FIN_TIME_LABEL,
    BOUNDARY_FIN_TIME_LABEL,
    MERKLE_FIN_TIME_LABEL,
    PROVE_EXCL_TRACE_TIME_LABEL,
    "main_trace_commit_time_ms",
    "generate_perm_trace_time_ms",
    "perm_trace_commit_time_ms",
    "quotient_poly_compute_time_ms",
    "quotient_poly_commit_time_ms",
    "pcs_opening_time_ms",
];
