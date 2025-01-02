use std::{collections::HashMap, io::Write};

use eyre::Result;
use serde::{Deserialize, Serialize};

use crate::types::{Labels, MetricDb};

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
    /// "group" label => metric aggregate statitics
    #[serde(flatten)]
    pub by_group: HashMap<String, HashMap<MetricName, Stats>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stats {
    pub sum: f64,
    pub max: f64,
    pub min: f64,
    pub avg: f64,
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
            sum: 0.0,
            max: 0.0,
            min: f64::MAX,
            avg: 0.0,
            count: 0,
        }
    }
    pub fn push(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
        if value > self.max {
            self.max = value;
        }
        if value < self.min {
            self.min = value;
        }
    }

    pub fn finalize(&mut self) {
        assert!(self.count != 0);
        self.avg = self.sum / self.count as f64;
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
                (group_name.clone(), group_summaries)
            })
            .collect();
        AggregateMetrics { by_group }
    }
}

// A hacky way to order the groups for display.
pub(crate) fn group_weight(name: &str) -> usize {
    if name.starts_with("root") {
        3
    } else if name.starts_with("internal") {
        2
    } else if name.starts_with("leaf") {
        1
    } else {
        0
    }
}

impl AggregateMetrics {
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
                let value = self.by_group.get(group_name).unwrap().clone();
                (key, value)
            })
            .collect()
    }

    pub fn write_markdown(&self, writer: &mut impl Write, metric_names: &[&str]) -> Result<()> {
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
                    let [avg, sum, max, min] = [summary.avg, summary.sum, summary.max, summary.min]
                        .map(MetricDb::format_number);
                    writeln!(
                        writer,
                        "| `{:<20}` | {:<10} | {:<10} | {:<10} | {:<10} |",
                        metric_name, avg, sum, max, min,
                    )?;
                }
            }
            writeln!(writer)?;
        }
        writeln!(writer)?;

        Ok(())
    }
}

pub const PROOF_TIME_LABEL: &str = "total_proof_time_ms";
pub const CELLS_USED_LABEL: &str = "total_cells_used";
pub const CYCLES_LABEL: &str = "total_cycles";
pub const EXECUTE_TIME_LABEL: &str = "execute_time_ms";
pub const TRACE_GEN_TIME_LABEL: &str = "trace_gen_time_ms";
pub const PROVE_EXCL_TRACE_TIME_LABEL: &str = "stark_prove_excluding_trace_time_ms";

pub const VM_METRIC_NAMES: &[&str] = &[
    PROOF_TIME_LABEL,
    CELLS_USED_LABEL,
    CYCLES_LABEL,
    EXECUTE_TIME_LABEL,
    TRACE_GEN_TIME_LABEL,
    PROVE_EXCL_TRACE_TIME_LABEL,
    "main_trace_commit_time_ms",
    "generate_perm_trace_time_ms",
    "perm_trace_commit_time_ms",
    "quotient_poly_compute_time_ms",
    "quotient_poly_commit_time_ms",
    "pcs_opening_time_ms",
];
