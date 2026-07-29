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
    /// In seconds (infinite parallelism)
    pub total_par_proof_time: MdTableCell,
    /// Per-group infinite-parallel proof time in seconds
    #[serde(skip)]
    pub par_by_group: HashMap<String, MdTableCell>,
    /// Per-group bounded parallel proof time in seconds
    #[serde(skip)]
    pub bounded_par_by_group: HashMap<String, MdTableCell>,
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
    #[serde(skip)]
    pub phase: Option<String>,
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
            phase: None,
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
                let group_entry = by_group
                    .entry(canonical_group_name(group_name).to_string())
                    .or_default();
                let mut labels = labels.clone();
                labels.remove(group_label_name);
                for metric in metrics {
                    group_entry
                        .entry(canonical_metric_name(&metric.name).to_string())
                        .or_default()
                        .push((metric.value, labels.clone()));
                }
            } else {
                for metric in metrics {
                    ungrouped
                        .entry(canonical_metric_name(&metric.name).to_string())
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

    /// Validates that execution-mode instruction counts all match each other.
    fn validate_instruction_counts(group_summaries: &HashMap<MetricName, Stats>) {
        let available = [
            EXECUTE_PURE_INSNS_LABEL,
            EXECUTE_METERED_INSNS_LABEL,
            EXECUTE_METERED_COST_INSNS_LABEL,
            EXECUTE_PREFLIGHT_INSNS_LABEL,
        ]
        .into_iter()
        .filter_map(|name| {
            group_summaries
                .get(name)
                .map(|stats| (name, stats.sum.val as u64))
        });
        let mut available = available.peekable();
        if let Some((expected_name, expected)) = available.next() {
            for (name, actual) in available {
                assert_eq!(
                    actual, expected,
                    "instruction count mismatch between {expected_name} and {name}"
                );
            }
        }
    }

    pub fn aggregate(&self, num_parallel: usize) -> AggregateMetrics {
        let by_group: HashMap<String, _> = self
            .by_group
            .iter()
            .map(|(group_name, metrics)| {
                let mut group_summaries: HashMap<MetricName, Stats> = metrics
                    .iter()
                    .map(|(metric_name, metrics)| {
                        let mut summary = Stats::new();
                        for (value, labels) in metrics {
                            summary.push(*value);
                            // Extract phase from labels if present
                            if summary.phase.is_none() {
                                if let Some(phase) = labels.get("phase") {
                                    summary.phase = Some(phase.to_string());
                                }
                            }
                        }
                        summary.finalize();
                        (metric_name.clone(), summary)
                    })
                    .collect();

                let preflight_rate = group_summaries
                    .get(EXECUTE_PREFLIGHT_INSNS_LABEL)
                    .zip(group_summaries.get(EXECUTE_PREFLIGHT_TIME_LABEL))
                    .and_then(|(insns, time)| {
                        (time.sum.val > 0.0).then_some(insns.sum.val / time.sum.val / 1000.0)
                    });
                if let (Some(rate), Some(summary)) = (
                    preflight_rate,
                    group_summaries.get_mut(EXECUTE_PREFLIGHT_INSN_MI_S_LABEL),
                ) {
                    summary.avg.val = rate;
                }

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
        metrics.par_by_group = self
            .compute_bounded_par_times(usize::MAX, &metrics.by_group)
            .into_iter()
            .map(|(k, v)| (k, MdTableCell::new(v, Some(0.0))))
            .collect();
        metrics.bounded_par_by_group = self
            .compute_bounded_par_times(num_parallel, &metrics.by_group)
            .into_iter()
            .map(|(k, v)| (k, MdTableCell::new(v, Some(0.0))))
            .collect();
        metrics.compute_total();

        metrics
    }

    /// Compute per-group parallel proof time with bounded parallelism.
    fn compute_bounded_par_times(
        &self,
        num_parallel: usize,
        stats_by_group: &HashMap<String, HashMap<MetricName, Stats>>,
    ) -> HashMap<String, f64> {
        let mut per_group = HashMap::new();

        for (group_name, metrics) in &self.by_group {
            if group_name.contains("keygen") {
                continue;
            }

            let mut group_time = 0.0;

            // Add serial execution time for app_proof groups
            if is_app_proof_group(group_name) {
                if let Some(stats) = stats_by_group.get(group_name) {
                    if let Some(metered) = stats.get(EXECUTE_METERED_TIME_LABEL) {
                        group_time += metered.avg.val / 1000.0;
                    }
                    if let Some(pure) = stats.get(EXECUTE_PURE_TIME_LABEL) {
                        group_time += pure.avg.val / 1000.0;
                    }
                }
            }

            if metrics.contains_key(PROOF_TIME_LABEL) {
                // Application segments are produced by one serial preflight driver.
                // Recursion proofs are independent tasks, so their complete proof
                // durations, including preflight, remain parallelizable.
                let (proof_times_ms, serial_preflight_ms) = if is_app_proof_group(group_name) {
                    parallel_proof_times_ms(metrics)
                } else {
                    (
                        metrics[PROOF_TIME_LABEL]
                            .iter()
                            .map(|(value, _)| *value)
                            .collect(),
                        0.0,
                    )
                };
                group_time += serial_preflight_ms / 1000.0;
                let times_s: Vec<f64> = proof_times_ms.into_iter().map(|ms| ms / 1000.0).collect();
                group_time += schedule_parallel(&times_s, num_parallel);
            }

            per_group.insert(group_name.clone(), group_time);
        }

        per_group
    }
}

/// Round-robin assignment: proof i -> slot i % num_parallel. Returns max slot time.
fn schedule_parallel(proof_times: &[f64], num_parallel: usize) -> f64 {
    if proof_times.is_empty() || num_parallel == 0 {
        return 0.0;
    }

    let mut slot_times = vec![0.0_f64; num_parallel.min(proof_times.len())];
    for (i, duration) in proof_times.iter().enumerate() {
        let slot = i % slot_times.len();
        slot_times[slot] += duration;
    }
    slot_times.iter().cloned().fold(0.0_f64, f64::max)
}

/// Returns parallelizable proof time per segment and the preflight time that
/// must remain serial. Samples are paired by their existing segment label.
fn parallel_proof_times_ms(metrics: &MetricsByName) -> (Vec<f64>, f64) {
    let proof_times = metrics
        .get(PROOF_TIME_LABEL)
        .expect("proof times must exist before parallel projection");
    let Some(preflight_times) = metrics.get(EXECUTE_PREFLIGHT_TIME_LABEL) else {
        return (proof_times.iter().map(|(value, _)| *value).collect(), 0.0);
    };

    let mut preflight_by_segment = HashMap::with_capacity(preflight_times.len());
    for (value, labels) in preflight_times {
        let segment = labels
            .get("segment")
            .expect("preflight execution metric is missing its segment label");
        assert!(
            preflight_by_segment
                .insert(segment.to_string(), *value)
                .is_none(),
            "duplicate preflight execution metric for segment {segment}"
        );
    }

    let mut paired = Vec::with_capacity(proof_times.len());
    for (proof_time, labels) in proof_times {
        let segment = labels
            .get("segment")
            .expect("total proof metric is missing its segment label");
        let preflight_time = preflight_by_segment.remove(segment).unwrap_or_else(|| {
            panic!("total proof segment {segment} has no preflight execution metric")
        });
        assert!(
            preflight_time <= *proof_time,
            "preflight execution exceeds total proof time for segment {segment}"
        );
        paired.push((segment.to_string(), proof_time - preflight_time));
    }
    assert!(
        preflight_by_segment.is_empty(),
        "preflight execution metric has no matching total proof segment"
    );
    paired.sort_unstable_by(
        |(a, _), (b, _)| match (a.parse::<u64>(), b.parse::<u64>()) {
            (Ok(a), Ok(b)) => a.cmp(&b),
            _ => a.cmp(b),
        },
    );

    let serial_preflight_ms = preflight_times.iter().map(|(value, _)| value).sum();
    (
        paired.into_iter().map(|(_, value)| value).collect(),
        serial_preflight_ms,
    )
}

fn is_app_proof_group(name: &str) -> bool {
    name != "leaf"
        && name != "root"
        && name != "halo2_outer"
        && name != "halo2_wrapper"
        && !name.starts_with("internal")
}

fn projection_diff_is_compatible(
    current: &HashMap<String, HashMap<MetricName, Stats>>,
    previous: &HashMap<String, HashMap<MetricName, Stats>>,
    group_name: &str,
) -> bool {
    let has_preflight = |groups: &HashMap<String, HashMap<MetricName, Stats>>| {
        groups
            .get(group_name)
            .is_some_and(|metrics| metrics.contains_key(EXECUTE_PREFLIGHT_TIME_LABEL))
    };
    has_preflight(current) == has_preflight(previous)
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
        let mut parallel_projection_diff_is_compatible = true;
        for (group_name, metrics) in &self.by_group {
            let stats = metrics.get(PROOF_TIME_LABEL);
            let execute_metered_stats = metrics.get(EXECUTE_METERED_TIME_LABEL);
            let execute_pure_stats = metrics.get(EXECUTE_PURE_TIME_LABEL);
            if stats.is_none() {
                continue;
            }
            let stats = stats.unwrap_or_else(|| {
                panic!("Missing proof time statistics for group '{group_name}'")
            });
            let mut sum = stats.sum;
            let projected_parallel = self.par_by_group.get(group_name);
            if projected_parallel.is_some_and(|parallel| parallel.diff.is_none()) {
                parallel_projection_diff_is_compatible = false;
            }
            let mut max = projected_parallel.copied().unwrap_or(stats.max);
            // convert ms to s
            sum.val /= 1000.0;
            if projected_parallel.is_none() {
                max.val /= 1000.0;
            }
            if let Some(diff) = &mut sum.diff {
                *diff /= 1000.0;
            }
            if projected_parallel.is_none() {
                if let Some(diff) = &mut max.diff {
                    *diff /= 1000.0;
                }
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

                // Account for the serial execute_metered and execute_pure for app outside of
                // segments
                if is_app_proof_group(group_name) {
                    if let Some(execute_metered_stats) = execute_metered_stats {
                        // For metered metrics without segment labels, we just use the value
                        // directly Count is 1, so avg = sum = max = min =
                        // value
                        total_proof_time.val += execute_metered_stats.avg.val / 1000.0;
                        if projected_parallel.is_none() {
                            total_par_proof_time.val += execute_metered_stats.avg.val / 1000.0;
                        }
                        if let Some(diff) = execute_metered_stats.avg.diff {
                            *total_proof_time
                                .diff
                                .as_mut()
                                .expect("total_proof_time.diff should be initialized") +=
                                diff / 1000.0;
                            if projected_parallel.is_none() {
                                *total_par_proof_time
                                    .diff
                                    .as_mut()
                                    .expect("total_par_proof_time.diff should be initialized") +=
                                    diff / 1000.0;
                            }
                        }
                    }

                    if let Some(execute_pure_stats) = execute_pure_stats {
                        total_proof_time.val += execute_pure_stats.avg.val / 1000.0;
                        if projected_parallel.is_none() {
                            total_par_proof_time.val += execute_pure_stats.avg.val / 1000.0;
                        }
                        if let Some(diff) = execute_pure_stats.avg.diff {
                            *total_proof_time
                                .diff
                                .as_mut()
                                .expect("total_proof_time.diff should be initialized") +=
                                diff / 1000.0;
                            if projected_parallel.is_none() {
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
        }
        if !parallel_projection_diff_is_compatible {
            total_par_proof_time.diff = None;
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
        for (group_name, bounded) in self.bounded_par_by_group.iter_mut() {
            if projection_diff_is_compatible(&self.by_group, &prev.by_group, group_name) {
                if let Some(prev_bounded) = prev.bounded_par_by_group.get(group_name) {
                    bounded.diff = Some(bounded.val - prev_bounded.val);
                }
            } else {
                bounded.diff = None;
            }
        }
        for (group_name, parallel) in self.par_by_group.iter_mut() {
            if projection_diff_is_compatible(&self.by_group, &prev.by_group, group_name) {
                if let Some(prev_parallel) = prev.par_by_group.get(group_name) {
                    parallel.diff = Some(parallel.val - prev_parallel.val);
                }
            } else {
                parallel.diff = None;
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
                    .unwrap_or_else(|| panic!("Group '{group_name}' should exist in by_group map"))
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

    pub fn write_markdown(
        &self,
        writer: &mut impl Write,
        metric_names: &[&str],
        num_parallel: usize,
    ) -> Result<()> {
        self.write_summary_markdown(writer, num_parallel)?;
        writeln!(writer)?;

        let metric_names = metric_names.to_vec();
        for (group_name, summaries) in self.to_vec() {
            if group_name.contains("keygen") {
                continue;
            }

            let names: Vec<&str> = if metric_names.is_empty() {
                summaries.keys().map(|s| s.as_str()).collect()
            } else {
                metric_names.clone()
            };
            let names: Vec<&str> = names
                .into_iter()
                .filter(|name| summaries.contains_key(*name))
                .collect();
            if names.is_empty() {
                continue;
            }

            writeln!(writer, "| {group_name} |||||")?;
            writeln!(writer, "|:---|---:|---:|---:|---:|")?;
            writeln!(writer, "|metric|avg|sum|max|min|")?;

            // Group metrics by phase
            let get_phase = |name: &str| -> Option<&str> {
                summaries.get(name).and_then(|stats| stats.phase.as_deref())
            };

            // Collect unique phases (preserving order: uncategorized first, then by phase)
            let mut phases: Vec<Option<&str>> = vec![None];
            for name in &names {
                if let Some(phase) = get_phase(name) {
                    if !phases.contains(&Some(phase)) {
                        phases.push(Some(phase));
                    }
                }
            }

            // Write metrics grouped by phase
            for phase in &phases {
                let phase_names: Vec<&str> = names
                    .iter()
                    .filter(|name| get_phase(name) == *phase)
                    .copied()
                    .collect();

                if phase_names.is_empty() {
                    continue;
                }

                // Write separator for non-default phases
                if let Some(p) = phase {
                    let label = p[0..1].to_uppercase() + &p[1..]; // Capitalize
                    writeln!(writer, "| __{label}__ |||||")?;
                }

                for metric_name in &phase_names {
                    self.write_metric_row(writer, &group_name, &summaries, metric_name)?;
                }
            }

            writeln!(writer)?;
        }
        writeln!(writer)?;

        Ok(())
    }

    fn write_metric_row(
        &self,
        writer: &mut impl Write,
        group_name: &str,
        summaries: &HashMap<MetricName, Stats>,
        metric_name: &str,
    ) -> Result<()> {
        let summary = summaries.get(metric_name);
        if let Some(summary) = summary {
            // Special handling for execute_metered metrics (not aggregated across segments
            // in the app proof case)
            if (metric_name == EXECUTE_METERED_TIME_LABEL
                || metric_name == EXECUTE_METERED_INSNS_LABEL)
                && is_app_proof_group(group_name)
            {
                writeln!(
                    writer,
                    "| `{:<20}` | {:<10} | {:<10} | {:<10} | {:<10} |",
                    metric_name, summary.avg, "-", "-", "-",
                )?;
            } else if metric_name == EXECUTE_PURE_INSN_MI_S_LABEL
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
        Ok(())
    }

    fn write_summary_markdown(&self, writer: &mut impl Write, num_parallel: usize) -> Result<()> {
        writeln!(
            writer,
            "| Summary | Proof Time (s) | Parallel Proof Time (s) | Parallel Proof Time ({} provers) (s) |",
            num_parallel
        )?;
        writeln!(writer, "|:---|---:|---:|---:|")?;
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
                panic!("Missing proof time statistics for group '{group_name}'")
            });
            let mut sum = stats.sum;
            let projected_parallel = self.par_by_group.get(&group_name);
            let mut max = projected_parallel.copied().unwrap_or(stats.max);
            // convert ms to s
            sum.val /= 1000.0;
            if projected_parallel.is_none() {
                max.val /= 1000.0;
            }
            if let Some(diff) = &mut sum.diff {
                *diff /= 1000.0;
            }
            if projected_parallel.is_none() {
                if let Some(diff) = &mut max.diff {
                    *diff /= 1000.0;
                }
            }
            // Add serial execution time for app_proof groups
            if is_app_proof_group(&group_name) {
                if let Some(metered) = summaries.get(EXECUTE_METERED_TIME_LABEL) {
                    sum.val += metered.avg.val / 1000.0;
                    if projected_parallel.is_none() {
                        max.val += metered.avg.val / 1000.0;
                    }
                }
                if let Some(pure) = summaries.get(EXECUTE_PURE_TIME_LABEL) {
                    sum.val += pure.avg.val / 1000.0;
                    if projected_parallel.is_none() {
                        max.val += pure.avg.val / 1000.0;
                    }
                }
            }
            rows.push((group_name, sum, max));
        }
        let mut total_bounded = MdTableCell::new(0.0, Some(0.0));
        for cell in self.bounded_par_by_group.values() {
            total_bounded.val += cell.val;
            if let Some(diff) = cell.diff {
                if let Some(total) = total_bounded.diff.as_mut() {
                    *total += diff;
                }
            } else {
                total_bounded.diff = None;
            }
        }
        writeln!(
            writer,
            "| Total | {} | {} | {} |",
            self.total_proof_time, self.total_par_proof_time, total_bounded
        )?;
        for (group_name, proof_time, par_proof_time) in rows {
            let bounded = self
                .bounded_par_by_group
                .get(&group_name)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string());
            writeln!(
                writer,
                "| {group_name} | {proof_time} | {par_proof_time} | {bounded} |"
            )?;
        }
        writeln!(writer)?;
        Ok(())
    }

    pub fn name(&self) -> Option<String> {
        // A hacky way to determine the app name
        let name = self
            .by_group
            .keys()
            .find(|k| group_weight(k) == 0)
            .or_else(|| self.by_group.keys().next())
            .cloned();
        if name.is_none() {
            eprintln!("Warning: no group found to determine app name; by_group is empty");
        }
        name
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
pub const EXECUTE_PURE_INSNS_LABEL: &str = "execute_pure_insns";
pub const EXECUTE_METERED_INSNS_LABEL: &str = "execute_metered_insns";
pub const EXECUTE_METERED_COST_INSNS_LABEL: &str = "execute_metered_cost_insns";
pub const EXECUTE_PREFLIGHT_INSNS_LABEL: &str = "execute_preflight_insns";
pub const EXECUTE_PURE_TIME_LABEL: &str = "execute_pure_time_ms";
pub const EXECUTE_PURE_INSN_MI_S_LABEL: &str = "execute_pure_insn_mi/s";
pub const EXECUTE_METERED_TIME_LABEL: &str = "execute_metered_time_ms";
pub const EXECUTE_METERED_INSN_MI_S_LABEL: &str = "execute_metered_insn_mi/s";
pub const EXECUTE_PREFLIGHT_TIME_LABEL: &str = "execute_preflight_time_ms";
pub const EXECUTE_PREFLIGHT_INSN_MI_S_LABEL: &str = "execute_preflight_insn_mi/s";
pub const EXECUTE_PREFLIGHT_INTERVALS_LABEL: &str = "execute_preflight_intervals";
pub const EXECUTE_PREFLIGHT_RESIDUALS_LABEL: &str = "execute_preflight_residuals";
pub const EXECUTE_PREFLIGHT_TRANSCRIPT_BYTES_LABEL: &str = "execute_preflight_transcript_bytes";
pub const COMPILE_PURE_TIME_LABEL: &str = "compile_pure_time_ms";
pub const COMPILE_METERED_TIME_LABEL: &str = "compile_metered_time_ms";
pub const COMPILE_METERED_SEGMENT_TIME_LABEL: &str = "compile_metered_segment_time_ms";
pub const COMPILE_METERED_COST_TIME_LABEL: &str = "compile_metered_cost_time_ms";
pub const COMPILE_PREFLIGHT_TIME_LABEL: &str = "compile_preflight_time_ms";
pub const PREPARE_PREFLIGHT_TIME_LABEL: &str = "prepare_preflight_time_ms";
pub const UPLOAD_PREFLIGHT_PROGRAM_TIME_LABEL: &str = "upload_preflight_program_time_ms";
pub const APP_PROVE_TIME_LABEL: &str = "app_prove_time_ms";
pub const POSTFLIGHT_TIME_LABEL: &str = "postflight_time_ms";
pub const POSTFLIGHT_REPLAY_COUNT_TIME_LABEL: &str = "postflight_replay_count_time_ms";
pub const POSTFLIGHT_REPLAY_EMIT_TIME_LABEL: &str = "postflight_replay_emit_time_ms";
pub const POSTFLIGHT_MEMORY_CHRONOLOGY_TIME_LABEL: &str = "postflight_memory_chronology_time_ms";
pub const POSTFLIGHT_PROGRAM_INDEX_TIME_LABEL: &str = "postflight_program_index_time_ms";
pub const TRACE_GEN_TIME_LABEL: &str = "trace_gen_time_ms";
pub const GENERATE_BLOB_TIME_LABEL: &str = "generate_blob_total_time_ms";
pub const SET_INITIAL_MEMORY_TIME_LABEL: &str = "set_initial_memory_time_ms";
pub const MEM_FIN_TIME_LABEL: &str = "memory_finalize_time_ms";
pub const BOUNDARY_FIN_TIME_LABEL: &str = "boundary_finalize_time_ms";
pub const MERKLE_FIN_TIME_LABEL: &str = "merkle_finalize_time_ms";
pub const PROVE_EXCL_TRACE_TIME_LABEL: &str = "stark_prove_excluding_trace_time_ms";

pub const HALO2_VERIFIER_K_LABEL: &str = "halo2_verifier_k";
pub const HALO2_WRAPPER_K_LABEL: &str = "halo2_wrapper_k";

fn canonical_group_name(name: &str) -> &str {
    if [
        "reth.prove_app.block_",
        "reth.prove_app_rvr.block_",
        "reth.prove_root.block_",
        "reth.prove_evm.block_",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
    {
        "app_proof"
    } else {
        name
    }
}

fn canonical_metric_name(name: &str) -> &str {
    match name {
        "prepare_rvr_checkpoint_time_ms" | "prepare_rvr_preflight_time_ms" => {
            PREPARE_PREFLIGHT_TIME_LABEL
        }
        "compile_checkpoint_preflight_time_ms" => COMPILE_PREFLIGHT_TIME_LABEL,
        "upload_checkpoint_program_time_ms" | "upload_postflight_program_time_ms" => {
            UPLOAD_PREFLIGHT_PROGRAM_TIME_LABEL
        }
        "app_prove_rvr_checkpoint_time_ms" => APP_PROVE_TIME_LABEL,
        "expand_checkpoint_replay_time_ms" => POSTFLIGHT_TIME_LABEL,
        "execute_checkpoint_preflight_insns" => EXECUTE_PREFLIGHT_INSNS_LABEL,
        "execute_preflight_checkpoints" | "execute_checkpoint_preflight_checkpoints" => {
            EXECUTE_PREFLIGHT_INTERVALS_LABEL
        }
        "execute_checkpoint_preflight_residuals" => EXECUTE_PREFLIGHT_RESIDUALS_LABEL,
        "execute_checkpoint_preflight_transcript_bytes" => EXECUTE_PREFLIGHT_TRANSCRIPT_BYTES_LABEL,
        "execute_checkpoint_preflight_time_ms" => EXECUTE_PREFLIGHT_TIME_LABEL,
        "execute_checkpoint_preflight_insn_mi/s" => EXECUTE_PREFLIGHT_INSN_MI_S_LABEL,
        _ => name,
    }
}

pub const AGGREGATED_METRIC_NAMES: &[&str] = &[
    PROOF_TIME_LABEL,
    MAIN_CELLS_USED_LABEL,
    TOTAL_CELLS_USED_LABEL,
    COMPILE_PURE_TIME_LABEL,
    COMPILE_METERED_TIME_LABEL,
    COMPILE_METERED_SEGMENT_TIME_LABEL,
    COMPILE_METERED_COST_TIME_LABEL,
    COMPILE_PREFLIGHT_TIME_LABEL,
    EXECUTE_PURE_TIME_LABEL,
    EXECUTE_PURE_INSN_MI_S_LABEL,
    EXECUTE_METERED_TIME_LABEL,
    EXECUTE_METERED_INSNS_LABEL,
    EXECUTE_METERED_COST_INSNS_LABEL,
    EXECUTE_METERED_INSN_MI_S_LABEL,
    SET_INITIAL_MEMORY_TIME_LABEL,
    EXECUTE_PREFLIGHT_INSNS_LABEL,
    EXECUTE_PREFLIGHT_TIME_LABEL,
    EXECUTE_PREFLIGHT_INSN_MI_S_LABEL,
    POSTFLIGHT_TIME_LABEL,
    POSTFLIGHT_REPLAY_COUNT_TIME_LABEL,
    POSTFLIGHT_REPLAY_EMIT_TIME_LABEL,
    POSTFLIGHT_MEMORY_CHRONOLOGY_TIME_LABEL,
    POSTFLIGHT_PROGRAM_INDEX_TIME_LABEL,
    TRACE_GEN_TIME_LABEL,
    GENERATE_BLOB_TIME_LABEL,
    MEM_FIN_TIME_LABEL,
    BOUNDARY_FIN_TIME_LABEL,
    MERKLE_FIN_TIME_LABEL,
    PROVE_EXCL_TRACE_TIME_LABEL,
    "prover.main_trace_commit_time_ms",
    "prover.rap_constraints_time_ms",
    "prover.openings_time_ms",
    "prover.rap_constraints.logup_gkr_time_ms",
    "prover.rap_constraints.round0_time_ms",
    "prover.rap_constraints.mle_rounds_time_ms",
    "prover.openings.stacked_reduction_time_ms",
    "prover.openings.stacked_reduction.round0_time_ms",
    "prover.openings.stacked_reduction.mle_rounds_time_ms",
    "prover.openings.whir_time_ms",
    HALO2_VERIFIER_K_LABEL,
    HALO2_WRAPPER_K_LABEL,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_reth_app_groups_use_the_canonical_name() {
        for group in [
            "reth.prove_app.block_23992138",
            "reth.prove_app_rvr.block_23992138",
            "reth.prove_root.block_23992138",
            "reth.prove_evm.block_23992138",
        ] {
            assert_eq!(canonical_group_name(group), "app_proof");
        }
        assert_eq!(canonical_group_name("root"), "root");
        assert_eq!(canonical_group_name("internal_0"), "internal_0");
    }

    #[test]
    fn legacy_preflight_metrics_use_canonical_phase_names() {
        assert_eq!(
            canonical_metric_name("execute_checkpoint_preflight_time_ms"),
            EXECUTE_PREFLIGHT_TIME_LABEL
        );
        assert_eq!(
            canonical_metric_name("execute_preflight_checkpoints"),
            EXECUTE_PREFLIGHT_INTERVALS_LABEL
        );
        assert_eq!(
            canonical_metric_name("expand_checkpoint_replay_time_ms"),
            POSTFLIGHT_TIME_LABEL
        );
        assert_eq!(
            canonical_metric_name("upload_checkpoint_program_time_ms"),
            UPLOAD_PREFLIGHT_PROGRAM_TIME_LABEL
        );
    }

    fn labels(segment: Option<usize>) -> Labels {
        Labels(
            segment
                .map(|segment| vec![("segment".to_string(), segment.to_string())])
                .unwrap_or_default(),
        )
    }

    fn grouped(metrics: MetricsByName) -> GroupedMetrics {
        GroupedMetrics {
            by_group: HashMap::from([("app".to_string(), metrics)]),
            ungrouped: HashMap::new(),
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }

    fn preflight_timing_metrics() -> MetricsByName {
        HashMap::from([
            (
                PROOF_TIME_LABEL.to_string(),
                vec![
                    (80.0, labels(Some(2))),
                    (100.0, labels(Some(0))),
                    (120.0, labels(Some(1))),
                ],
            ),
            (
                EXECUTE_PREFLIGHT_TIME_LABEL.to_string(),
                vec![
                    (20.0, labels(Some(1))),
                    (30.0, labels(Some(2))),
                    (10.0, labels(Some(0))),
                ],
            ),
            (
                EXECUTE_METERED_TIME_LABEL.to_string(),
                vec![(50.0, labels(None))],
            ),
        ])
    }

    #[test]
    fn preflight_execution_remains_serial_in_parallel_projections() {
        let metrics = preflight_timing_metrics();

        let aggregate = grouped(metrics).aggregate(2);

        // Sequential: 50 ms metered + 300 ms segment totals.
        assert_close(aggregate.total_proof_time.val, 0.35);
        // Infinite parallel: 50 ms metered + 60 ms preflight execution +
        // max(90, 100, 50) ms remaining segment work.
        assert_close(aggregate.total_par_proof_time.val, 0.21);
        // Two provers: 50 ms metered + 60 ms preflight execution +
        // max(90 + 50, 100) ms remaining segment work.
        assert_close(aggregate.bounded_par_by_group["app"].val, 0.25);
        assert_eq!(
            aggregate.by_group["app"][EXECUTE_PREFLIGHT_TIME_LABEL]
                .sum
                .val,
            60.0
        );
    }

    #[test]
    fn recursion_preflight_remains_part_of_each_parallel_proof() {
        let indexed_labels = |idx: usize| Labels(vec![("idx".to_string(), idx.to_string())]);
        let metrics = HashMap::from([
            (
                PROOF_TIME_LABEL.to_string(),
                vec![
                    (100.0, indexed_labels(0)),
                    (120.0, indexed_labels(1)),
                    (80.0, indexed_labels(2)),
                ],
            ),
            (
                EXECUTE_PREFLIGHT_TIME_LABEL.to_string(),
                vec![
                    (10.0, indexed_labels(0)),
                    (20.0, indexed_labels(1)),
                    (30.0, indexed_labels(2)),
                ],
            ),
        ]);
        let grouped = GroupedMetrics {
            by_group: HashMap::from([("leaf".to_string(), metrics)]),
            ungrouped: HashMap::new(),
        };

        let aggregate = grouped.aggregate(2);

        assert_close(aggregate.total_proof_time.val, 0.3);
        assert_close(aggregate.total_par_proof_time.val, 0.12);
        assert_close(aggregate.bounded_par_by_group["leaf"].val, 0.18);
    }

    #[test]
    fn preflight_throughput_uses_total_instructions_and_time() {
        let metrics = HashMap::from([
            (
                EXECUTE_PREFLIGHT_INSNS_LABEL.to_string(),
                vec![
                    (1_000_000.0, labels(Some(0))),
                    (1_000_000.0, labels(Some(1))),
                ],
            ),
            (
                EXECUTE_PREFLIGHT_TIME_LABEL.to_string(),
                vec![(1.0, labels(Some(0))), (9.0, labels(Some(1)))],
            ),
            (
                EXECUTE_PREFLIGHT_INSN_MI_S_LABEL.to_string(),
                vec![
                    (1_000.0, labels(Some(0))),
                    (1_000_000.0 / 9_000.0, labels(Some(1))),
                ],
            ),
        ]);

        let aggregate = grouped(metrics).aggregate(1);
        let throughput = &aggregate.by_group["app"][EXECUTE_PREFLIGHT_INSN_MI_S_LABEL];

        assert_close(throughput.avg.val, 200.0);
        assert_close(throughput.max.val, 1_000.0);
        assert_close(throughput.min.val, 1_000_000.0 / 9_000.0);
    }

    #[test]
    fn report_orders_frontend_phases_without_overlapping_wrappers() {
        let one = |segment| vec![(1.0, labels(segment))];
        let metrics = HashMap::from([
            (PROOF_TIME_LABEL.to_string(), one(Some(0))),
            (COMPILE_PREFLIGHT_TIME_LABEL.to_string(), one(None)),
            (PREPARE_PREFLIGHT_TIME_LABEL.to_string(), one(None)),
            (UPLOAD_PREFLIGHT_PROGRAM_TIME_LABEL.to_string(), one(None)),
            (APP_PROVE_TIME_LABEL.to_string(), one(None)),
            (EXECUTE_METERED_TIME_LABEL.to_string(), one(None)),
            (SET_INITIAL_MEMORY_TIME_LABEL.to_string(), one(Some(0))),
            (EXECUTE_PREFLIGHT_TIME_LABEL.to_string(), one(Some(0))),
            (EXECUTE_PREFLIGHT_INTERVALS_LABEL.to_string(), one(None)),
            (EXECUTE_PREFLIGHT_RESIDUALS_LABEL.to_string(), one(None)),
            (
                EXECUTE_PREFLIGHT_TRANSCRIPT_BYTES_LABEL.to_string(),
                one(None),
            ),
            (POSTFLIGHT_TIME_LABEL.to_string(), one(Some(0))),
            (
                POSTFLIGHT_MEMORY_CHRONOLOGY_TIME_LABEL.to_string(),
                one(Some(0)),
            ),
            (TRACE_GEN_TIME_LABEL.to_string(), one(Some(0))),
            (PROVE_EXCL_TRACE_TIME_LABEL.to_string(), one(Some(0))),
        ]);
        let aggregate = grouped(metrics).aggregate(1);
        let mut markdown = Vec::new();

        aggregate
            .write_markdown(&mut markdown, AGGREGATED_METRIC_NAMES, 1)
            .unwrap();
        let markdown = String::from_utf8(markdown).unwrap();

        let ordered = [
            COMPILE_PREFLIGHT_TIME_LABEL,
            EXECUTE_METERED_TIME_LABEL,
            SET_INITIAL_MEMORY_TIME_LABEL,
            EXECUTE_PREFLIGHT_TIME_LABEL,
            POSTFLIGHT_TIME_LABEL,
            POSTFLIGHT_MEMORY_CHRONOLOGY_TIME_LABEL,
            TRACE_GEN_TIME_LABEL,
            PROVE_EXCL_TRACE_TIME_LABEL,
        ];
        let mut previous = 0;
        for metric in ordered {
            let position = markdown
                .find(metric)
                .unwrap_or_else(|| panic!("{metric} is missing from the report"));
            assert!(position >= previous, "{metric} is out of pipeline order");
            previous = position;
        }
        for metric in [
            PREPARE_PREFLIGHT_TIME_LABEL,
            UPLOAD_PREFLIGHT_PROGRAM_TIME_LABEL,
            APP_PROVE_TIME_LABEL,
            EXECUTE_PREFLIGHT_INTERVALS_LABEL,
            EXECUTE_PREFLIGHT_RESIDUALS_LABEL,
            EXECUTE_PREFLIGHT_TRANSCRIPT_BYTES_LABEL,
        ] {
            assert!(
                !markdown.contains(metric),
                "{metric} should remain raw data instead of a summary row"
            );
        }
    }

    #[test]
    fn preflight_parallel_diffs_require_matching_baseline_metrics() {
        let mut legacy_metrics = preflight_timing_metrics();
        legacy_metrics.remove(EXECUTE_PREFLIGHT_TIME_LABEL);
        let legacy = grouped(legacy_metrics).aggregate(2);
        let mut current = grouped(preflight_timing_metrics()).aggregate(2);

        current.set_diff(&legacy);

        assert_eq!(current.total_proof_time.diff, Some(0.0));
        assert_eq!(
            current.by_group["app"][PROOF_TIME_LABEL].sum.diff,
            Some(0.0)
        );
        assert_eq!(current.par_by_group["app"].diff, None);
        assert_eq!(current.bounded_par_by_group["app"].diff, None);
        assert_eq!(current.total_par_proof_time.diff, None);
    }

    #[test]
    fn all_available_execution_counts_match() {
        let metrics = HashMap::from([
            (
                EXECUTE_PURE_INSNS_LABEL.to_string(),
                vec![(100.0, labels(None))],
            ),
            (
                EXECUTE_METERED_INSNS_LABEL.to_string(),
                vec![(100.0, labels(None))],
            ),
            (
                EXECUTE_METERED_COST_INSNS_LABEL.to_string(),
                vec![(100.0, labels(None))],
            ),
            (
                EXECUTE_PREFLIGHT_INSNS_LABEL.to_string(),
                vec![(40.0, labels(Some(0))), (60.0, labels(Some(1)))],
            ),
        ]);

        let aggregate = grouped(metrics).aggregate(2);
        assert_eq!(
            aggregate.by_group["app"][EXECUTE_PREFLIGHT_INSNS_LABEL]
                .sum
                .val,
            100.0
        );
    }

    #[test]
    #[should_panic(
        expected = "instruction count mismatch between execute_pure_insns and execute_preflight_insns"
    )]
    fn mismatched_execution_counts_are_rejected() {
        let metrics = HashMap::from([
            (
                EXECUTE_PURE_INSNS_LABEL.to_string(),
                vec![(100.0, labels(None))],
            ),
            (
                EXECUTE_PREFLIGHT_INSNS_LABEL.to_string(),
                vec![(99.0, labels(None))],
            ),
        ]);

        grouped(metrics).aggregate(2);
    }
}
