use std::{collections::HashMap, fs::File, path::Path};

use aggregate::{PROOF_TIME_LABEL, PROVE_EXCL_TRACE_TIME_LABEL, TRACE_GEN_TIME_LABEL};
use eyre::Result;
use memmap2::Mmap;

use crate::{
    aggregate::{EXECUTE_METERED_TIME_LABEL, EXECUTE_PREFLIGHT_TIME_LABEL},
    types::{Labels, Metric, MetricDb, MetricsFile},
};

pub mod aggregate;
pub mod summary;
pub mod types;

impl MetricDb {
    pub fn new(metrics_file: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(metrics_file)?;
        // SAFETY: File is read-only mapped. File will not be modified by other
        // processes during the mapping's lifetime.
        let mmap = unsafe { Mmap::map(&file)? };
        let metrics: MetricsFile = serde_json::from_slice(&mmap)?;

        let mut db = MetricDb::default();

        // Process counters
        for entry in metrics.counter {
            if entry.value == 0.0 {
                continue;
            }
            let labels = Labels::from(entry.labels);
            db.add_to_flat_dict(labels, entry.metric, entry.value);
        }

        // Process gauges
        for entry in metrics.gauge {
            let labels = Labels::from(entry.labels);
            db.add_to_flat_dict(labels, entry.metric, entry.value);
        }

        db.apply_aggregations();
        db.separate_by_label_types();

        Ok(db)
    }

    // Currently hardcoding aggregations
    pub fn apply_aggregations(&mut self) {
        for metrics in self.flat_dict.values_mut() {
            let get = |key: &str| metrics.iter().find(|m| m.name == key).map(|m| m.value);
            let total_proof_time = get(PROOF_TIME_LABEL);
            if total_proof_time.is_some() {
                // We have instrumented total_proof_time_ms
                continue;
            }
            // otherwise, calculate it from sub-components
            let execute_metered_time = get(EXECUTE_METERED_TIME_LABEL);
            let execute_preflight_time = get(EXECUTE_PREFLIGHT_TIME_LABEL);
            let trace_gen_time = get(TRACE_GEN_TIME_LABEL);
            let prove_excl_trace_time = get(PROVE_EXCL_TRACE_TIME_LABEL);
            if let (
                Some(execute_preflight_time),
                Some(trace_gen_time),
                Some(prove_excl_trace_time),
            ) = (
                execute_preflight_time,
                trace_gen_time,
                prove_excl_trace_time,
            ) {
                let total_time = execute_metered_time.unwrap_or(0.0)
                    + execute_preflight_time
                    + trace_gen_time
                    + prove_excl_trace_time;
                metrics.push(Metric::new(PROOF_TIME_LABEL.to_string(), total_time));
            }
        }
    }

    pub fn add_to_flat_dict(&mut self, labels: Labels, metric: String, value: f64) {
        self.flat_dict
            .entry(labels)
            .or_default()
            .push(Metric::new(metric, value));
    }

    // Custom sorting function that ensures 'group' comes first.
    // Other keys are sorted alphabetically.
    pub fn custom_sort_label_keys(label_keys: &mut [String]) {
        // Prioritize 'group' by giving it the lowest possible sort value
        label_keys.sort_by_key(|key| {
            if key == "group" {
                (0, key.clone()) // Lowest priority for 'group'
            } else {
                (1, key.clone()) // Normal priority for other keys
            }
        });
    }

    pub fn separate_by_label_types(&mut self) {
        self.dict_by_label_types.clear();

        for (labels, metrics) in &self.flat_dict {
            // Get sorted label keys
            let mut label_keys: Vec<String> = labels.0.iter().map(|(key, _)| key.clone()).collect();
            Self::custom_sort_label_keys(&mut label_keys);

            // Create label_values based on sorted keys
            let label_dict: HashMap<String, String> = labels.0.iter().cloned().collect();

            let label_values: Vec<String> = label_keys
                .iter()
                .map(|key| {
                    label_dict
                        .get(key)
                        .unwrap_or_else(|| panic!("Label key '{key}' should exist in label_dict"))
                        .clone()
                })
                .collect();

            // Add to dict_by_label_types
            self.dict_by_label_types
                .entry(label_keys)
                .or_default()
                .entry(label_values)
                .or_default()
                .extend(metrics.clone());
        }
    }

    pub fn generate_markdown_tables(&self) -> String {
        let mut markdown_output = String::new();
        // Get sorted keys to iterate in consistent order
        let mut sorted_keys: Vec<_> = self.dict_by_label_types.keys().cloned().collect();
        sorted_keys.sort();

        for label_keys in sorted_keys {
            if label_keys.contains(&"cycle_tracker_span".to_string()) {
                // Skip cycle_tracker_span as it is too long for markdown and visualized in
                // flamegraphs
                continue;
            }
            // Skip wide tables with group + block_number that have too many metrics
            if label_keys.contains(&"group".to_string())
                && label_keys.contains(&"block_number".to_string())
                && !label_keys.contains(&"idx".to_string())
                && !label_keys.contains(&"segment".to_string())
                && !label_keys.contains(&"air_id".to_string())
            {
                continue;
            }
            let metrics_dict = &self.dict_by_label_types[&label_keys];
            let mut metric_names: Vec<String> = metrics_dict
                .values()
                .flat_map(|metrics| metrics.iter().map(|m| m.name.clone()))
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            metric_names.sort_by(|a, b| b.cmp(a));

            // Create table header
            let header = format!(
                "| {} | {} |",
                label_keys.join(" | "),
                metric_names.join(" | ")
            );

            let separator = "| ".to_string()
                + &vec!["---"; label_keys.len() + metric_names.len()].join(" | ")
                + " |";

            markdown_output.push_str(&header);
            markdown_output.push('\n');
            markdown_output.push_str(&separator);
            markdown_output.push('\n');

            // Fill table rows
            for (label_values, metrics) in metrics_dict {
                let mut row = String::new();
                row.push_str("| ");
                row.push_str(&label_values.join(" | "));
                row.push_str(" | ");

                // Add metric values
                for metric_name in &metric_names {
                    let metric_value = metrics
                        .iter()
                        .find(|m| &m.name == metric_name)
                        .map(|m| Self::format_number(m.value))
                        .unwrap_or_default();

                    row.push_str(&format!("{metric_value} | "));
                }

                markdown_output.push_str(&row);
                markdown_output.push('\n');
            }

            markdown_output.push('\n');
        }

        markdown_output
    }

    /// Aggregate trace dimension metrics by (group, idx/segment), then by group.
    ///
    /// For each (group, idx/segment) pair, computes:
    /// - sum of main_cols
    /// - sum of perm_cols
    /// - total_cols = main_cols + 4 * perm_cols
    /// - max rows
    /// - sum of cells
    ///
    /// Then for each group, aggregates over all idx/segment values to get avg/max/sum.
    pub fn aggregate_trace_dimensions(&self) -> TraceDimensionAggregates {
        // First pass: group by (group, idx_or_segment) and aggregate metrics
        // We use either "idx" or "segment" as the second grouping key
        let mut by_group_idx: HashMap<(String, String), TraceDimGroupIdxStats> = HashMap::new();

        for (labels, metrics) in &self.flat_dict {
            let group = labels.get("group");
            // Use idx if present, otherwise use segment
            let idx_or_segment = labels.get("idx").or_else(|| labels.get("segment"));

            // Only process entries that have both group and idx/segment labels
            if let (Some(group), Some(idx)) = (group, idx_or_segment) {
                let key = (group.to_string(), idx.to_string());
                let entry = by_group_idx.entry(key).or_default();

                for metric in metrics {
                    match metric.name.as_str() {
                        "main_cols" => entry.main_cols_sum += metric.value,
                        "perm_cols" => entry.perm_cols_sum += metric.value,
                        "rows" => {
                            if metric.value > entry.rows_max {
                                entry.rows_max = metric.value;
                            }
                        }
                        "cells" => entry.cells_sum += metric.value,
                        "total_constraint_count" => {
                            if metric.value > entry.total_constraint_count {
                                entry.total_constraint_count = metric.value;
                            }
                        }
                        "batch_size" => {
                            if metric.value > entry.batch_size {
                                entry.batch_size = metric.value;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Calculate total_cols for each (group, idx) pair
        for stats in by_group_idx.values_mut() {
            stats.total_cols = stats.main_cols_sum + 4.0 * stats.perm_cols_sum;
        }

        // Second pass: group by group and aggregate over all idx values
        let mut by_group: HashMap<String, TraceDimGroupStats> = HashMap::new();

        for ((group, _idx), stats) in &by_group_idx {
            let entry = by_group.entry(group.clone()).or_default();
            entry.push(stats);
        }

        // Finalize the group stats
        for stats in by_group.values_mut() {
            stats.finalize();
        }

        // Third pass: group by (group, air_id) and sum cells over all idx/segment
        let mut by_group_air: HashMap<(String, String), TraceDimAirStats> = HashMap::new();

        for (labels, metrics) in &self.flat_dict {
            let group = labels.get("group");
            let air_id = labels.get("air_id");
            let air_name = labels.get("air_name");

            // Only process entries that have group, air_id, and air_name labels
            if let (Some(group), Some(air_id), Some(air_name)) = (group, air_id, air_name) {
                let key = (group.to_string(), air_id.to_string());
                let entry = by_group_air.entry(key).or_default();

                // Set air_name (should be consistent for same air_id)
                if entry.air_name.is_empty() {
                    entry.air_name = air_name.to_string();
                }

                for metric in metrics {
                    if metric.name == "cells" {
                        entry.cells_sum += metric.value;
                    }
                }
            }
        }

        TraceDimensionAggregates {
            by_group_idx,
            by_group,
            by_group_air,
        }
    }

    pub fn generate_trace_dimension_tables(&self) -> String {
        let aggregates = self.aggregate_trace_dimensions();

        if aggregates.by_group_idx.is_empty() {
            return String::new();
        }

        let mut output = String::new();

        // Table 1: Per (group, idx/segment) aggregates
        output.push_str("### Trace Dimensions by (group, idx/segment)\n\n");
        output.push_str("| group | idx/segment | main_cols | perm_cols | total_cols | rows | cells |\n");
        output.push_str("| --- | --- | --- | --- | --- | --- | --- |\n");

        // Sort by group, then by idx (numerically if possible)
        let mut sorted_keys: Vec<_> = aggregates.by_group_idx.keys().collect();
        sorted_keys.sort_by(|a, b| {
            let group_cmp = a.0.cmp(&b.0);
            if group_cmp != std::cmp::Ordering::Equal {
                return group_cmp;
            }
            // Try to sort idx numerically
            match (a.1.parse::<i64>(), b.1.parse::<i64>()) {
                (Ok(a_idx), Ok(b_idx)) => a_idx.cmp(&b_idx),
                _ => a.1.cmp(&b.1),
            }
        });

        for key in sorted_keys {
            let stats = &aggregates.by_group_idx[key];
            output.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} |\n",
                key.0,
                key.1,
                Self::format_number(stats.main_cols_sum),
                Self::format_number(stats.perm_cols_sum),
                Self::format_number(stats.total_cols),
                Self::format_number(stats.rows_max),
                Self::format_number(stats.cells_sum),
            ));
        }
        output.push('\n');

        // Sort groups
        let mut sorted_groups: Vec<_> = aggregates.by_group.keys().collect();
        sorted_groups.sort();

        // Table 2a: Max total_constraint_count and batch_size per group
        output.push_str("### Constraint Count and Batch Size by group\n\n");
        output.push_str("| group | max(total_constraint_count) | max(batch_size) |\n");
        output.push_str("| --- | --- | --- |\n");
        for group in &sorted_groups {
            let stats = &aggregates.by_group[*group];
            if stats.total_constraint_count_max > 0.0 || stats.batch_size_max > 0.0 {
                output.push_str(&format!(
                    "| {} | {} | {} |\n",
                    group,
                    Self::format_number(stats.total_constraint_count_max),
                    Self::format_number(stats.batch_size_max),
                ));
            }
        }
        output.push('\n');

        // Table 2b: Per group aggregates (avg, max, sum over all idx/segment)
        output.push_str("### Trace Dimensions by group (aggregated over idx/segment)\n\n");
        output.push_str("| group | metric | avg | max | sum |\n");
        output.push_str("| --- | --- | --- | --- | --- |\n");

        for group in &sorted_groups {
            let stats = &aggregates.by_group[*group];
            // main_cols
            output.push_str(&format!(
                "| {} | main_cols | {} | {} | {} |\n",
                group,
                Self::format_number(stats.main_cols.avg),
                Self::format_number(stats.main_cols.max),
                Self::format_number(stats.main_cols.sum),
            ));
            // perm_cols
            output.push_str(&format!(
                "| {} | perm_cols | {} | {} | {} |\n",
                group,
                Self::format_number(stats.perm_cols.avg),
                Self::format_number(stats.perm_cols.max),
                Self::format_number(stats.perm_cols.sum),
            ));
            // total_cols
            output.push_str(&format!(
                "| {} | total_cols | {} | {} | {} |\n",
                group,
                Self::format_number(stats.total_cols.avg),
                Self::format_number(stats.total_cols.max),
                Self::format_number(stats.total_cols.sum),
            ));
            // rows
            output.push_str(&format!(
                "| {} | rows | {} | {} | {} |\n",
                group,
                Self::format_number(stats.rows.avg),
                Self::format_number(stats.rows.max),
                Self::format_number(stats.rows.sum),
            ));
            // cells
            output.push_str(&format!(
                "| {} | cells | {} | {} | {} |\n",
                group,
                Self::format_number(stats.cells.avg),
                Self::format_number(stats.cells.max),
                Self::format_number(stats.cells.sum),
            ));
        }
        output.push('\n');

        // Table 3: Per (group, air_id) - cells summed over all idx/segment
        output.push_str("### Cells by (group, air_id)\n\n");
        output.push_str("| group | air_id | air_name | cells |\n");
        output.push_str("| --- | --- | --- | --- |\n");

        // Sort by group, then by cells descending
        let mut sorted_air_keys: Vec<_> = aggregates.by_group_air.keys().collect();
        sorted_air_keys.sort_by(|a, b| {
            let group_cmp = a.0.cmp(&b.0);
            if group_cmp != std::cmp::Ordering::Equal {
                return group_cmp;
            }
            // Sort by cells descending within each group
            let a_cells = aggregates.by_group_air[*a].cells_sum;
            let b_cells = aggregates.by_group_air[*b].cells_sum;
            b_cells
                .partial_cmp(&a_cells)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for key in sorted_air_keys {
            let stats = &aggregates.by_group_air[key];
            output.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                key.0,
                key.1,
                stats.air_name,
                Self::format_number(stats.cells_sum),
            ));
        }
        output.push('\n');

        output
    }
}

/// Statistics for a single (group, idx) pair
#[derive(Debug, Clone, Default)]
pub struct TraceDimGroupIdxStats {
    pub main_cols_sum: f64,
    pub perm_cols_sum: f64,
    pub total_cols: f64, // main_cols + 4 * perm_cols
    pub rows_max: f64,
    pub cells_sum: f64,
    pub total_constraint_count: f64,
    pub batch_size: f64,
}

/// Statistics for a single (group, air_id) pair
#[derive(Debug, Clone, Default)]
pub struct TraceDimAirStats {
    pub air_name: String,
    pub cells_sum: f64,
}

/// Aggregate statistics (avg, max, sum) for a metric
#[derive(Debug, Clone, Default)]
pub struct MetricAggregate {
    pub avg: f64,
    pub max: f64,
    pub sum: f64,
    count: usize,
}

impl MetricAggregate {
    fn push(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
        if value > self.max {
            self.max = value;
        }
    }

    fn finalize(&mut self) {
        if self.count > 0 {
            self.avg = self.sum / self.count as f64;
        }
    }
}

/// Statistics for a group (aggregated over all idx values)
#[derive(Debug, Clone, Default)]
pub struct TraceDimGroupStats {
    pub main_cols: MetricAggregate,
    pub perm_cols: MetricAggregate,
    pub total_cols: MetricAggregate,
    pub rows: MetricAggregate,
    pub cells: MetricAggregate,
    pub total_constraint_count_max: f64,
    pub batch_size_max: f64,
}

impl TraceDimGroupStats {
    fn push(&mut self, stats: &TraceDimGroupIdxStats) {
        self.main_cols.push(stats.main_cols_sum);
        self.perm_cols.push(stats.perm_cols_sum);
        self.total_cols.push(stats.total_cols);
        self.rows.push(stats.rows_max);
        self.cells.push(stats.cells_sum);
        if stats.total_constraint_count > self.total_constraint_count_max {
            self.total_constraint_count_max = stats.total_constraint_count;
        }
        if stats.batch_size > self.batch_size_max {
            self.batch_size_max = stats.batch_size;
        }
    }

    fn finalize(&mut self) {
        self.main_cols.finalize();
        self.perm_cols.finalize();
        self.total_cols.finalize();
        self.rows.finalize();
        self.cells.finalize();
    }
}

/// Container for all trace dimension aggregates
#[derive(Debug, Clone, Default)]
pub struct TraceDimensionAggregates {
    /// Stats per (group, idx/segment) pair
    pub by_group_idx: HashMap<(String, String), TraceDimGroupIdxStats>,
    /// Stats per group (aggregated over all idx/segment)
    pub by_group: HashMap<String, TraceDimGroupStats>,
    /// Stats per (group, air_id) - cells summed over all idx/segment
    pub by_group_air: HashMap<(String, String), TraceDimAirStats>,
}
