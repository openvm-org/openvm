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
                        .unwrap_or_else(|| panic!("Label key '{}' should exist in label_dict", key))
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

                    row.push_str(&format!("{} | ", metric_value));
                }

                markdown_output.push_str(&row);
                markdown_output.push('\n');
            }

            markdown_output.push('\n');
        }

        markdown_output
    }
}
