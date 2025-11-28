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

    /// Generate a Mermaid XY chart for GPU memory usage over modules
    pub fn generate_gpu_memory_chart(&self) -> Option<String> {
        // (timestamp, tracked_gb, reserved_gb, device_gb)
        let mut data: Vec<(f64, f64, f64, f64)> = Vec::new();
        // module -> [(tracked_gb, context_label)]
        let mut module_stats: HashMap<String, Vec<(f64, String)>> = HashMap::new();

        for (label_keys, metrics_dict) in &self.dict_by_label_types {
            let module_idx = match label_keys.iter().position(|k| k == "module") {
                Some(idx) => idx,
                None => continue,
            };

            for (label_values, metrics) in metrics_dict {
                let get = |name: &str| metrics.iter().find(|m| m.name == name).map(|m| m.value);
                let ts = get("gpu_mem.timestamp_ms");
                let tracked = get("gpu_mem.tracked_bytes");
                let reserved = get("gpu_mem.reserved_bytes");
                let device = get("gpu_mem.device_bytes");

                if let (Some(ts), Some(tracked), Some(reserved), Some(device)) =
                    (ts, tracked, reserved, device)
                {
                    let tracked_gb = tracked / 1e9;
                    let reserved_gb = reserved / 1e9;
                    let device_gb = device / 1e9;
                    data.push((ts, tracked_gb, reserved_gb, device_gb));

                    let module_name = label_values.get(module_idx).cloned().unwrap_or_default();
                    let context_label: String = label_keys
                        .iter()
                        .zip(label_values.iter())
                        .filter(|(k, _)| *k != "module" && *k != "block_number")
                        .map(|(_, v)| v.as_str())
                        .collect::<Vec<_>>()
                        .join(".");

                    module_stats
                        .entry(module_name)
                        .or_default()
                        .push((tracked_gb, context_label));
                }
            }
        }

        if data.is_empty() {
            return None;
        }

        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let max_tracked = data.iter().map(|(_, t, _, _)| *t).fold(0.0_f64, f64::max);
        let max_reserved = data.iter().map(|(_, _, r, _)| *r).fold(0.0_f64, f64::max);
        let max_device = data.iter().map(|(_, _, _, d)| *d).fold(0.0_f64, f64::max);
        let chart_max = max_tracked.max(max_reserved).max(max_device);

        let mut chart = String::new();
        chart.push_str("```mermaid\n");
        chart.push_str("---\n");
        chart.push_str("config:\n");
        chart.push_str("    xyChart:\n");
        chart.push_str("        xAxis:\n");
        chart.push_str("            showLabel: false\n");
        chart.push_str("    themeVariables:\n");
        chart.push_str("        xyChart:\n");
        chart.push_str("            plotColorPalette: \"#2563eb, #16a34a, #dc2626\"\n");
        chart.push_str("---\n");
        chart.push_str("xychart-beta\n");
        chart.push_str("    title \"GPU Memory Usage\"\n");
        chart.push_str(&format!(
            "    y-axis \"Memory (GB)\" 0 --> {:.1}\n",
            chart_max * 1.1
        ));
        // Tracked memory line (blue)
        chart.push_str("    line [");
        chart.push_str(
            &data
                .iter()
                .map(|(_, tracked, _, _)| format!("{:.2}", tracked))
                .collect::<Vec<_>>()
                .join(", "),
        );
        chart.push_str("]\n");
        // Reserved memory line (green)
        chart.push_str("    line [");
        chart.push_str(
            &data
                .iter()
                .map(|(_, _, reserved, _)| format!("{:.2}", reserved))
                .collect::<Vec<_>>()
                .join(", "),
        );
        chart.push_str("]\n");
        // Device memory line (red)
        chart.push_str("    line [");
        chart.push_str(
            &data
                .iter()
                .map(|(_, _, _, device)| format!("{:.2}", device))
                .collect::<Vec<_>>()
                .join(", "),
        );
        chart.push_str("]\n");
        chart.push_str("```\n");

        chart.push_str("\n> ");
        chart.push_str("🔵 Tracked (Current) | ");
        chart.push_str("🟢 Reserved (Pool) | ");
        chart.push_str("🔴 Device\n");

        // Per-module stats table
        chart.push_str("\n| Module | Max (GB) | Max At |\n");
        chart.push_str("| --- | ---: | --- |\n");

        let mut module_rows: Vec<_> = module_stats
            .iter()
            .map(|(module, entries)| {
                let (max_tracked, max_at) = entries
                    .iter()
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(t, label)| (*t, label.as_str()))
                    .unwrap_or((0.0, ""));
                (module, max_tracked, max_at)
            })
            .collect();
        module_rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (module, max_tracked, max_at) in module_rows {
            chart.push_str(&format!(
                "| {} | {:.2} | {} |\n",
                module, max_tracked, max_at
            ));
        }

        Some(chart)
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

            // Filter out gpu_mem metrics - these are summarized in the GPU memory chart
            metric_names.retain(|n| !n.starts_with("gpu_mem."));

            // Skip tables that have no metrics left after filtering
            if metric_names.is_empty() {
                continue;
            }

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
}
