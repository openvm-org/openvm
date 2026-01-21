use std::{collections::HashMap, fs::File, path::Path};

use aggregate::{PROOF_TIME_LABEL, PROVE_EXCL_TRACE_TIME_LABEL, TRACE_GEN_TIME_LABEL};
use eyre::Result;
use memmap2::Mmap;

use crate::{
    aggregate::{EXECUTE_METERED_TIME_LABEL, EXECUTE_PREFLIGHT_TIME_LABEL},
    types::{Labels, Metric, MetricDb, MetricsFile},
};

pub mod aggregate;
pub mod instruction_count;
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

            // Remove cycle_tracker_span and dsl_ir if present as they are too long for markdown and
            // visualized in flamegraphs
            let mut keys = label_keys.clone();
            let mut values = label_values.clone();

            // Remove cycle_tracker_span if present
            if let Some(index) = keys.iter().position(|k| k == "cycle_tracker_span") {
                keys.remove(index);
                values.remove(index);
            }

            // Remove dsl_ir if present
            if let Some(index) = keys.iter().position(|k| k == "dsl_ir") {
                keys.remove(index);
                values.remove(index);
            }

            let (final_label_keys, final_label_values) = (keys, values);

            // Add to dict_by_label_types, combining metrics with same name by summing values
            let entry = self
                .dict_by_label_types
                .entry(final_label_keys)
                .or_default()
                .entry(final_label_values)
                .or_default();

            for metric in metrics.clone() {
                if let Some(existing_metric) = entry.iter_mut().find(|m| m.name == metric.name) {
                    // Sum the values for metrics with the same name
                    existing_metric.value += metric.value;
                } else {
                    // Add new metric if no existing one with same name
                    entry.push(metric);
                }
            }
        }
    }

    /// Generate an SVG chart for GPU memory usage over modules.
    /// Returns a tuple of (svg_string, markdown_table) where the SVG can be embedded
    /// directly in HTML/markdown and the table provides per-module statistics.
    pub fn generate_gpu_memory_chart(&self) -> Option<(String, String)> {
        // (timestamp, current_gb, local_peak_gb, reserved_gb)
        let mut data: Vec<(f64, f64, f64, f64)> = Vec::new();
        // module -> [(local_peak_gb, context_label)]
        let mut module_stats: HashMap<String, Vec<(f64, String)>> = HashMap::new();

        for (label_keys, metrics_dict) in &self.dict_by_label_types {
            let module_idx = match label_keys.iter().position(|k| k == "module") {
                Some(idx) => idx,
                None => continue,
            };

            for (label_values, metrics) in metrics_dict {
                let get = |name: &str| metrics.iter().find(|m| m.name == name).map(|m| m.value);
                let ts = get("gpu_mem.timestamp_ms");
                let current = get("gpu_mem.current_bytes");
                let local_peak = get("gpu_mem.local_peak_bytes");
                let reserved = get("gpu_mem.reserved_bytes");

                if let (Some(ts), Some(current), Some(local_peak), Some(reserved)) =
                    (ts, current, local_peak, reserved)
                {
                    let current_gb = current / f64::from(1 << 30);
                    let local_peak_gb = local_peak / f64::from(1 << 30);
                    let reserved_gb = reserved / f64::from(1 << 30);
                    data.push((ts, current_gb, local_peak_gb, reserved_gb));

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
                        .push((local_peak_gb, context_label));
                }
            }
        }

        if data.is_empty() {
            return None;
        }

        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Downsample if too many points (SVG handles ~1000 points well)
        let max_points = 1000;
        let data = if data.len() > max_points {
            let step = data.len() / max_points;
            data.into_iter()
                .enumerate()
                .filter(|(i, _)| i % step == 0)
                .map(|(_, d)| d)
                .collect::<Vec<_>>()
        } else {
            data
        };

        let svg = Self::render_gpu_memory_svg(&data);

        // Per-module stats table
        let mut table = String::new();
        table.push_str("| Module | Max (GB) | Max At |\n");
        table.push_str("| --- | ---: | --- |\n");

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
            table.push_str(&format!("| {} | {:.2} | {} |\n", module, max_tracked, max_at));
        }

        Some((svg, table))
    }

    /// Render GPU memory data as an SVG chart
    fn render_gpu_memory_svg(data: &[(f64, f64, f64, f64)]) -> String {
        // Chart dimensions
        let width = 800.0_f64;
        let height = 400.0_f64;
        let margin_left = 60.0_f64;
        let margin_right = 20.0_f64;
        let margin_top = 40.0_f64;
        let margin_bottom = 50.0_f64;

        let plot_width = width - margin_left - margin_right;
        let plot_height = height - margin_top - margin_bottom;

        // Calculate data ranges
        let ts_min = data.iter().map(|(ts, _, _, _)| *ts).fold(f64::MAX, f64::min);
        let ts_max = data.iter().map(|(ts, _, _, _)| *ts).fold(f64::MIN, f64::max);
        let ts_range = if (ts_max - ts_min).abs() < f64::EPSILON {
            1.0
        } else {
            ts_max - ts_min
        };

        let max_current = data.iter().map(|(_, c, _, _)| *c).fold(0.0_f64, f64::max);
        let max_local_peak = data.iter().map(|(_, _, lp, _)| *lp).fold(0.0_f64, f64::max);
        let max_reserved = data.iter().map(|(_, _, _, r)| *r).fold(0.0_f64, f64::max);
        let y_max = max_current.max(max_local_peak).max(max_reserved) * 1.1;
        let y_max = if y_max < f64::EPSILON { 1.0 } else { y_max };

        // Helper to convert data coordinates to SVG coordinates
        let to_svg_x = |ts: f64| -> f64 { margin_left + (ts - ts_min) / ts_range * plot_width };
        let to_svg_y = |val: f64| -> f64 { margin_top + plot_height - (val / y_max) * plot_height };

        // Build path strings for each series
        let build_path = |series: &[(f64, f64)]| -> String {
            series
                .iter()
                .enumerate()
                .map(|(i, (x, y))| {
                    let cmd = if i == 0 { "M" } else { "L" };
                    format!("{}{:.1},{:.1}", cmd, x, y)
                })
                .collect::<Vec<_>>()
                .join(" ")
        };

        let current_points: Vec<(f64, f64)> = data
            .iter()
            .map(|(ts, current, _, _)| (to_svg_x(*ts), to_svg_y(*current)))
            .collect();
        let local_peak_points: Vec<(f64, f64)> = data
            .iter()
            .map(|(ts, _, lp, _)| (to_svg_x(*ts), to_svg_y(*lp)))
            .collect();
        let reserved_points: Vec<(f64, f64)> = data
            .iter()
            .map(|(ts, _, _, r)| (to_svg_x(*ts), to_svg_y(*r)))
            .collect();

        let current_path = build_path(&current_points);
        let local_peak_path = build_path(&local_peak_points);
        let reserved_path = build_path(&reserved_points);

        // Generate Y-axis ticks (5-6 ticks)
        let y_tick_count = 5;
        let y_tick_interval = y_max / y_tick_count as f64;
        let y_ticks: Vec<f64> = (0..=y_tick_count).map(|i| i as f64 * y_tick_interval).collect();

        // Generate gridlines and tick labels
        let mut gridlines = String::new();
        let mut tick_labels = String::new();
        for y_val in &y_ticks {
            let y_pos = to_svg_y(*y_val);
            gridlines.push_str(&format!(
                "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>",
                margin_left,
                y_pos,
                width - margin_right,
                y_pos
            ));
            gridlines.push('\n');
            tick_labels.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" font-size=\"12\" fill=\"#6b7280\">{:.1}</text>",
                margin_left - 8.0,
                y_pos + 4.0,
                y_val
            ));
            tick_labels.push('\n');
        }

        // X-axis time labels (show duration)
        let duration_sec = ts_range / 1000.0;
        let duration_label = if duration_sec < 60.0 {
            format!("{:.1}s", duration_sec)
        } else if duration_sec < 3600.0 {
            format!("{:.1}m", duration_sec / 60.0)
        } else {
            format!("{:.1}h", duration_sec / 3600.0)
        };

        // Colors
        let color_current = "#2563eb"; // blue
        let color_local_peak = "#16a34a"; // green
        let color_reserved = "#dc2626"; // red

        format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <style>
    .title {{ font: bold 16px sans-serif; }}
    .axis-label {{ font: 12px sans-serif; fill: #6b7280; }}
    .legend-text {{ font: 12px sans-serif; }}
  </style>

  <!-- Background -->
  <rect width="{width}" height="{height}" fill="white"/>

  <!-- Title -->
  <text x="{title_x}" y="24" text-anchor="middle" class="title">GPU Memory Usage</text>

  <!-- Gridlines -->
  {gridlines}

  <!-- Y-axis tick labels -->
  {tick_labels}

  <!-- Y-axis label -->
  <text x="16" y="{y_label_y}" transform="rotate(-90, 16, {y_label_y})" text-anchor="middle" class="axis-label">Memory (GB)</text>

  <!-- X-axis -->
  <line x1="{margin_left}" y1="{x_axis_y}" x2="{x_axis_end}" y2="{x_axis_y}" stroke="#9ca3af" stroke-width="1"/>
  <text x="{margin_left}" y="{x_label_y}" text-anchor="start" class="axis-label">0</text>
  <text x="{x_axis_end}" y="{x_label_y}" text-anchor="end" class="axis-label">{duration_label}</text>

  <!-- Data lines -->
  <path d="{reserved_path}" fill="none" stroke="{color_reserved}" stroke-width="1.5" stroke-opacity="0.8"/>
  <path d="{local_peak_path}" fill="none" stroke="{color_local_peak}" stroke-width="1.5" stroke-opacity="0.8"/>
  <path d="{current_path}" fill="none" stroke="{color_current}" stroke-width="2"/>

  <!-- Legend -->
  <g transform="translate({legend_x}, {legend_y})">
    <rect x="0" y="0" width="180" height="70" fill="white" fill-opacity="0.9" stroke="#e5e7eb" rx="4"/>
    <line x1="10" y1="18" x2="30" y2="18" stroke="{color_current}" stroke-width="2"/>
    <text x="38" y="22" class="legend-text">Current</text>
    <line x1="10" y1="38" x2="30" y2="38" stroke="{color_local_peak}" stroke-width="1.5"/>
    <text x="38" y="42" class="legend-text">Local Peak</text>
    <line x1="10" y1="58" x2="30" y2="58" stroke="{color_reserved}" stroke-width="1.5"/>
    <text x="38" y="62" class="legend-text">Reserved (Pool)</text>
  </g>
</svg>"##,
            width = width,
            height = height,
            title_x = width / 2.0,
            gridlines = gridlines,
            tick_labels = tick_labels,
            y_label_y = margin_top + plot_height / 2.0,
            margin_left = margin_left,
            x_axis_y = margin_top + plot_height,
            x_axis_end = width - margin_right,
            x_label_y = margin_top + plot_height + 20.0,
            duration_label = duration_label,
            reserved_path = reserved_path,
            local_peak_path = local_peak_path,
            current_path = current_path,
            color_current = color_current,
            color_local_peak = color_local_peak,
            color_reserved = color_reserved,
            legend_x = width - margin_right - 190.0,
            legend_y = margin_top + 10.0,
        )
    }

    pub fn sum_metric_grouped_by(
        &mut self,
        metric_name: &str,
        group_by_keys: &[&str],
        new_metric_name: &str,
    ) {
        let mut sums: HashMap<Vec<(String, String)>, f64> = HashMap::new();

        for (labels, metrics) in &self.flat_dict {
            let group_values: Option<Vec<(String, String)>> = group_by_keys
                .iter()
                .map(|key| labels.get(key).map(|v| (key.to_string(), v.to_string())))
                .collect();

            let Some(group_values) = group_values else {
                continue;
            };

            for metric in metrics {
                if metric.name == metric_name {
                    *sums.entry(group_values.clone()).or_default() += metric.value;
                }
            }
        }

        for (group_labels, sum) in sums {
            let labels = Labels(group_labels);
            self.add_to_flat_dict(labels, new_metric_name.to_string(), sum);
        }
    }

    pub fn generate_markdown_tables(&self) -> String {
        let mut markdown_output = String::new();
        // Get sorted keys to iterate in consistent order
        let mut sorted_keys: Vec<_> = self.dict_by_label_types.keys().cloned().collect();
        sorted_keys.sort();

        for label_keys in sorted_keys {
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
            // Filter out wrapper spans - these are just for propagating fields to child spans
            metric_names.retain(|n| !n.starts_with("wrapper."));

            // Skip tables that have no metrics left after filtering
            if metric_names.is_empty() {
                continue;
            }

            // Create table header
            let header = if label_keys.is_empty() {
                format!("| {} |", metric_names.join(" | "))
            } else {
                format!(
                    "| {} | {} |",
                    label_keys.join(" | "),
                    metric_names.join(" | ")
                )
            };

            let separator = "| ".to_string()
                + &vec!["---"; label_keys.len() + metric_names.len()].join(" | ")
                + " |";

            markdown_output.push_str(&header);
            markdown_output.push('\n');
            markdown_output.push_str(&separator);
            markdown_output.push('\n');

            // Sort rows: first by segment (ascending) if present, then by frequency (descending) if
            // present
            let mut rows: Vec<_> = metrics_dict.iter().collect();
            let segment_index = label_keys.iter().position(|k| k == "segment");
            let has_frequency = metric_names.contains(&"frequency".to_string());

            if segment_index.is_some() || has_frequency {
                rows.sort_by(|(label_values_a, metrics_a), (label_values_b, metrics_b)| {
                    // First, sort by segment (ascending) if present
                    if let Some(seg_idx) = segment_index {
                        let seg_a = label_values_a
                            .get(seg_idx)
                            .map(|s| s.as_str())
                            .unwrap_or("");
                        let seg_b = label_values_b
                            .get(seg_idx)
                            .map(|s| s.as_str())
                            .unwrap_or("");
                        let seg_cmp = seg_a.cmp(seg_b);
                        if seg_cmp != std::cmp::Ordering::Equal {
                            return seg_cmp;
                        }
                    }

                    // Then, sort by frequency (descending) if present
                    if has_frequency {
                        let freq_a = metrics_a
                            .iter()
                            .find(|m| m.name == "frequency")
                            .map(|m| m.value)
                            .unwrap_or(0.0);
                        let freq_b = metrics_b
                            .iter()
                            .find(|m| m.name == "frequency")
                            .map(|m| m.value)
                            .unwrap_or(0.0);
                        return freq_b
                            .partial_cmp(&freq_a)
                            .unwrap_or(std::cmp::Ordering::Equal);
                    }

                    std::cmp::Ordering::Equal
                });
            }

            // Fill table rows
            for (label_values, metrics) in rows {
                let mut row = String::new();
                row.push_str("| ");
                if !label_values.is_empty() {
                    row.push_str(&label_values.join(" | "));
                    row.push_str(" | ");
                }

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
