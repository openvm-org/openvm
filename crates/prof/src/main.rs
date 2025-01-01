mod types;

use std::collections::HashMap;

use clap::Parser;
use serde::{Deserialize, Serialize};

use crate::types::{Labels, Metric, MetricDb, MetricsFile};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the metrics JSON file
    #[arg(value_name = "METRICS_JSON")]
    metrics_json: String,

    /// Path to the aggregation JSON file
    #[arg(long, value_name = "AGGREGATION_JSON")]
    aggregation_json: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AggregationFile {
    aggregations: Vec<Aggregation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Aggregation {
    name: String,
    group_by: Vec<String>,
    metrics: Vec<String>,
    operation: String,
}

impl MetricDb {
    fn new(metrics_file: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(metrics_file)?;
        let metrics: MetricsFile = serde_json::from_reader(file)?;

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

        db.separate_by_label_types();

        Ok(db)
    }

    fn add_to_flat_dict(&mut self, labels: Labels, metric: String, value: f64) {
        self.flat_dict
            .entry(labels)
            .or_default()
            .push(Metric::new(metric, value));
    }

    // Custom sorting function that ensures 'group' comes first.
    // Other keys are sorted alphabetically.
    fn custom_sort_label_keys(label_keys: &mut [String]) {
        // Prioritize 'group' by giving it the lowest possible sort value
        label_keys.sort_by_key(|key| {
            if key == "group" {
                (0, key.clone()) // Lowest priority for 'group'
            } else {
                (1, key.clone()) // Normal priority for other keys
            }
        });
    }

    fn separate_by_label_types(&mut self) {
        self.dict_by_label_types.clear();

        for (labels, metrics) in &self.flat_dict {
            // Get sorted label keys
            let mut label_keys: Vec<String> = labels.0.iter().map(|(key, _)| key.clone()).collect();
            Self::custom_sort_label_keys(&mut label_keys);

            // Create label_values based on sorted keys
            let label_dict: HashMap<String, String> = labels.0.iter().cloned().collect();

            let label_values: Vec<String> = label_keys
                .iter()
                .map(|key| label_dict.get(key).unwrap().clone())
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

    fn generate_markdown_tables(&self) -> String {
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

    fn generate_aggregation_tables(&self, aggregations: &[Aggregation]) -> String {
        let mut markdown_output = String::new();
        let group_tuple = vec!["group".to_string()];

        // Get metrics grouped by "group" label
        if let Some(metrics_dict) = self.dict_by_label_types.get(&group_tuple) {
            let mut group_to_metrics: HashMap<String, Vec<Metric>> = HashMap::new();

            // Collect metrics for each group
            for (group_values, metrics) in metrics_dict {
                let group_name = &group_values[0];
                let agg_metrics: Vec<Metric> = metrics
                    .iter()
                    .filter(|metric| aggregations.iter().any(|a| a.name == metric.name))
                    .cloned()
                    .collect();

                if !agg_metrics.is_empty() {
                    group_to_metrics
                        .entry(group_name.clone())
                        .or_default()
                        .extend(agg_metrics);
                }
            }

            if !group_to_metrics.is_empty() {
                // Get all unique metric names
                let mut metric_names: Vec<String> = group_to_metrics
                    .values()
                    .flat_map(|metrics| metrics.iter().map(|m| m.name.clone()))
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                metric_names.sort();

                // Create table header
                let header = format!("| group | {} |", metric_names.join(" | "));
                let separator =
                    format!("| --- | {} |", vec!["---"; metric_names.len()].join(" | "));
                markdown_output.push_str(&header);
                markdown_output.push('\n');
                markdown_output.push_str(&separator);
                markdown_output.push('\n');

                // Fill table rows
                for (group_name, metrics) in group_to_metrics {
                    let mut row = format!("| {} |", group_name);

                    for metric_name in &metric_names {
                        let metric_str = metrics
                            .iter()
                            .find(|m| &m.name == metric_name)
                            .map(|m| format!(" {} |", Self::format_number(m.value)))
                            .unwrap_or_default();

                        row.push_str(&metric_str);
                    }

                    markdown_output.push_str(&row);
                    markdown_output.push('\n');
                }
                markdown_output.push('\n');
            }
        }

        markdown_output
    }

    fn read_aggregations(
        aggregation_file: &str,
    ) -> Result<Vec<Aggregation>, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(aggregation_file)?;
        let aggregation_data: AggregationFile = serde_json::from_reader(file)?;
        Ok(aggregation_data.aggregations)
    }

    fn apply_aggregations(&mut self, aggregations: &[Aggregation]) {
        for aggregation in aggregations {
            let mut group_by_dict: HashMap<Vec<String>, f64> = HashMap::new();

            if aggregation.operation == "sum" || aggregation.operation == "unique" {
                for (tuple_keys, metrics_dict) in &self.dict_by_label_types {
                    // Skip if not all group_by keys are present in tuple_keys
                    if !aggregation
                        .group_by
                        .iter()
                        .all(|key| tuple_keys.contains(key))
                    {
                        continue;
                    }

                    for (tuple_values, metrics) in metrics_dict {
                        // Create a mapping from label keys to values
                        let label_dict: HashMap<_, _> =
                            tuple_keys.iter().zip(tuple_values.iter()).collect();

                        // Extract values for group_by keys
                        let group_by_values: Vec<String> = aggregation
                            .group_by
                            .iter()
                            .map(|key| label_dict[key].clone())
                            .collect();

                        // Process metrics
                        for metric in metrics {
                            if aggregation.metrics.contains(&metric.name) {
                                match aggregation.operation.as_str() {
                                    "sum" => {
                                        *group_by_dict
                                            .entry(group_by_values.clone())
                                            .or_default() += metric.value;
                                    }
                                    "unique" => {
                                        let entry = group_by_dict
                                            .entry(group_by_values.clone())
                                            .or_default();
                                        if *entry != 0.0 && *entry != metric.value {
                                            println!(
                                                "[WARN] Overwriting {}: previous value = {}, new value = {}",
                                                metric.name, entry, metric.value
                                            );
                                        }
                                        *entry = metric.value;
                                    }
                                    _ => unreachable!(),
                                }
                            }
                        }
                    }
                }

                // Add aggregated metrics back to the database
                for (group_by_values, agg_value) in group_by_dict {
                    let labels = Labels(
                        aggregation
                            .group_by
                            .iter()
                            .zip(group_by_values.iter())
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                    );

                    let metric = Metric::new(aggregation.name.clone(), agg_value);

                    // Check if metric already exists
                    if let Some(metrics) = self.flat_dict.get_mut(&labels) {
                        if let Some(existing_metric) =
                            metrics.iter_mut().find(|m| m.name == aggregation.name)
                        {
                            if existing_metric.value != agg_value {
                                println!(
                                    "[WARN] Overwriting {}: previous value = {}, new value = {}",
                                    aggregation.name, existing_metric.value, agg_value
                                );
                            }
                            existing_metric.value = agg_value;
                        } else {
                            metrics.push(metric);
                        }
                    } else {
                        self.flat_dict.insert(labels, vec![metric]);
                    }
                }
            }
        }

        self.separate_by_label_types();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut db = MetricDb::new(&args.metrics_json)?;

    let mut markdown_output = String::new();

    if let Some(aggregation_file) = args.aggregation_json {
        let aggregations = MetricDb::read_aggregations(&aggregation_file)?;
        db.apply_aggregations(&aggregations);

        // Generate aggregation tables
        let agg_tables = db.generate_aggregation_tables(&aggregations);
        markdown_output.push_str(&agg_tables);

        // Add detailed metrics in a collapsible section
        markdown_output.push_str("\n<details>\n<summary>Detailed Metrics</summary>\n\n");
        markdown_output.push_str(&db.generate_markdown_tables());
        markdown_output.push_str("</details>\n\n");
    } else {
        markdown_output.push_str(&db.generate_markdown_tables());
    }

    println!("{}", markdown_output);
    Ok(())
}
