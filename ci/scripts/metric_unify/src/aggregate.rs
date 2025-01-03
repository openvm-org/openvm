use std::{collections::HashSet, path::Path};

use crate::metric::{AggregationEntry, AggregationFile, AggregationOperation, Metric};

/// Load aggregation metrics from a file
pub fn load_aggregation_metrics<P: AsRef<Path>>(
    aggregation_file_path: P,
) -> Result<Vec<AggregationEntry>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(aggregation_file_path)?;
    let aggregation_file: AggregationFile = serde_json::from_reader(file)?;
    Ok(aggregation_file.aggregations)
}

/// Generate aggregation tables
pub fn aggregate_metrics(agg_entries: Vec<AggregationEntry>, metrics: Vec<Metric>) -> Vec<Metric> {
    let mut results = Vec::new();
    for agg_entry in agg_entries {
        let group_by = &agg_entry.group_by;
        let name = &agg_entry.name;

        // 1. Filter metrics by group_by(primary_labels) and name(metric_name)
        let filtered_metrics_by_primary_labels: Vec<_> = metrics
            .iter()
            .filter(|m| group_by.iter().all(|g| m.primary_labels.contains(g)) && name == &m.name)
            .collect();
        if filtered_metrics_by_primary_labels.is_empty() {
            continue;
        }

        // 2. Group filtered_metrics by secondary_labels
        let secondary_labels_set: HashSet<Vec<String>> = filtered_metrics_by_primary_labels
            .iter()
            .map(|m| m.secondary_labels.clone())
            .collect();
        let grouped_metrics_by_secondary_labels =
            secondary_labels_set.into_iter().map(|secondary_labels| {
                filtered_metrics_by_primary_labels
                    .iter()
                    .filter(|m| m.secondary_labels == secondary_labels)
                    .collect::<Vec<_>>()
            });

        // 3. Aggregate metrics by secondary_labels and operation
        let aggregated_metrics: Vec<Metric> = grouped_metrics_by_secondary_labels
            .map(|grouped_metrics| {
                let secondary_labels = grouped_metrics[0].secondary_labels.clone();
                let aggregated_value: f64 =
                    grouped_metrics
                        .into_iter()
                        .fold(0.0, |acc, m| match agg_entry.operation {
                            AggregationOperation::Sum => acc + m.value,
                            AggregationOperation::Unique => {
                                assert!(acc == 0.0 || acc == m.value);
                                m.value
                            }
                        });
                Metric {
                    name: name.to_string(),
                    primary_labels: group_by.clone(),
                    secondary_labels,
                    value: aggregated_value,
                    ..Default::default()
                }
            })
            .collect();

        // 4. Generate table
        results.extend(aggregated_metrics);
    }

    results
}
