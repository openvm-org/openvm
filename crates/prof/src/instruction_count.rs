use std::collections::HashMap;

use crate::types::MetricDb;

const OPCODE_COUNT_METRIC: &str = "opcode_count";
const LEGACY_FREQUENCY_METRIC: &str = "frequency";

pub fn generate_instruction_count_table(db: &MetricDb) -> String {
    let has_metric = |metric_name: &str| {
        db.dict_by_label_types
            .iter()
            .any(|(label_keys, metrics_dict)| {
                label_keys.iter().any(|key| key == "opcode")
                    && metrics_dict
                        .values()
                        .flatten()
                        .any(|metric| metric.name == metric_name)
            })
    };
    let metric_name = if has_metric(OPCODE_COUNT_METRIC) {
        OPCODE_COUNT_METRIC
    } else if has_metric(LEGACY_FREQUENCY_METRIC) {
        LEGACY_FREQUENCY_METRIC
    } else {
        return String::new();
    };

    let mut markdown_output = String::new();
    let mut aggregated: HashMap<(Option<String>, String), f64> = HashMap::new();
    let mut has_group = false;

    for (label_keys, metrics_dict) in &db.dict_by_label_types {
        let opcode_index = label_keys.iter().position(|key| key == "opcode");
        let has_selected_metric = metrics_dict
            .values()
            .flatten()
            .any(|metric| metric.name == metric_name);
        let (Some(opcode_index), true) = (opcode_index, has_selected_metric) else {
            continue;
        };

        let group_index = label_keys.iter().position(|k| k == "group");
        has_group |= group_index.is_some();

        for (label_values, metrics) in metrics_dict {
            let opcode = label_values.get(opcode_index).cloned().unwrap_or_default();
            let group = group_index.and_then(|index| label_values.get(index).cloned());
            let count = metrics
                .iter()
                .find(|metric| metric.name == metric_name)
                .map(|metric| metric.value)
                .unwrap_or(0.0);
            *aggregated.entry((group, opcode)).or_insert(0.0) += count;
        }
    }

    if aggregated.is_empty() {
        return String::new();
    }

    let mut header_parts = Vec::new();
    if has_group {
        header_parts.push("group");
    }
    header_parts.extend(["opcode", "Instructions"]);

    let header = format!("| {} |", header_parts.join(" | "));
    let separator = format!("| {} |", vec!["---"; header_parts.len()].join(" | "));

    markdown_output.push_str(&header);
    markdown_output.push('\n');
    markdown_output.push_str(&separator);
    markdown_output.push('\n');

    let mut sorted_entries: Vec<_> = aggregated.into_iter().collect();
    sorted_entries.sort_by(|(_, count_a), (_, count_b)| {
        count_b
            .partial_cmp(count_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for ((group, opcode), count) in sorted_entries {
        let formatted_count = MetricDb::format_number(count);
        if has_group {
            markdown_output.push_str(&format!(
                "| {} | {} | {} |\n",
                group.unwrap_or_default(),
                opcode,
                formatted_count
            ));
        } else {
            markdown_output.push_str(&format!("| {} | {} |\n", opcode, formatted_count));
        }
    }

    markdown_output.push('\n');
    markdown_output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Labels, Metric};

    fn metric_db(metrics: &[(&str, f64)]) -> MetricDb {
        let mut db = MetricDb::default();
        let labels = Labels(vec![
            ("group".to_string(), "app_proof".to_string()),
            ("opcode".to_string(), "ADD".to_string()),
        ]);
        db.flat_dict.insert(
            labels,
            metrics
                .iter()
                .map(|(name, value)| Metric::new((*name).to_string(), *value))
                .collect(),
        );
        db.separate_by_label_types();
        db
    }

    #[test]
    fn canonical_opcode_count_takes_precedence() {
        let db = metric_db(&[(LEGACY_FREQUENCY_METRIC, 99.0), (OPCODE_COUNT_METRIC, 7.0)]);

        let markdown = generate_instruction_count_table(&db);

        assert!(markdown.contains("| group | opcode | Instructions |"));
        assert!(markdown.contains("| app_proof | ADD | 7 |"));
        assert!(!markdown.contains("99"));
    }

    #[test]
    fn legacy_frequency_remains_supported() {
        let db = metric_db(&[(LEGACY_FREQUENCY_METRIC, 11.0)]);

        let markdown = generate_instruction_count_table(&db);

        assert!(markdown.contains("| group | opcode | Instructions |"));
        assert!(markdown.contains("| app_proof | ADD | 11 |"));
    }
}
