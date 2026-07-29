use std::collections::HashMap;

use crate::types::MetricDb;

const OPCODE_COUNT_METRIC: &str = "opcode_count";

pub fn generate_instruction_count_table(db: &MetricDb) -> String {
    let mut markdown_output = String::new();
    let mut aggregated: HashMap<(Option<String>, String), f64> = HashMap::new();
    let mut has_group = false;

    for (label_keys, metrics_dict) in &db.dict_by_label_types {
        let Some(opcode_index) = label_keys.iter().position(|key| key == "opcode") else {
            continue;
        };

        let group_index = label_keys.iter().position(|k| k == "group");
        has_group |= group_index.is_some();

        for (label_values, metrics) in metrics_dict {
            let Some(count) = metrics
                .iter()
                .find(|metric| metric.name == OPCODE_COUNT_METRIC)
                .map(|metric| metric.value)
            else {
                continue;
            };
            let opcode = label_values.get(opcode_index).cloned().unwrap_or_default();
            let group = group_index.and_then(|index| label_values.get(index).cloned());
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
    sorted_entries.sort_by(|(key_a, count_a), (key_b, count_b)| {
        count_b.total_cmp(count_a).then_with(|| key_a.cmp(key_b))
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

    #[test]
    fn aggregates_opcode_counts_across_label_sets() {
        let mut db = MetricDb::default();
        db.flat_dict.insert(
            Labels(vec![
                ("group".to_string(), "app_proof".to_string()),
                ("opcode".to_string(), "ADD".to_string()),
                ("segment".to_string(), "0".to_string()),
            ]),
            vec![
                Metric::new(OPCODE_COUNT_METRIC.to_string(), 3.0),
                Metric::new("frequency".to_string(), 99.0),
            ],
        );
        db.flat_dict.insert(
            Labels(vec![
                ("group".to_string(), "app_proof".to_string()),
                ("opcode".to_string(), "ADD".to_string()),
            ]),
            vec![Metric::new(OPCODE_COUNT_METRIC.to_string(), 4.0)],
        );
        db.separate_by_label_types();

        let markdown = generate_instruction_count_table(&db);

        assert!(markdown.contains("| app_proof | ADD | 7 |"));
        assert!(!markdown.contains("99"));
    }
}
