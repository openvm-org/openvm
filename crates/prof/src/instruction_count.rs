use std::collections::HashMap;

use crate::types::MetricDb;

pub fn generate_instruction_count_table(db: &MetricDb) -> String {
    let mut markdown_output = String::new();
    
    // Find the table that has "opcode" and "frequency"
    for (label_keys, metrics_dict) in &db.dict_by_label_types {
        let has_opcode = label_keys.iter().any(|k| k == "opcode");
        let has_frequency = metrics_dict
            .values()
            .flat_map(|metrics| metrics.iter().map(|m| m.name.clone()))
            .any(|name| name == "frequency");
        
        if !has_opcode || !has_frequency {
            continue;
        }
        
        // Aggregate frequencies by opcode (and group if present)
        let mut aggregated: HashMap<(Option<String>, String), f64> = HashMap::new();
        let group_index = label_keys.iter().position(|k| k == "group");
        let opcode_index = label_keys.iter().position(|k| k == "opcode");
        
        for (label_values, metrics) in metrics_dict {
            // Get opcode value
            let opcode = if let Some(idx) = opcode_index {
                label_values.get(idx).cloned().unwrap_or_default()
            } else {
                continue;
            };
            
            // Get group value if present
            let group = if let Some(idx) = group_index {
                label_values.get(idx).cloned()
            } else {
                None
            };
            
            // Get frequency value
            let frequency = metrics
                .iter()
                .find(|m| m.name == "frequency")
                .map(|m| m.value)
                .unwrap_or(0.0);
            
            // Aggregate by (group, opcode) tuple
            let key = (group, opcode);
            *aggregated.entry(key).or_insert(0.0) += frequency;
        }
        
        // Create table header
        let mut header_parts = Vec::new();
        if group_index.is_some() {
            header_parts.push("group".to_string());
        }
        header_parts.push("opcode".to_string());
        header_parts.push("frequency".to_string());
        
        let header = format!("| {} |", header_parts.join(" | "));
        let separator = format!("| {} |", vec!["---"; header_parts.len()].join(" | "));
        
        markdown_output.push_str(&header);
        markdown_output.push('\n');
        markdown_output.push_str(&separator);
        markdown_output.push('\n');
        
        // Sort by frequency (descending)
        let mut sorted_entries: Vec<_> = aggregated.into_iter().collect();
        sorted_entries.sort_by(|(_, freq_a), (_, freq_b)| {
            freq_b.partial_cmp(freq_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Generate rows
        for ((group, opcode), frequency) in sorted_entries {
            let formatted_freq = MetricDb::format_number(frequency);
            if let Some(group_val) = group {
                markdown_output.push_str(&format!("| {} | {} | {} |\n", group_val, opcode, formatted_freq));
            } else {
                markdown_output.push_str(&format!("| {} | {} |\n", opcode, formatted_freq));
            }
        }
        
        markdown_output.push('\n');
        break; // Only process the first matching table
    }
    
    markdown_output
}
