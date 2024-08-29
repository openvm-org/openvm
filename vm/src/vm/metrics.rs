use core::fmt;
use std::{collections::BTreeMap, fmt::Display};

use afs_stark_backend::prover::metrics::format_number_with_underscores;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::vm::cycle_tracker::span::CanDiff;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VmMetrics {
    pub chip_metrics: BTreeMap<String, usize>,
    pub opcode_counts: BTreeMap<String, usize>,
    pub dsl_counts: BTreeMap<String, usize>,
    pub opcode_trace_cells: BTreeMap<String, usize>,
}

impl Display for VmMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let opcode_counts: Vec<(String, usize)> = self
            .opcode_counts
            .clone()
            .into_iter()
            .sorted_by(|a, b| b.1.cmp(&a.1))
            .collect();

        let dsl_counts: Vec<(String, usize)> = self
            .dsl_counts
            .clone()
            .into_iter()
            .sorted_by(|a, b| b.1.cmp(&a.1))
            .collect();

        writeln!(f, "<details>")?;
        writeln!(f, "<summary>")?;
        writeln!(f)?;
        writeln!(f, "### Custom VM metrics")?;
        writeln!(f)?;
        writeln!(f, "</summary>")?;
        writeln!(f)?;

        writeln!(f, "| Name | Value |")?;
        writeln!(f, "|------|------:|")?;
        for (name, value) in self.chip_metrics.iter() {
            writeln!(
                f,
                "| {:<20} | `{:>15}` |",
                name,
                format_number_with_underscores(*value)
            )?;
        }

        writeln!(f)?;
        writeln!(f, "#### Opcode metrics")?;
        writeln!(f, "| Name | Frequency | Trace Cells Contributed |")?;
        writeln!(f, "|------|------:|-----:|")?;
        for (name, value) in opcode_counts.iter() {
            let cell_count = *self.opcode_trace_cells.get(name).unwrap_or(&0);
            writeln!(
                f,
                "| {:<20} | `{:>15}` | `{:>15}` |",
                name,
                format_number_with_underscores(*value),
                format_number_with_underscores(cell_count)
            )?;
        }
        for (name, value) in self.opcode_trace_cells.iter() {
            if !self.opcode_counts.contains_key(name) {
                // this should never happen
                writeln!(
                    f,
                    "| {:<20} | 0 | `{:>15}` |",
                    name,
                    format_number_with_underscores(*value)
                )?;
            }
        }

        writeln!(f)?;
        writeln!(f, "### DSL counts")?;
        writeln!(f, "How many opcodes each DSL instruction generates:")?;
        writeln!(f, "| Name | Count |")?;
        writeln!(f, "|------|------:|")?;
        for (name, value) in dsl_counts.iter() {
            writeln!(
                f,
                "| {:<20} | `{:>15}` |",
                name,
                format_number_with_underscores(*value)
            )?;
        }

        writeln!(f, "</details>")?;
        Ok(())
    }
}

impl CanDiff for VmMetrics {
    fn diff(&mut self, start: &Self) {
        *self = Self {
            chip_metrics: count_diff(&start.chip_metrics, &self.chip_metrics),
            opcode_counts: count_diff(&start.opcode_counts, &self.opcode_counts),
            dsl_counts: count_diff(&start.dsl_counts, &self.dsl_counts),
            opcode_trace_cells: count_diff(&start.opcode_trace_cells, &self.opcode_trace_cells),
        };
    }
}

fn count_diff(
    start: &BTreeMap<String, usize>,
    end: &BTreeMap<String, usize>,
) -> BTreeMap<String, usize> {
    let mut ret = BTreeMap::new();
    for (key, value) in end {
        let diff = value - start.get(key).unwrap_or(&0);
        ret.insert(key.clone(), diff);
    }
    ret
}
