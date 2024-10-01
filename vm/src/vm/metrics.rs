use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::vm::cycle_tracker::CanDiff;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VmMetrics {
    pub chip_heights: BTreeMap<String, usize>,
    pub opcode_counts: BTreeMap<String, usize>,
    pub dsl_counts: BTreeMap<String, usize>,
    pub opcode_trace_cells: BTreeMap<String, usize>,
    pub dsl_trace_cells: BTreeMap<String, usize>,
}

#[cfg(feature = "bench-metrics")]
mod emit {
    use itertools::Itertools;
    use metrics::counter;

    use super::VmMetrics;

    impl VmMetrics {
        pub fn emit(&self) {
            for (name, value) in self.chip_heights.iter() {
                let labels = [("chip_name", name.clone())];
                counter!("rows_used", &labels).absolute(*value as u64);
            }

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

            for (name, value) in opcode_counts.iter() {
                let cell_count = *self.opcode_trace_cells.get(name).unwrap_or(&0);
                let labels = [("opcode", name.clone())];
                counter!("frequency", &labels).absolute(*value as u64);
                counter!("cells_used", &labels).absolute(cell_count as u64);
            }

            for (name, value) in dsl_counts.iter() {
                let cell_count = *self.dsl_trace_cells.get(name).unwrap_or(&0);
                let labels = [("dsl_ir", name.clone())];
                counter!("frequency", &labels).absolute(*value as u64);
                counter!("cells_used", &labels).absolute(cell_count as u64);
            }
        }
    }
}

impl CanDiff for VmMetrics {
    fn diff(&mut self, start: &Self) {
        *self = Self {
            chip_heights: count_diff(&start.chip_heights, &self.chip_heights),
            opcode_counts: count_diff(&start.opcode_counts, &self.opcode_counts),
            dsl_counts: count_diff(&start.dsl_counts, &self.dsl_counts),
            opcode_trace_cells: count_diff(&start.opcode_trace_cells, &self.opcode_trace_cells),
            dsl_trace_cells: count_diff(&start.dsl_trace_cells, &self.dsl_trace_cells),
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
