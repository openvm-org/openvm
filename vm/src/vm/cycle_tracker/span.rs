use std::fmt::Display;

use crate::vm::metrics::VmMetrics;

#[derive(Debug, Clone)]
pub struct CycleTrackerSpan {
    pub is_active: bool,
    pub start: VmMetrics,
    pub end: VmMetrics,
}

impl CycleTrackerSpan {
    #[allow(clippy::too_many_arguments)]
    pub fn start(metrics: VmMetrics) -> Self {
        Self {
            is_active: true,
            start: metrics,
            end: Default::default(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn end(&mut self, metrics: VmMetrics) {
        self.is_active = false;
        for (key, value) in metrics.chip_metrics {
            let diff = value - self.start.chip_metrics.get(&key).unwrap();
            self.end.chip_metrics.insert(key, diff);
        }
        for (key, value) in metrics.opcode_counts {
            let diff = value - self.start.opcode_counts.get(&key).unwrap_or(&0);
            self.end.opcode_counts.insert(key, diff);
        }
        for (key, value) in metrics.dsl_counts {
            let diff = value - self.start.dsl_counts.get(&key).unwrap_or(&0);
            self.end.dsl_counts.insert(key, diff);
        }
        for (key, value) in metrics.opcode_trace_cells {
            let diff = value - self.start.opcode_trace_cells.get(&key).unwrap_or(&0);
            self.end.opcode_trace_cells.insert(key.clone(), diff);
        }
    }
}

impl Display for CycleTrackerSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (key, value) in &self.end.chip_metrics {
            writeln!(f, "  - {}: {}", key, value)?;
        }

        let mut sorted_opcode_counts: Vec<(&String, &usize)> =
            self.end.opcode_counts.iter().collect();
        sorted_opcode_counts.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

        for (key, value) in sorted_opcode_counts {
            if *value > 0 {
                writeln!(f, "  - {}: {}", key, value)?;
            }
        }

        let mut sorted_dsl_counts: Vec<(&String, &usize)> = self.end.dsl_counts.iter().collect();
        sorted_dsl_counts.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

        for (key, value) in sorted_dsl_counts {
            if *value > 0 {
                writeln!(f, "  - {}: {}", key, value)?;
            }
        }

        let mut sorted_opcode_trace_cells: Vec<(&String, &usize)> =
            self.end.opcode_trace_cells.iter().collect();
        sorted_opcode_trace_cells.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

        for (key, value) in sorted_opcode_trace_cells {
            if *value > 0 {
                writeln!(f, "  - {}: {}", key, value)?;
            }
        }

        Ok(())
    }
}
