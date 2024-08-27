use std::fmt::Display;

use crate::vm::metrics::VmMetrics;

#[derive(Debug, Clone)]
pub struct CycleTrackerSpan<M: CanDiff> {
    pub is_active: bool,
    pub metrics: M,
}

pub trait CanDiff {
    fn diff(&mut self, another: &Self);
}

impl<M: CanDiff> CycleTrackerSpan<M> {
    pub fn start(metrics: M) -> Self {
        Self {
            is_active: true,
            metrics,
        }
    }

    pub fn end(&mut self, mut metrics: M) {
        self.is_active = false;
        metrics.diff(&self.metrics);
        self.metrics = metrics;
    }
}

impl Display for CycleTrackerSpan<VmMetrics> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (key, value) in &self.metrics.chip_metrics {
            writeln!(f, "  - {}: {}", key, value)?;
        }

        let mut sorted_opcode_counts: Vec<(&String, &usize)> =
            self.metrics.opcode_counts.iter().collect();
        sorted_opcode_counts.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

        for (key, value) in sorted_opcode_counts {
            if *value > 0 {
                writeln!(f, "  - {}: {}", key, value)?;
            }
        }

        let mut sorted_dsl_counts: Vec<(&String, &usize)> =
            self.metrics.dsl_counts.iter().collect();
        sorted_dsl_counts.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

        for (key, value) in sorted_dsl_counts {
            if *value > 0 {
                writeln!(f, "  - {}: {}", key, value)?;
            }
        }

        let mut sorted_opcode_trace_cells: Vec<(&String, &usize)> =
            self.metrics.opcode_trace_cells.iter().collect();
        sorted_opcode_trace_cells.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

        for (key, value) in sorted_opcode_trace_cells {
            if *value > 0 {
                writeln!(f, "  - {}: {}", key, value)?;
            }
        }

        Ok(())
    }
}
