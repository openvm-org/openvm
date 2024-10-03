use std::fmt::Display;

use afs_stark_backend::prover::metrics::format_number_with_underscores;

use super::{CanDiff, SpanRef};
use crate::vm::metrics::VmMetrics;

#[derive(Debug, Clone)]
pub struct CycleTrackerSpan<M: CanDiff> {
    pub is_active: bool,
    pub metrics: M,
    /// The name of the parent span, if any
    pub parent: Option<SpanRef>,
}

impl<M: CanDiff> CycleTrackerSpan<M> {
    pub fn start(metrics: M, parent: Option<SpanRef>) -> Self {
        Self {
            is_active: true,
            metrics,
            parent,
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
        for (key, value) in &self.metrics.chip_heights {
            writeln!(f, "  - {}: {}", key, format_number_with_underscores(*value))?;
        }

        let mut sorted_opcode_counts: Vec<_> = self.metrics.counts.iter().collect();
        sorted_opcode_counts.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

        for ((dsl_ir, opcode), value) in sorted_opcode_counts {
            if *value > 0 {
                writeln!(
                    f,
                    "  - {};{}: {}",
                    dsl_ir.as_ref().unwrap_or(&String::new()),
                    opcode,
                    format_number_with_underscores(*value)
                )?;
            }
        }

        let mut sorted_opcode_trace_cells: Vec<_> = self.metrics.trace_cells.iter().collect();
        sorted_opcode_trace_cells.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

        for ((dsl_ir, opcode, air_name), value) in sorted_opcode_trace_cells {
            if *value > 0 {
                writeln!(
                    f,
                    "  - {};{};{}: {}",
                    dsl_ir.as_ref().unwrap_or(&String::new()),
                    opcode,
                    air_name,
                    format_number_with_underscores(*value)
                )?;
            }
        }

        Ok(())
    }
}
