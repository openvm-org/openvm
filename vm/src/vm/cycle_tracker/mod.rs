use std::{
    collections::{btree_map::Entry, BTreeMap},
    fmt::Display,
};

use afs_stark_backend::prover::metrics::format_number_with_underscores;

use self::span::CycleTrackerSpan;
use super::metrics::VmMetrics;
use crate::vm::cycle_tracker::span::CanDiff;

pub mod span;

#[derive(Clone, Debug, Default)]
pub struct CycleTracker<M: CanDiff> {
    pub instances: BTreeMap<String, Vec<CycleTrackerSpan<M>>>,
    pub order: Vec<String>,
    pub num_active_instances: usize,
}

impl<M: CanDiff> CycleTracker<M> {
    pub fn new() -> Self {
        Self {
            instances: BTreeMap::new(),
            order: vec![],
            num_active_instances: 0,
        }
    }

    /// Starts a new cycle tracker span for the given name.
    /// If a span already exists for the given name, it ends the existing span and pushes a new one to the vec.
    pub fn start(&mut self, name: String, metrics: M) {
        let cycle_tracker_span = CycleTrackerSpan::start(metrics);
        match self.instances.entry(name.clone()) {
            Entry::Occupied(mut entry) => {
                let spans = entry.get_mut();
                let ct_last = spans.last_mut().unwrap();
                if ct_last.is_active {
                    panic!("Attempting to start another cycle tracker span named '{}' while previous is still active", name);
                }
                spans.push(cycle_tracker_span);
            }
            Entry::Vacant(_) => {
                self.instances
                    .insert(name.clone(), vec![cycle_tracker_span]);
                self.order.push(name);
            }
        }

        self.num_active_instances += 1;
    }

    /// Ends the cycle tracker span for the given name.
    /// If no span exists for the given name, it panics.
    pub fn end(&mut self, name: String, metrics: M) {
        match self.instances.entry(name.clone()) {
            Entry::Occupied(mut entry) => {
                let spans = entry.get_mut();
                let last = spans.last_mut().unwrap();
                last.end(metrics);
            }
            Entry::Vacant(_) => {
                panic!("Cycle tracker instance {} does not exist", name);
            }
        }
        self.num_active_instances -= 1;
    }
}

pub trait CanPrint {
    /// Prints the cycle tracker to the logger at INFO level.
    fn print(&self);
}

impl<M: CanDiff> CanPrint for CycleTracker<M>
where
    CycleTracker<M>: Display,
{
    fn print(&self) {
        tracing::info!("{}", self);
        if self.num_active_instances != 0 {
            tracing::warn!(
                "There are {} unclosed cycle tracker instances",
                self.num_active_instances
            );
        }
    }
}

impl Display for CycleTracker<VmMetrics> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.instances.is_empty() {
            return Ok(());
        }
        for name in &self.order {
            let spans = self.instances.get(name).unwrap();
            let num_spans = spans.len();

            if num_spans == 0 {
                continue;
            }

            let mut total_vm_metrics = std::collections::HashMap::new();
            let mut total_opcode_counts = std::collections::HashMap::new();
            let mut total_dsl_counts = std::collections::HashMap::new();
            let mut total_opcode_trace_cells = std::collections::HashMap::new();

            for span in spans {
                for (key, value) in &span.metrics.chip_metrics {
                    *total_vm_metrics.entry(key.clone()).or_insert(0) += value;
                }
                for (key, value) in &span.metrics.opcode_counts {
                    *total_opcode_counts.entry(key.clone()).or_insert(0) += value;
                }
                for (key, value) in &span.metrics.dsl_counts {
                    *total_dsl_counts.entry(key.clone()).or_insert(0) += value;
                }
                for (key, value) in &span.metrics.opcode_trace_cells {
                    *total_opcode_trace_cells.entry(key.clone()).or_insert(0) += value;
                }
            }

            writeln!(
                f,
                "span [{}] ({}):",
                name,
                format_number_with_underscores(num_spans)
            )?;
            for (key, value) in &total_vm_metrics {
                let avg_value = value / num_spans;
                if num_spans == 1 {
                    writeln!(f, "  - {}: {}", key, format_number_with_underscores(*value))?;
                } else {
                    writeln!(
                        f,
                        "  - tot_{}: {}",
                        key,
                        format_number_with_underscores(*value)
                    )?;
                    writeln!(
                        f,
                        "  - avg_{}: {}",
                        key,
                        format_number_with_underscores(avg_value)
                    )?;
                }
            }

            let mut sorted_opcode_counts: Vec<(&String, &usize)> =
                total_opcode_counts.iter().collect();
            sorted_opcode_counts.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

            for (key, value) in sorted_opcode_counts {
                if *value > 0 {
                    writeln!(f, "  - {}: {}", key, format_number_with_underscores(*value))?;
                }
            }

            let mut sorted_dsl_counts: Vec<(&String, &usize)> = total_dsl_counts.iter().collect();
            sorted_dsl_counts.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

            for (key, value) in sorted_dsl_counts {
                if *value > 0 {
                    writeln!(f, "  - {}: {}", key, format_number_with_underscores(*value))?;
                }
            }

            let mut sorted_opcode_trace_cells: Vec<(&String, &usize)> =
                total_opcode_trace_cells.iter().collect();
            sorted_opcode_trace_cells.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

            for (key, value) in sorted_opcode_trace_cells {
                if *value > 0 {
                    writeln!(f, "  - {}: {}", key, format_number_with_underscores(*value))?;
                }
            }

            writeln!(f)?;
        }
        Ok(())
    }
}
