use std::{
    collections::{btree_map::Entry, BTreeMap},
    fmt::Display,
};

use afs_stark_backend::prover::metrics::format_number_with_underscores;

use self::span::CycleTrackerSpan;
use super::metrics::VmMetrics;

mod span;

#[derive(Clone, Debug, Default)]
pub struct CycleTracker<M: CanDiff> {
    pub instances: BTreeMap<String, Vec<CycleTrackerSpan<M>>>,
    pub order: Vec<String>,
    pub num_active_instances: usize,
    /// Stack of span names, with most recent at the end
    stack: Vec<SpanRef>,
}

// Internal struct to use as a pointer to a CycleTrackerSpan within
// CycleTracker.instances
#[derive(Clone, Debug, derive_new::new)]
pub struct SpanRef {
    name: String,
    index: usize,
}

impl<M: CanDiff> CycleTracker<M> {
    pub fn new() -> Self {
        Self {
            instances: BTreeMap::new(),
            order: vec![],
            num_active_instances: 0,
            stack: vec![],
        }
    }

    /// Starts a new cycle tracker span for the given name.
    /// If a span already exists for the given name, it ends the existing span and pushes a new one to the vec.
    pub fn start(&mut self, name: String, metrics: M) {
        let parent = self.stack.last().cloned();
        let cycle_tracker_span = CycleTrackerSpan::start(metrics, parent);
        let mut span_index = 0;
        match self.instances.entry(name.clone()) {
            Entry::Occupied(mut entry) => {
                let spans = entry.get_mut();
                let ct_last = spans.last_mut().unwrap();
                if ct_last.is_active {
                    panic!("Attempting to start another cycle tracker span named '{}' while previous is still active", name);
                }
                span_index = spans.len();
                spans.push(cycle_tracker_span);
            }
            Entry::Vacant(_) => {
                self.instances
                    .insert(name.clone(), vec![cycle_tracker_span]);
                self.order.push(name.clone());
            }
        }

        self.num_active_instances += 1;
        self.stack.push(SpanRef::new(name, span_index));
    }

    /// Ends the cycle tracker span for the given name.
    /// If no span exists for the given name, it panics.
    pub fn end(&mut self, name: String, metrics: M) {
        let stack_top = self.stack.pop();
        assert_eq!(
            stack_top.unwrap().name,
            name,
            "Stack top does not match name"
        );
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

    #[allow(dead_code)]
    /// Get full name of span with all parent names separated by ";" in flamegraph format
    fn get_full_name(&self, span_ref: SpanRef) -> String {
        let mut span = &self.instances[&span_ref.name][span_ref.index];
        let mut full_name = span_ref.name;
        loop {
            let parent = span.parent.as_ref();
            if parent.is_none() {
                break;
            }
            let parent = parent.unwrap();
            full_name = format!("{};{}", &parent.name, full_name);
            span = &self.instances[&parent.name][parent.index];
        }
        full_name
    }
}

pub trait CanDiff {
    fn diff(&mut self, another: &Self);
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

#[cfg(feature = "bench-metrics")]
mod emit {
    use metrics::counter;

    use super::{CycleTracker, SpanRef};
    use crate::vm::metrics::VmMetrics;

    impl CycleTracker<VmMetrics> {
        pub fn emit(&self) {
            if self.instances.is_empty() {
                return;
            }
            for name in &self.order {
                let spans = self.instances.get(name).unwrap().clone();
                let num_spans = spans.len();

                if num_spans == 0 {
                    continue;
                }
                counter!("num_spans", &[("cycle_tracker_span", name.clone())])
                    .absolute(num_spans as u64);

                for (i, span) in spans.into_iter().enumerate() {
                    let full_name = self.get_full_name(SpanRef::new(name.clone(), i));
                    for (key, value) in span.metrics.chip_heights {
                        let labels = [
                            ("chip_name", key),
                            ("cycle_tracker_span", full_name.clone()),
                        ];
                        counter!("rows_used", &labels).increment(value as u64);
                    }
                    for (key, value) in span.metrics.opcode_counts {
                        if value > 0 {
                            let labels =
                                [("opcode", key), ("cycle_tracker_span", full_name.clone())];
                            counter!("frequency", &labels).increment(value as u64);
                        }
                    }
                    for (key, value) in span.metrics.opcode_trace_cells {
                        if value > 0 {
                            let labels =
                                [("opcode", key), ("cycle_tracker_span", full_name.clone())];
                            counter!("cells_used", &labels).increment(value as u64);
                        }
                    }
                    for (key, value) in span.metrics.dsl_counts {
                        if value > 0 {
                            let labels =
                                [("dsl_ir", key), ("cycle_tracker_span", full_name.clone())];
                            counter!("frequency", &labels).absolute(value as u64);
                        }
                    }
                }
            }
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
                for (key, value) in &span.metrics.chip_heights {
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
            let mut sorted_opcode_counts: Vec<(&String, &usize)> =
                total_opcode_counts.iter().collect();
            sorted_opcode_counts.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value
            let mut sorted_dsl_counts: Vec<(&String, &usize)> = total_dsl_counts.iter().collect();
            sorted_dsl_counts.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value
            let mut sorted_opcode_trace_cells: Vec<(&String, &usize)> =
                total_opcode_trace_cells.iter().collect();
            sorted_opcode_trace_cells.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

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

            for (key, value) in sorted_opcode_counts {
                if *value > 0 {
                    writeln!(f, "  - {}: {}", key, format_number_with_underscores(*value))?;
                }
            }

            for (key, value) in sorted_dsl_counts {
                if *value > 0 {
                    writeln!(f, "  - {}: {}", key, format_number_with_underscores(*value))?;
                }
            }

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
