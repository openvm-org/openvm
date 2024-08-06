use std::{
    collections::{btree_map::Entry, BTreeMap},
    fmt::Display,
    marker::PhantomData,
};

use p3_field::PrimeField32;

use self::span::CycleTrackerSpan;

pub mod span;

#[derive(Debug, Default)]
pub struct CycleTracker<F> {
    pub instances: BTreeMap<String, Vec<CycleTrackerSpan>>,
    pub order: Vec<String>,
    pub num_active_instances: usize,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> CycleTracker<F> {
    pub fn new() -> Self {
        Self {
            instances: BTreeMap::new(),
            order: vec![],
            num_active_instances: 0,
            _marker: PhantomData,
        }
    }

    /// Starts a new cycle tracker span for the given name.
    /// If a span already exists for the given name, it ends the existing span and pushes a new one to the vec.
    pub fn start(
        &mut self,
        name: String,
        vm_metrics: &BTreeMap<String, usize>,
        opcode_counts: &BTreeMap<String, usize>,
        dsl_counts: &BTreeMap<String, usize>,
        opcode_trace_cells: &BTreeMap<String, usize>,
    ) {
        let cycle_tracker_span =
            CycleTrackerSpan::start(vm_metrics, opcode_counts, dsl_counts, opcode_trace_cells);
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
    pub fn end(
        &mut self,
        name: String,
        vm_metrics: &BTreeMap<String, usize>,
        opcode_counts: &BTreeMap<String, usize>,
        dsl_counts: &BTreeMap<String, usize>,
        opcode_trace_cells: &BTreeMap<String, usize>,
    ) {
        match self.instances.entry(name.clone()) {
            Entry::Occupied(mut entry) => {
                let spans = entry.get_mut();
                let last = spans.last_mut().unwrap();
                last.end(vm_metrics, opcode_counts, dsl_counts, opcode_trace_cells);
            }
            Entry::Vacant(_) => {
                panic!("Cycle tracker instance {} does not exist", name);
            }
        }
        self.num_active_instances -= 1;
    }

    /// Prints the cycle tracker to the console.
    pub fn print(&self) {
        println!("{}", self);
        if self.num_active_instances != 0 {
            println!(
                "Warning: there are {} unclosed cycle tracker instances",
                self.num_active_instances
            );
        }
    }
}

impl<F: PrimeField32> Display for CycleTracker<F> {
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
                for (key, value) in &span.end.vm_metrics {
                    *total_vm_metrics.entry(key.clone()).or_insert(0) += value;
                }
                for (key, value) in &span.end.opcode_counts {
                    *total_opcode_counts.entry(key.clone()).or_insert(0) += value;
                }
                for (key, value) in &span.end.dsl_counts {
                    *total_dsl_counts.entry(key.clone()).or_insert(0) += value;
                }
                for (key, value) in &span.end.opcode_trace_cells {
                    *total_opcode_trace_cells.entry(key.clone()).or_insert(0) += value;
                }
            }

            writeln!(f, "span [{}] ({}):", name, num_spans)?;
            for (key, value) in &total_vm_metrics {
                let avg_value = value / num_spans;
                if num_spans == 1 {
                    writeln!(f, "  - {}: {}", key, value)?;
                } else {
                    writeln!(f, "  - tot_{}: {}", key, value)?;
                    writeln!(f, "  - avg_{}: {}", key, avg_value)?;
                }
            }

            let mut sorted_opcode_counts: Vec<(&String, &usize)> =
                total_opcode_counts.iter().collect();
            sorted_opcode_counts.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

            for (key, value) in sorted_opcode_counts {
                if *value > 0 {
                    writeln!(f, "  - {}: {}", key, value)?;
                }
            }

            let mut sorted_dsl_counts: Vec<(&String, &usize)> = total_dsl_counts.iter().collect();
            sorted_dsl_counts.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

            for (key, value) in sorted_dsl_counts {
                if *value > 0 {
                    writeln!(f, "  - {}: {}", key, value)?;
                }
            }

            let mut sorted_opcode_trace_cells: Vec<(&String, &usize)> =
                total_opcode_trace_cells.iter().collect();
            sorted_opcode_trace_cells.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

            for (key, value) in sorted_opcode_trace_cells {
                if *value > 0 {
                    writeln!(f, "  - {}: {}", key, value)?;
                }
            }

            writeln!(f)?;
        }
        Ok(())
    }
}
