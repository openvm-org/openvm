#[derive(Clone, Debug, Default)]
pub struct CycleTracker {
    /// Stack of span names, with most recent at the end
    stack: Vec<String>,
}

impl CycleTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts a new cycle tracker span for the given name.
    /// If a span already exists for the given name, it ends the existing span and pushes a new one to the vec.
    pub fn start(&mut self, name: String) {
        self.stack.push(name);
    }

    /// Ends the cycle tracker span for the given name.
    /// If no span exists for the given name, it panics.
    pub fn end(&mut self, name: String) {
        let stack_top = self.stack.pop();
        assert_eq!(stack_top.unwrap(), name, "Stack top does not match name");
    }

    /// Get full name of span with all parent names separated by ";" in flamegraph format
    pub fn get_full_name(&self) -> String {
        self.stack.join(";")
    }
}

#[cfg(feature = "bench-metrics")]
mod emit {
    use metrics::counter;

    use super::CycleTracker;

    impl CycleTracker {
        pub fn increment_opcode(&self, (dsl_ir, opcode): &(Option<String>, String)) {
            counter!("total_cycles").increment(1u64);
            let labels = [
                ("opcode", opcode.clone()),
                ("dsl_ir", dsl_ir.clone().unwrap_or_default()),
                ("cycle_tracker_span", self.get_full_name()),
            ];
            counter!("frequency", &labels).increment(1u64);
        }

        pub fn increment_cells_used(
            &self,
            (dsl_ir, opcode, air_name): &(Option<String>, String, String),
            trace_cells_used: usize,
        ) {
            if trace_cells_used == 0 {
                return;
            }
            let labels = [
                ("air_name", air_name.clone()),
                ("opcode", opcode.clone()),
                ("dsl_ir", dsl_ir.clone().unwrap_or_default()),
                ("cycle_tracker_span", self.get_full_name()),
            ];
            counter!("cells_used", &labels).increment(trace_cells_used as u64);
        }
    }
}
