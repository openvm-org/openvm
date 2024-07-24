use std::collections::{btree_map::Entry, BTreeMap};
use std::fmt::Display;
use std::marker::PhantomData;

use p3_field::PrimeField32;

use self::span::CycleTrackerSpan;

use super::VirtualMachine;

pub mod span;

#[derive(Debug, Default)]
pub struct CycleTracker<const WORD_SIZE: usize, F> {
    pub instances: BTreeMap<String, Vec<CycleTrackerSpan>>,
    pub order: Vec<String>,
    pub num_active_instances: usize,
    _marker: PhantomData<F>,
}

impl<const WORD_SIZE: usize, F: PrimeField32> CycleTracker<WORD_SIZE, F> {
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
        vm: &VirtualMachine<WORD_SIZE, F>,
        rows: &[F],
        clock_cycle: usize,
        timestamp: usize,
    ) {
        let cycle_tracker_span = CycleTrackerSpan::start(
            rows.len(),
            clock_cycle,
            timestamp,
            vm.memory_chip.accesses.len(),
            vm.field_arithmetic_chip.operations.len(),
            vm.field_extension_chip.operations.len(),
            vm.range_checker.count.len(),
            vm.poseidon2_chip.rows.len(),
            vm.input_stream.len(),
        );
        match self.instances.entry(name.clone()) {
            Entry::Occupied(mut entry) => {
                // If a span already exists here, end it before starting a new one
                let spans = entry.get_mut();
                let ct_last = spans.last_mut().unwrap();
                ct_last.end(
                    rows.len(),
                    clock_cycle,
                    timestamp,
                    vm.memory_chip.accesses.len(),
                    vm.field_arithmetic_chip.operations.len(),
                    vm.field_extension_chip.operations.len(),
                    vm.range_checker.count.len(),
                    vm.poseidon2_chip.rows.len(),
                    vm.input_stream.len(),
                );
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
        vm: &VirtualMachine<WORD_SIZE, F>,
        rows: &[F],
        clock_cycle: usize,
        timestamp: usize,
    ) {
        match self.instances.entry(name.clone()) {
            Entry::Occupied(mut entry) => {
                let spans = entry.get_mut();
                let last = spans.last_mut().unwrap();
                last.end(
                    rows.len(),
                    clock_cycle,
                    timestamp,
                    vm.memory_chip.accesses.len(),
                    vm.field_arithmetic_chip.operations.len(),
                    vm.field_extension_chip.operations.len(),
                    vm.range_checker.count.len(),
                    vm.poseidon2_chip.rows.len(),
                    vm.input_stream.len(),
                );
            }
            Entry::Vacant(_) => {
                panic!("Cycle tracker instance {} does not exist", name);
            }
        }
        self.num_active_instances -= 1;
    }

    /// Ends all active cycle tracker spans. Called at the end of execution to close any open spans.
    pub fn end_all_active(
        &mut self,
        vm: &VirtualMachine<WORD_SIZE, F>,
        rows: &[F],
        clock_cycle: usize,
        timestamp: usize,
    ) {
        let active_instances: Vec<String> = self
            .order
            .iter()
            .filter(|name| self.instances.get(*name).unwrap().last().unwrap().is_active)
            .cloned()
            .collect();

        for name in active_instances {
            self.end(name, vm, rows, clock_cycle, timestamp);
        }
    }

    /// Prints the cycle tracker to the console.
    pub fn print(&self) {
        println!("{}", self);
    }
}

impl<const WORD_SIZE: usize, F: PrimeField32> Display for CycleTracker<WORD_SIZE, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.instances.is_empty() {
            return Ok(());
        }
        for name in &self.order {
            let spans = self.instances.get(name).unwrap();
            let num_spans = spans.len();
            for (i, span) in spans.iter().enumerate() {
                let postfix = if num_spans == 1 {
                    String::new()
                } else {
                    format!(" {}", i)
                };
                writeln!(f, "span [{}{}]:", name, postfix)?;
                writeln!(f, "  - cpu_rows: {}", span.end.cpu_rows)?;
                writeln!(f, "  - clock_cycles: {}", span.end.clock_cycles)?;
                writeln!(f, "  - time_elapsed: {}", span.end.time_elapsed)?;
                writeln!(f, "  - mem_accesses: {}", span.end.mem_accesses)?;
                writeln!(
                    f,
                    "  - field_arithmetic_ops: {}",
                    span.end.field_arithmetic_ops
                )?;
                writeln!(
                    f,
                    "  - field_extension_ops: {}",
                    span.end.field_extension_ops
                )?;
                writeln!(
                    f,
                    "  - range_checker_count: {}",
                    span.end.range_checker_count
                )?;
                writeln!(f, "  - poseidon2_rows: {}", span.end.poseidon2_rows)?;
                writeln!(f, "  - input_stream_len: {}", span.end.input_stream_len)?;
                writeln!(f)?;
            }
        }
        Ok(())
    }
}
