use std::collections::{btree_map::Entry, BTreeMap};
use std::fmt::Display;
use std::marker::PhantomData;

use p3_field::PrimeField32;

use super::VirtualMachine;

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
            for (i, span) in spans.into_iter().enumerate() {
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

#[derive(Debug)]
pub struct CycleTrackerData {
    pub cpu_rows: usize,
    pub clock_cycles: usize,
    pub time_elapsed: usize,
    pub mem_accesses: usize,
    pub field_arithmetic_ops: usize,
    pub field_extension_ops: usize,
    pub range_checker_count: usize,
    pub poseidon2_rows: usize,
    pub input_stream_len: usize,
}

#[derive(Debug)]
pub struct CycleTrackerSpan {
    pub is_active: bool,
    pub start: CycleTrackerData,
    pub end: CycleTrackerData,
}

impl CycleTrackerSpan {
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        start_cpu_rows: usize,
        start_clock_cycle: usize,
        start_timestamp: usize,
        start_mem_accesses: usize,
        start_field_arithmetic_ops: usize,
        start_field_extension_ops: usize,
        start_range_checker_count: usize,
        start_poseidon2_rows: usize,
        start_input_stream_len: usize,
    ) -> Self {
        Self {
            is_active: true,
            start: CycleTrackerData {
                cpu_rows: start_cpu_rows,
                clock_cycles: start_clock_cycle,
                time_elapsed: start_timestamp,
                mem_accesses: start_mem_accesses,
                field_arithmetic_ops: start_field_arithmetic_ops,
                field_extension_ops: start_field_extension_ops,
                range_checker_count: start_range_checker_count,
                poseidon2_rows: start_poseidon2_rows,
                input_stream_len: start_input_stream_len,
            },
            end: CycleTrackerData {
                cpu_rows: 0,
                clock_cycles: 0,
                time_elapsed: 0,
                mem_accesses: 0,
                field_arithmetic_ops: 0,
                field_extension_ops: 0,
                range_checker_count: 0,
                poseidon2_rows: 0,
                input_stream_len: 0,
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn end(
        &mut self,
        end_cpu_rows: usize,
        end_clock_cycle: usize,
        end_timestamp: usize,
        end_mem_accesses: usize,
        end_field_arithmetic_ops: usize,
        end_field_extension_ops: usize,
        end_range_checker_count: usize,
        end_poseidon2_rows: usize,
        end_input_stream_len: usize,
    ) {
        self.is_active = false;
        self.end.cpu_rows = end_cpu_rows - self.start.cpu_rows;
        self.end.clock_cycles = end_clock_cycle - self.start.clock_cycles;
        self.end.time_elapsed = end_timestamp - self.start.time_elapsed;
        self.end.mem_accesses = end_mem_accesses - self.start.mem_accesses;
        self.end.field_arithmetic_ops = end_field_arithmetic_ops - self.start.field_arithmetic_ops;
        self.end.field_extension_ops = end_field_extension_ops - self.start.field_extension_ops;
        self.end.range_checker_count = end_range_checker_count - self.start.range_checker_count;
        self.end.poseidon2_rows = end_poseidon2_rows - self.start.poseidon2_rows;
        self.end.input_stream_len = end_input_stream_len - self.start.input_stream_len;
    }
}
