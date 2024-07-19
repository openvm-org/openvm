use std::collections::{btree_map::Entry, BTreeMap};
use std::fmt::Display;
use std::marker::PhantomData;

use p3_field::PrimeField32;

use super::VirtualMachine;

#[derive(Debug, Default)]
pub struct CycleTracker<const WORD_SIZE: usize, F> {
    pub instances: BTreeMap<String, CycleTrackerSpan>,
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

    pub fn start(&mut self, name: String, vm: &VirtualMachine<WORD_SIZE, F>, rows: &[F]) {
        match self.instances.entry(name.clone()) {
            Entry::Occupied(_) => {
                panic!("Cycle tracker instance {} is already active", name.clone());
            }
            Entry::Vacant(_) => {
                self.instances.insert(
                    name.clone(),
                    CycleTrackerSpan::start(rows.len(), vm.memory_chip.accesses.len()),
                );
            }
        }
        self.num_active_instances += 1;
        self.order.push(name);
    }

    pub fn end(&mut self, name: String, vm: &VirtualMachine<WORD_SIZE, F>, rows: &[F]) {
        match self.instances.entry(name.clone()) {
            Entry::Occupied(mut span) => {
                span.get_mut()
                    .end(rows.len(), vm.memory_chip.accesses.len());
            }
            Entry::Vacant(_) => {
                panic!("Cycle tracker instance {} does not exist", name);
            }
        }
        self.num_active_instances -= 1;
    }

    pub fn end_all(&mut self, vm: &VirtualMachine<WORD_SIZE, F>, rows: &[F]) {
        for name in &self.order {
            self.end(name, vm, &[]);
        }
    }
}

impl<const WORD_SIZE: usize, F: PrimeField32> Display for CycleTracker<WORD_SIZE, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.instances.is_empty() {
            return Ok(());
        }
        for name in &self.order {
            let span = self.instances.get(name).unwrap();
            writeln!(f, "span [{}]:", name)?;
            writeln!(f, "  - cpu_rows: {}", span.end.cpu_rows)?;
            writeln!(f, "  - mem_accesses: {}", span.end.mem_accesses)?;
            writeln!(f)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct CycleTrackerData {
    pub cpu_rows: usize,
    pub mem_accesses: usize,
}

#[derive(Debug)]
pub struct CycleTrackerSpan {
    pub is_active: bool,
    pub start: CycleTrackerData,
    pub end: CycleTrackerData,
}

impl CycleTrackerSpan {
    pub fn start(start_row: usize, start_mem: usize) -> Self {
        Self {
            is_active: true,
            start: CycleTrackerData {
                cpu_rows: start_row,
                mem_accesses: start_mem,
            },
            end: CycleTrackerData {
                cpu_rows: 0,
                mem_accesses: 0,
            },
        }
    }

    pub fn end(&mut self, end_row: usize, end_mem: usize) {
        self.is_active = false;
        self.end.cpu_rows = end_row - self.start.cpu_rows;
        self.end.mem_accesses = end_mem - self.start.mem_accesses;
    }
}
