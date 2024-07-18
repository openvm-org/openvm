use std::collections::{btree_map::Entry, BTreeMap};
use std::fmt::Display;

#[derive(Debug, Default)]
pub struct CycleTracker {
    pub instances: BTreeMap<String, CycleTrackerSpan>,
    pub num_active_instances: usize,
}

impl CycleTracker {
    pub fn new() -> Self {
        Self {
            instances: BTreeMap::new(),
            num_active_instances: 0,
        }
    }

    pub fn start(&mut self, name: String, start_row: usize) {
        match self.instances.entry(name.clone()) {
            Entry::Occupied(_) => {
                panic!("Cycle tracker instance {} is already active", name.clone());
            }
            Entry::Vacant(_) => {
                self.instances
                    .insert(name.clone(), CycleTrackerSpan::start(start_row));
            }
        }
        self.num_active_instances += 1;
    }

    pub fn end(&mut self, name: String, end_row: usize) {
        match self.instances.entry(name.clone()) {
            Entry::Occupied(mut span) => {
                span.get_mut().end(end_row);
            }
            Entry::Vacant(_) => {
                panic!("Cycle tracker instance {} does not exist", name);
            }
        }
        self.num_active_instances -= 1;
    }
}

impl Display for CycleTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.instances.is_empty() {
            return Ok(());
        }
        for (name, span) in &self.instances {
            writeln!(f, "span [{}]:", name)?;
            writeln!(f, "  - cpu_rows: {}", span.cpu_rows)?;
            writeln!(f)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct CycleTrackerSpan {
    pub is_active: bool,
    pub start_row: usize,
    pub cpu_rows: usize,
}

impl CycleTrackerSpan {
    pub fn start(start_row: usize) -> Self {
        Self {
            is_active: true,
            start_row,
            cpu_rows: 0,
        }
    }

    pub fn end(&mut self, end_row: usize) {
        self.is_active = false;
        self.cpu_rows = end_row - self.start_row;
    }
}
