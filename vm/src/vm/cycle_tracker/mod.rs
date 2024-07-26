use std::collections::{btree_map::Entry, BTreeMap};
use std::fmt::Display;
use std::marker::PhantomData;

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
        rows: &[F],
        clock_cycle: usize,
        timestamp: usize,
        vm_metrics: &BTreeMap<String, usize>,
    ) {
        let cycle_tracker_span =
            CycleTrackerSpan::start(rows.len(), clock_cycle, timestamp, vm_metrics);
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
        rows: &[F],
        clock_cycle: usize,
        timestamp: usize,
        vm_metrics: &BTreeMap<String, usize>,
    ) {
        match self.instances.entry(name.clone()) {
            Entry::Occupied(mut entry) => {
                let spans = entry.get_mut();
                let last = spans.last_mut().unwrap();
                last.end(rows.len(), clock_cycle, timestamp, vm_metrics);
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
            for (i, span) in spans.iter().enumerate() {
                let postfix = if num_spans == 1 {
                    String::new()
                } else {
                    format!(" {}", i)
                };
                writeln!(f, "span [{}{}]:", name, postfix)?;
                writeln!(f, "{}", span)?;
            }
        }
        Ok(())
    }
}
