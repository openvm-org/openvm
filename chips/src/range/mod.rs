// Adapted from Valida

pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

use std::sync::Arc;
use std::{collections::BTreeMap, sync::atomic::AtomicU32};

#[derive(Default)]
pub struct RangeCheckerChip<const MAX: u32> {
    /// The index for the Range Checker bus.
    bus_index: usize,
    pub count: BTreeMap<u32, Arc<AtomicU32>>,
}

impl<const MAX: u32> RangeCheckerChip<MAX> {
    pub fn new(bus_index: usize) -> Self {
        Self {
            bus_index,
            count: BTreeMap::new(),
        }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    pub fn add_count(&mut self, val: u32) {
        self.count
            .entry(val)
            .or_insert_with(|| Arc::new(AtomicU32::new(0)))
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
