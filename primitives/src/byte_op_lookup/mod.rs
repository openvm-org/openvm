use std::sync::atomic::AtomicU32;

mod air;
mod bus;
mod columns;
mod trace;

#[cfg(test)]
pub mod tests;

pub use air::*;
pub use bus::*;
pub use columns::*;

// Lookup chip for operations on size NUM_BITS integers. Currently has pre-processed columns
// for (x + y) % 256 and x ^ y. Interactions are of form [x, y, z, op], where x and y are
// integers, op is an opcode (see ByteOperationLookupOpcode in air.rs), and z is x op y.

#[derive(Debug)]
pub struct ByteOperationLookupChip<const NUM_BITS: usize> {
    pub air: ByteOperationLookupAir<NUM_BITS>,
    count_add: Vec<AtomicU32>,
    count_xor: Vec<AtomicU32>,
}

impl<const NUM_BITS: usize> ByteOperationLookupChip<NUM_BITS> {
    pub fn new(bus: ByteOperationLookupBus) -> Self {
        let num_rows = (1 << NUM_BITS) * (1 << NUM_BITS);
        let count_add = (0..num_rows).map(|_| AtomicU32::new(0)).collect();
        let count_xor = (0..num_rows).map(|_| AtomicU32::new(0)).collect();
        Self {
            air: ByteOperationLookupAir::new(bus),
            count_add,
            count_xor,
        }
    }

    pub fn bus(&self) -> ByteOperationLookupBus {
        self.air.bus
    }

    pub fn air_width(&self) -> usize {
        NUM_BYTE_OP_LOOKUP_COLS
    }

    pub fn add_count(&self, x: u32, y: u32, op: ByteOperationLookupOpcode) {
        let idx = (x as usize) * (1 << NUM_BITS) + (y as usize);
        assert!(
            idx < self.count_add.len(),
            "range exceeded: {} >= {}",
            idx,
            self.count_add.len()
        );
        let val_atomic = if op == ByteOperationLookupOpcode::ADD {
            &self.count_add[idx]
        } else {
            &self.count_xor[idx]
        };
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn clear(&self) {
        for i in 0..self.count_add.len() {
            self.count_add[i].store(0, std::sync::atomic::Ordering::Relaxed);
            self.count_xor[i].store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
}
