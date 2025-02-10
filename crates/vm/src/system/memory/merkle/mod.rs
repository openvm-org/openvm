use openvm_stark_backend::p3_field::PrimeField32;

use super::{controller::dimensions::MemoryDimensions, PagedVec, PAGE_SIZE};
mod air;
mod bus;
mod columns;
mod trace;

pub use air::*;
pub use bus::*;
pub use columns::*;
pub(super) use trace::SerialReceiver;

#[cfg(test)]
mod tests;

pub struct MemoryMerkleChip<const CHUNK: usize, F> {
    pub air: MemoryMerkleAir<CHUNK>,
    touched_nodes: PagedVec<i32, PAGE_SIZE>,
    num_touched_nonleaves: usize,
    final_state: Option<FinalState<CHUNK, F>>,
    overridden_height: Option<usize>,
}
#[derive(Debug)]
struct FinalState<const CHUNK: usize, F> {
    rows: Vec<MemoryMerkleCols<F, CHUNK>>,
    init_root: [F; CHUNK],
    final_root: [F; CHUNK],
}

impl<const CHUNK: usize, F: PrimeField32> MemoryMerkleChip<CHUNK, F> {
    /// `compression_bus` is the bus for direct (no-memory involved) interactions to call the cryptographic compression function.
    pub fn new(
        memory_dimensions: MemoryDimensions,
        merkle_bus: MemoryMerkleBus,
        compression_bus: DirectCompressionBus,
    ) -> Self {
        assert!(memory_dimensions.as_height > 0);
        assert!(memory_dimensions.address_height > 0);
        Self {
            air: MemoryMerkleAir {
                memory_dimensions,
                merkle_bus,
                compression_bus,
            },
            touched_nodes: PagedVec::new(2 << memory_dimensions.overall_height()),
            num_touched_nonleaves: 1,
            final_state: None,
            overridden_height: None,
        }
    }
    pub fn set_overridden_height(&mut self, override_height: usize) {
        self.overridden_height = Some(override_height);
    }

    fn encode(&self, height: usize, as_label: u32, address_label: u32) -> usize {
        let label = ((as_label as usize) << self.air.memory_dimensions.address_height)
            + address_label as usize;
        debug_assert!(label & ((1 << height) - 1) == 0);
        (1 << (self.air.memory_dimensions.overall_height() - height)) + (label >> height)
    }

    fn add_point(&mut self, idx: usize, delta: i32) {
        if idx >= self.touched_nodes.memory_size() {
            return;
        }
        if let Some(value) = self.touched_nodes.get_mut(idx) {
            *value += delta;
        } else {
            self.touched_nodes.set(idx, delta);
        }
    }

    pub fn touch_range(&mut self, address_space: u32, address: u32, len: u32) {
        let as_label = address_space - self.air.memory_dimensions.as_offset;
        let first_address_label = address / CHUNK as u32;
        let last_address_label = (address + len - 1) / CHUNK as u32;
        let mut left = self.encode(0, as_label, first_address_label);
        let mut right = self.encode(0, as_label, last_address_label);
        while left > 0 {
            self.add_point(left, 1);
            self.add_point(right + 1, -1);
            left /= 2;
            right /= 2;
        }
    }
}
