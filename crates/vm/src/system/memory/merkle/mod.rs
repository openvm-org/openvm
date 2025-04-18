use openvm_stark_backend::{interaction::PermutationCheckBus, p3_field::PrimeField32};

use super::controller::dimensions::MemoryDimensions;
mod air;
mod columns;
mod trace;

pub mod tree;

pub use air::*;
pub use columns::*;
pub(super) use trace::SerialReceiver;

// #[cfg(test)]
// mod tests;

pub struct MemoryMerkleChip<const CHUNK: usize, F> {
    pub air: MemoryMerkleAir<CHUNK>,
    final_state: Option<FinalState<CHUNK, F>>,
    overridden_height: Option<usize>,
}
#[derive(Debug)]
pub struct FinalState<const CHUNK: usize, F> {
    rows: Vec<MemoryMerkleCols<F, CHUNK>>,
    init_root: [F; CHUNK],
    final_root: [F; CHUNK],
}

impl<const CHUNK: usize, F: PrimeField32> MemoryMerkleChip<CHUNK, F> {
    /// `compression_bus` is the bus for direct (no-memory involved) interactions to call the
    /// cryptographic compression function.
    pub fn new(
        memory_dimensions: MemoryDimensions,
        merkle_bus: PermutationCheckBus,
        compression_bus: PermutationCheckBus,
    ) -> Self {
        assert!(memory_dimensions.as_height > 0);
        assert!(memory_dimensions.address_height > 0);
        Self {
            air: MemoryMerkleAir {
                memory_dimensions,
                merkle_bus,
                compression_bus,
            },
            final_state: None,
            overridden_height: None,
        }
    }
    pub fn set_overridden_height(&mut self, override_height: usize) {
        self.overridden_height = Some(override_height);
    }
}
