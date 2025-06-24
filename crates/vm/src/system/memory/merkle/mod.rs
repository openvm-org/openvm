use openvm_stark_backend::{
    interaction::PermutationCheckBus, p3_field::PrimeField32, p3_maybe_rayon::prelude::*,
};

use super::{controller::dimensions::MemoryDimensions, MemoryImage};

mod air;
mod columns;
pub mod public_values;
mod trace;
mod tree;

pub use air::*;
pub use columns::*;
pub(super) use trace::SerialReceiver;
pub use tree::*;

// TODO: add back
// #[cfg(test)]
// mod tests;

pub struct MemoryMerkleChip<const CHUNK: usize, F> {
    pub air: MemoryMerkleAir<CHUNK>,
    final_state: Option<FinalState<CHUNK, F>>,
    // TODO(AG): how are these two different? Doesn't one just end up being copied to the other?
    trace_height: Option<usize>,
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
        assert!(memory_dimensions.addr_space_height > 0);
        assert!(memory_dimensions.address_height > 0);
        Self {
            air: MemoryMerkleAir {
                memory_dimensions,
                merkle_bus,
                compression_bus,
            },
            final_state: None,
            trace_height: None,
            overridden_height: None,
        }
    }
    pub fn set_overridden_height(&mut self, override_height: usize) {
        self.overridden_height = Some(override_height);
    }
}

#[tracing::instrument(level = "info", skip_all)]
fn memory_to_vec_partition<F: PrimeField32, const N: usize>(
    memory: &MemoryImage,
    md: &MemoryDimensions,
) -> Vec<(u64, [F; N])> {
    let mut memory_partition = Vec::new();
    for ((address_space, pointer), value) in memory.items().collect::<Vec<_>>() {
        if pointer as usize / N >= md.address_height {
            // the memory image may be larger than the memory dimensions due to rounding up to page
            // sizes
            continue;
        }
        let label = md.label_to_index((address_space, pointer / N as u32));
        if memory_partition
            .last()
            .is_none_or(|(last_label, _)| *last_label < label)
        {
            memory_partition.push((label, [F::ZERO; N]));
        }
        memory_partition.last_mut().unwrap().1[(pointer % N as u32) as usize] = value;
    }
    memory_partition
}
