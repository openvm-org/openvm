use openvm_stark_backend::{interaction::PermutationCheckBus, p3_field::PrimeField32};
use rustc_hash::FxHashSet;

use super::{controller::dimensions::MemoryDimensions, Equipartition, MemoryImage};
mod air;
mod columns;
mod trace;
mod tree;

pub use air::*;
pub use columns::*;
pub(super) use trace::SerialReceiver;

// TODO: add back
// #[cfg(test)]
// mod tests;

pub struct MemoryMerkleChip<const CHUNK: usize, F> {
    pub air: MemoryMerkleAir<CHUNK>,
    touched_nodes: FxHashSet<(usize, u32, u32)>,
    final_state: Option<FinalState<CHUNK, F>>,
    // TODO(AG): how are these two different? Doesn't one just end up being copied to the other?
    trace_height: Option<usize>,
    overridden_height: Option<usize>,
}
#[derive(Debug)]
struct FinalState<const CHUNK: usize, F> {
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
        let mut touched_nodes = FxHashSet::default();
        touched_nodes.insert((memory_dimensions.overall_height(), 0, 0));
        Self {
            air: MemoryMerkleAir {
                memory_dimensions,
                merkle_bus,
                compression_bus,
            },
            touched_nodes,
            final_state: None,
            trace_height: None,
            overridden_height: None,
        }
    }
    pub fn set_overridden_height(&mut self, override_height: usize) {
        self.overridden_height = Some(override_height);
    }
}

fn memory_to_partition<F: PrimeField32, const N: usize>(
    memory: &MemoryImage,
) -> Equipartition<F, N> {
    let mut memory_partition = Equipartition::new();
    for ((address_space, pointer), value) in memory.items() {
        let label = (address_space, pointer / N as u32);
        let chunk = memory_partition
            .entry(label)
            .or_insert_with(|| [F::default(); N]);
        chunk[(pointer % N as u32) as usize] = value;
    }
    memory_partition
}
