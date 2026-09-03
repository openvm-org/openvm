use std::array;

use openvm_stark_backend::{
    interaction::PermutationCheckBus, p3_field::PrimeField32, p3_maybe_rayon::prelude::*,
};

use super::{controller::dimensions::MemoryDimensions, has_nonzero_byte, online::LinearMemory};
use crate::{
    arch::AddressSpaceHostLayout,
    system::memory::{online::PAGE_SIZE, AddressMap},
};

mod air;
mod columns;
pub mod public_values;
mod trace;
mod tree;

pub use air::*;
pub use columns::*;
pub(super) use trace::SerialReceiver;
pub use tree::*;

#[cfg(test)]
mod tests;

pub struct MemoryMerkleChip<const DIGEST_WIDTH: usize, F> {
    pub air: MemoryMerkleAir<DIGEST_WIDTH>,
    final_state: Option<FinalState<DIGEST_WIDTH, F>>,
    overridden_height: Option<usize>,
    pub(crate) top_tree: Vec<[F; DIGEST_WIDTH]>,
    /// Used for metric collection purposes only
    #[cfg(feature = "metrics")]
    pub(crate) current_height: usize,
}
#[derive(Debug)]
pub struct FinalState<const DIGEST_WIDTH: usize, F> {
    rows: Vec<MemoryMerkleCols<F, DIGEST_WIDTH>>,
    init_root: [F; DIGEST_WIDTH],
    final_root: [F; DIGEST_WIDTH],
}

impl<const DIGEST_WIDTH: usize, F: PrimeField32> MemoryMerkleChip<DIGEST_WIDTH, F> {
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
            overridden_height: None,
            top_tree: vec![],
            #[cfg(feature = "metrics")]
            current_height: 0,
        }
    }
    pub fn set_overridden_height(&mut self, override_height: usize) {
        self.overridden_height = Some(override_height);
    }
}

#[tracing::instrument(level = "info", skip_all)]
pub(crate) fn memory_to_vec_partition<F: PrimeField32, const N: usize>(
    memory: &AddressMap,
    md: &MemoryDimensions,
) -> Vec<(u64, [F; N])> {
    let mut partition = (0..memory.mem.len())
        .into_par_iter()
        .map(move |as_idx| {
            let space_mem = memory.mem[as_idx].as_slice();
            let addr_space_layout = memory.config[as_idx].layout;
            let cell_size = addr_space_layout.size();
            let leaf_bytes = cell_size * N;
            debug_assert_eq!(PAGE_SIZE % leaf_bytes, 0);

            memory.touched_pages[as_idx]
                .touched_byte_ranges(space_mem.len())
                .into_par_iter()
                .flat_map(|(start, end)| {
                    debug_assert_eq!(start % leaf_bytes, 0);
                    space_mem[start..end]
                        .par_chunks(leaf_bytes)
                        .enumerate()
                        .filter_map(move |(local_idx, leaf)| {
                            if leaf.len() != leaf_bytes || !has_nonzero_byte(leaf) {
                                return None;
                            }
                            let byte_offset = start + local_idx * leaf_bytes;
                            let leaf_idx = byte_offset / leaf_bytes;
                            (leaf_idx < 1 << md.address_height).then(|| {
                                (
                                    md.label_to_index((as_idx as u32, leaf_idx as u32)),
                                    array::from_fn(|i| unsafe {
                                        // SAFETY: `byte_offset` identifies a complete leaf in
                                        // `space_mem`, and `i < N`, so this cell-sized slice is
                                        // entirely within that leaf.
                                        addr_space_layout.to_field(
                                            &*core::ptr::slice_from_raw_parts(
                                                space_mem.as_ptr().add(byte_offset + i * cell_size),
                                                cell_size,
                                            ),
                                        )
                                    }),
                                )
                            })
                        })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    partition.sort_unstable_by_key(|(index, _)| *index);
    partition
}
