use std::array;

use openvm_stark_backend::{
    interaction::PermutationCheckBus, p3_field::PrimeField32, p3_maybe_rayon::prelude::*,
};

use super::{controller::dimensions::MemoryDimensions, online::LinearMemory, MemoryImage};
use crate::system::memory::online::PAGE_SIZE;

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
    (0..memory.mem.len())
        .into_par_iter()
        .map(move |as_idx| {
            let space_mem = memory.mem[as_idx].as_slice();
            let cell_size = memory.cell_size[as_idx];
            debug_assert_eq!(PAGE_SIZE % (cell_size * N), 0);

            let num_nonzero_pages = space_mem
                .par_chunks(PAGE_SIZE)
                .enumerate()
                .flat_map(|(idx, page)| {
                    if page.iter().any(|x| *x != 0) {
                        Some(idx + 1)
                    } else {
                        None
                    }
                })
                .max()
                .unwrap_or(0);

            let space_mem = &space_mem[..(num_nonzero_pages * PAGE_SIZE).min(space_mem.len())];
            let mut num_elements = space_mem.len() / (cell_size * N);
            // virtual memory may be larger than dimensions due to rounding up to page size
            num_elements = num_elements.min(1 << md.address_height);

            // TODO: handle different cell sizes better
            if cell_size == 1 {
                (0..num_elements)
                    .into_par_iter()
                    .map(move |idx| {
                        let byte_index = idx * cell_size * N;
                        unsafe {
                            let ptr = space_mem.as_ptr();
                            let src = ptr.add(byte_index);
                            (
                                md.label_to_index((as_idx as u32, idx as u32)),
                                array::from_fn(|i| {
                                    F::from_canonical_u8(core::ptr::read(src.add(i)))
                                }),
                            )
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                assert_eq!(cell_size, 4);
                (0..num_elements)
                    .into_par_iter()
                    .map(move |idx| {
                        let byte_index = idx * cell_size * N;
                        unsafe {
                            let ptr = space_mem.as_ptr();
                            let src = ptr.add(byte_index) as *const F;
                            (
                                md.label_to_index((as_idx as u32, idx as u32)),
                                array::from_fn(|i| core::ptr::read(src.add(i))),
                            )
                        }
                    })
                    .collect::<Vec<_>>()
            }
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
}
