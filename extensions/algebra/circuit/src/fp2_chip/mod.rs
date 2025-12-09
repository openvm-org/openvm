use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionFiller};
use openvm_rv32_adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterFiller};

use crate::FieldExprVecHeapExecutor;

mod addsub;
pub use addsub::*;
mod muldiv;
pub use muldiv::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

/// CHUNKS = BLOCK_SIZE / 4 (the number of 4-byte chunks per block)
pub type Fp2Air<const BLOCKS: usize, const BLOCK_SIZE: usize, const CHUNKS: usize> = VmAirWrapper<
    Rv32VecHeapAdapterAir<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE, CHUNKS, CHUNKS>,
    FieldExpressionCoreAir,
>;

/// CHUNKS = BLOCK_SIZE / 4 (the number of 4-byte chunks per block)
pub type Fp2Executor<const BLOCKS: usize, const BLOCK_SIZE: usize, const CHUNKS: usize> =
    FieldExprVecHeapExecutor<BLOCKS, BLOCK_SIZE, true, CHUNKS>;

/// CHUNKS = BLOCK_SIZE / 4 (the number of 4-byte chunks per block)
pub type Fp2Chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize, const CHUNKS: usize> =
    VmChipWrapper<
    F,
        FieldExpressionFiller<
            Rv32VecHeapAdapterFiller<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE, CHUNKS, CHUNKS>,
        >,
>;

#[cfg(test)]
mod tests;
