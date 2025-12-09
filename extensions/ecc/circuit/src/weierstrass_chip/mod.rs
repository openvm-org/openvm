mod add_ne;
mod curves;
mod double;

pub use add_ne::*;
pub use double::*;

#[cfg(test)]
mod tests;

use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionFiller};
use openvm_rv32_adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterFiller};

/// CHUNKS = BLOCK_SIZE / 4 (the number of 4-byte chunks per block)
pub type WeierstrassAir<
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CHUNKS: usize,
> = VmAirWrapper<
    Rv32VecHeapAdapterAir<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE, CHUNKS, CHUNKS>,
        FieldExpressionCoreAir,
    >;

/// CHUNKS = BLOCK_SIZE / 4 (the number of 4-byte chunks per block)
pub type WeierstrassChip<
    F,
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const CHUNKS: usize,
> = VmChipWrapper<
        F,
        FieldExpressionFiller<
        Rv32VecHeapAdapterFiller<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE, CHUNKS, CHUNKS>,
        >,
    >;
