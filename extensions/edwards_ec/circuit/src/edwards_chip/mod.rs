mod add;
mod curves;
mod utils;

pub use add::*;

#[cfg(test)]
mod tests;

use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionFiller};
use openvm_rv32_adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterFiller};

pub type EdwardsAir<const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize> =
    VmAirWrapper<
        Rv32VecHeapAdapterAir<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreAir,
    >;

pub type EdwardsChip<F, const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize> =
    VmChipWrapper<
        F,
        FieldExpressionFiller<
            Rv32VecHeapAdapterFiller<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >,
    >;
