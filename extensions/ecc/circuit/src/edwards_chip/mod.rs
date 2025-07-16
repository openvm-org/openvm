mod add;
pub use add::*;

mod utils;

#[cfg(test)]
mod tests;

use openvm_algebra_circuit::FieldExprVecHeapStep;
use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};
use openvm_mod_circuit_builder::FieldExpressionCoreAir;
use openvm_rv32_adapters::Rv32VecHeapAdapterAir;

pub(crate) type EdwardsAir<const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize> =
    VmAirWrapper<
        Rv32VecHeapAdapterAir<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreAir,
    >;

pub(crate) type EdwardsStep<const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize> =
    FieldExprVecHeapStep<NUM_READS, BLOCKS, BLOCK_SIZE>;

pub(crate) type EdwardsChip<
    F,
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = NewVmChipWrapper<
    F,
    EdwardsAir<NUM_READS, BLOCKS, BLOCK_SIZE>,
    EdwardsStep<NUM_READS, BLOCKS, BLOCK_SIZE>,
    MatrixRecordArena<F>,
>;
