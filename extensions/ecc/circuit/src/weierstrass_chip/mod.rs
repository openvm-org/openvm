mod add_ne;
mod double;

pub use add_ne::*;
pub use double::*;

#[cfg(test)]
mod tests;

use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionStep};
use openvm_rv32_adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterStep};

pub(crate) type WeierstrassAir<
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = VmAirWrapper<
    Rv32VecHeapAdapterAir<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

pub(crate) type WeierstrassStep<
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = FieldExpressionStep<Rv32VecHeapAdapterStep<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>;

pub(crate) type WeierstrassChip<
    F,
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = NewVmChipWrapper<
    F,
    WeierstrassAir<NUM_READS, BLOCKS, BLOCK_SIZE>,
    WeierstrassStep<NUM_READS, BLOCKS, BLOCK_SIZE>,
    MatrixRecordArena<F>,
>;
