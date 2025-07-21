mod add_ne;
mod double;

pub use add_ne::*;
pub use double::*;
#[cfg(test)]
use openvm_algebra_circuit::FieldExprVecHeapStep;
use openvm_circuit::arch::VmAirWrapper;
use openvm_mod_circuit_builder::FieldExpressionCoreAir;
use openvm_rv32_adapters::Rv32VecHeapAdapterAir;

pub(crate) type WeierstrassAir<
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = VmAirWrapper<
    Rv32VecHeapAdapterAir<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

#[cfg(test)]
pub(crate) type WeierstrassStep<
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = FieldExprVecHeapStep<NUM_READS, BLOCKS, BLOCK_SIZE>;
