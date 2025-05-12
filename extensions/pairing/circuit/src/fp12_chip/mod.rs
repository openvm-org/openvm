mod add;
mod mul;
mod sub;

pub use add::*;
pub use mul::*;
use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionStep};
use openvm_rv32_adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterStep};
pub use sub::*;

#[cfg(test)]
mod tests;

pub(crate) type Fp12Air<
    const NUM_READS: usize,
    const INPUT_BLOCKS: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = VmAirWrapper<
    Rv32VecHeapAdapterAir<NUM_READS, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

pub(crate) type Fp12Step<
    const NUM_READS: usize,
    const INPUT_BLOCKS: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = FieldExpressionStep<
    Rv32VecHeapAdapterStep<NUM_READS, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
>;

pub(crate) type Fp12Chip<
    F,
    const NUM_READS: usize,
    const INPUT_BLOCKS: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = NewVmChipWrapper<
    F,
    Fp12Air<NUM_READS, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE>,
    Fp12Step<NUM_READS, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE>,
>;
