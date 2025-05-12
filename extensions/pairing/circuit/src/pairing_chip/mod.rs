mod line;
mod miller_double_step;

pub use line::*;
pub use miller_double_step::*;

mod miller_double_and_add_step;
pub use miller_double_and_add_step::*;
use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionStep};
use openvm_rv32_adapters::{
    Rv32VecHeapAdapterAir, Rv32VecHeapAdapterStep, Rv32VecHeapTwoReadsAdapterAir,
    Rv32VecHeapTwoReadsAdapterStep,
};

/// Two types of PairingChips are implemented:
/// - PairingHeapAdapterChip: used by `EcLineMul013By013Chip`, `EcLineMul023By023Chip`
/// - PairingTwoReadsChip: used by `EcLineMulBy01234Chip`, `EcLineMulBy02345Chip`, `EvaluateLineChip`
pub(crate) type PairingHeapAdapterAir<
    const NUM_READS: usize,
    const INPUT_BLOCKS: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = VmAirWrapper<
    Rv32VecHeapAdapterAir<NUM_READS, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

pub(crate) type PairingHeapAdapterStep<
    const NUM_READS: usize,
    const INPUT_BLOCKS: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = FieldExpressionStep<
    Rv32VecHeapAdapterStep<NUM_READS, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
>;

pub(crate) type PairingHeapAdapterChip<
    F,
    const NUM_READS: usize,
    const INPUT_BLOCKS: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = NewVmChipWrapper<
    F,
    PairingHeapAdapterAir<NUM_READS, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE>,
    PairingHeapAdapterStep<NUM_READS, INPUT_BLOCKS, OUTPUT_BLOCKS, BLOCK_SIZE>,
>;

pub(crate) type PairingTwoReadsAir<
    const INPUT_BLOCKS1: usize,
    const INPUT_BLOCKS2: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = VmAirWrapper<
    Rv32VecHeapTwoReadsAdapterAir<
        INPUT_BLOCKS1,
        INPUT_BLOCKS2,
        OUTPUT_BLOCKS,
        BLOCK_SIZE,
        BLOCK_SIZE,
    >,
    FieldExpressionCoreAir,
>;

pub(crate) type PairingTwoReadsStep<
    const INPUT_BLOCKS1: usize,
    const INPUT_BLOCKS2: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = FieldExpressionStep<
    Rv32VecHeapTwoReadsAdapterStep<
        INPUT_BLOCKS1,
        INPUT_BLOCKS2,
        OUTPUT_BLOCKS,
        BLOCK_SIZE,
        BLOCK_SIZE,
    >,
>;

pub(crate) type PairingTwoReadsChip<
    F,
    const INPUT_BLOCKS1: usize,
    const INPUT_BLOCKS2: usize,
    const OUTPUT_BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = NewVmChipWrapper<
    F,
    PairingTwoReadsAir<INPUT_BLOCKS1, INPUT_BLOCKS2, OUTPUT_BLOCKS, BLOCK_SIZE>,
    PairingTwoReadsStep<INPUT_BLOCKS1, INPUT_BLOCKS2, OUTPUT_BLOCKS, BLOCK_SIZE>,
>;
