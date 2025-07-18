use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionFiller};
use openvm_rv32_adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterFiller};

use crate::FieldExprVecHeapStep;

mod addsub;
pub use addsub::*;

mod muldiv;
pub use muldiv::*;

pub type Fp2Air<const BLOCKS: usize, const BLOCK_SIZE: usize> = VmAirWrapper<
    Rv32VecHeapAdapterAir<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

pub type Fp2Step<const BLOCKS: usize, const BLOCK_SIZE: usize> =
    FieldExprVecHeapStep<2, BLOCKS, BLOCK_SIZE>;

pub type Fp2Chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> = VmChipWrapper<
    F,
    FieldExpressionFiller<Rv32VecHeapAdapterFiller<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>,
>;
