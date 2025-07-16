use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};
use openvm_mod_circuit_builder::FieldExpressionCoreAir;
use openvm_rv32_adapters::Rv32VecHeapAdapterAir;

use crate::FieldExprVecHeapStep;

mod is_eq;
pub use is_eq::*;
mod addsub;
pub use addsub::*;
mod muldiv;
pub use muldiv::*;
use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use openvm_mod_circuit_builder::{
    FieldExpressionCoreAir, FieldExpressionFiller, FieldExpressionStep,
};
use openvm_rv32_adapters::{
    Rv32IsEqualModAdapterAir, Rv32IsEqualModAdapterFiller, Rv32IsEqualModAdapterStep,
    Rv32VecHeapAdapterAir, Rv32VecHeapAdapterFiller, Rv32VecHeapAdapterStep,
};

#[cfg(test)]
mod tests;

pub(crate) type ModularAir<const BLOCKS: usize, const BLOCK_SIZE: usize> = VmAirWrapper<
    Rv32VecHeapAdapterAir<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

pub(crate) type ModularStep<const BLOCKS: usize, const BLOCK_SIZE: usize> =
    FieldExprVecHeapStep<2, BLOCKS, BLOCK_SIZE>;

pub(crate) type ModularChip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> = VmChipWrapper<
    F,
    FieldExpressionFiller<Rv32VecHeapAdapterFiller<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>,
>;
// Must have TOTAL_LIMBS = NUM_LANES * LANE_SIZE
pub type ModularIsEqualAir<
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
> = VmAirWrapper<
    Rv32IsEqualModAdapterAir<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
    ModularIsEqualCoreAir<TOTAL_LIMBS, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

pub type VmModularIsEqualStep<
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
> = ModularIsEqualStep<
    Rv32IsEqualModAdapterStep<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
    TOTAL_LIMBS,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;

pub type ModularIsEqualChip<
    F,
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
> = VmChipWrapper<
    F,
    ModularIsEqualFiller<
        Rv32IsEqualModAdapterFiller<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        TOTAL_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;
