use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};
use openvm_circuit_derive::PreflightExecutor;
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionFiller};
use openvm_riscv_adapters::{
    Rv64IsEqualModAdapterU16Air, Rv64IsEqualModAdapterU16Executor, Rv64IsEqualModAdapterU16Filler,
    Rv64VecHeapAdapterAir, Rv64VecHeapAdapterFiller,
};

use crate::FieldExprVecHeapExecutor;

mod is_eq;
pub use is_eq::*;
mod addsub;
pub use addsub::*;
mod muldiv;
pub use muldiv::*;

#[cfg(test)]
mod tests;

pub type ModularAir<const BLOCKS: usize, const BLOCK_SIZE: usize> = VmAirWrapper<
    Rv64VecHeapAdapterAir<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

pub type ModularExecutor<const BLOCKS: usize, const BLOCK_SIZE: usize> =
    FieldExprVecHeapExecutor<BLOCKS, BLOCK_SIZE, false>;

pub type ModularChip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> = VmChipWrapper<
    F,
    FieldExpressionFiller<Rv64VecHeapAdapterFiller<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>,
>;

// U16-shaped is_eq AIR/chip/executor/filler. The chip reads two u16-celled heap operands and
// writes a single BLOCK_FE_WIDTH-wide u16 word into rd. LIMB_BITS = 16, WRITE_LIMBS =
// BLOCK_FE_WIDTH (= 4).
//
// `NUM_LANES * LANE_SIZE` must equal `TOTAL_LIMBS` (u16 cell count), and `LANE_SIZE` must equal
// `BLOCK_FE_WIDTH` per the adapter's assertion.
pub type ModularIsEqualU16Air<
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
> = VmAirWrapper<
    Rv64IsEqualModAdapterU16Air<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
    ModularIsEqualCoreAir<TOTAL_LIMBS, BLOCK_FE_WIDTH, 16>,
>;

#[derive(Clone, PreflightExecutor)]
pub struct VmModularIsEqualU16Executor<
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
>(
    pub  ModularIsEqualExecutor<
        Rv64IsEqualModAdapterU16Executor<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        TOTAL_LIMBS,
        BLOCK_FE_WIDTH,
        16,
    >,
);

pub type ModularIsEqualU16Chip<
    F,
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
> = VmChipWrapper<
    F,
    ModularIsEqualFiller<
        Rv64IsEqualModAdapterU16Filler<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        TOTAL_LIMBS,
        BLOCK_FE_WIDTH,
        16,
    >,
>;
