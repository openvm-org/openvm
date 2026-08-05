use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionFiller};
use openvm_riscv_adapters::{IsEqualModU16AdapterAir, VecHeapAdapterAir, VecHeapAdapterFiller};
use openvm_riscv_circuit::adapters::U16_BITS;

use crate::FieldExprVecHeapExecutor;

mod is_eq;
pub use is_eq::*;
mod addsub;
pub use addsub::*;
mod muldiv;
pub use muldiv::*;

#[cfg(test)]
mod tests;

pub type ModularAir<const BLOCKS: usize> =
    VmAirWrapper<VecHeapAdapterAir<2, BLOCKS, BLOCKS>, FieldExpressionCoreAir>;

pub type ModularExecutor<const BLOCKS: usize> = FieldExprVecHeapExecutor<BLOCKS, false>;

pub type ModularChip<F, const BLOCKS: usize> =
    VmChipWrapper<F, FieldExpressionFiller<VecHeapAdapterFiller<2, BLOCKS, BLOCKS>>>;

/// U16-shaped is_eq wrapper: two heap operands, one BLOCK_FE_WIDTH-cell register write.
pub type ModularIsEqualU16Air<const NUM_LANES: usize, const TOTAL_LIMBS: usize> = VmAirWrapper<
    IsEqualModU16AdapterAir<2, NUM_LANES, TOTAL_LIMBS>,
    ModularIsEqualCoreAir<TOTAL_LIMBS, BLOCK_FE_WIDTH, U16_BITS>,
>;

#[derive(Clone)]
pub struct VmModularIsEqualU16Executor<const NUM_LANES: usize, const TOTAL_LIMBS: usize>(
    ModularIsEqualExecutor<TOTAL_LIMBS>,
);

pub type ModularIsEqualU16Chip<F, const TOTAL_LIMBS: usize> =
    VmChipWrapper<F, ModularIsEqualFiller<TOTAL_LIMBS, U16_BITS>>;
