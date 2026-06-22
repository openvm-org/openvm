use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};
use openvm_circuit_derive::PreflightExecutor;
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionFiller};
use openvm_riscv_adapters::{
    Rv64IsEqualModU16AdapterAir, Rv64IsEqualModU16AdapterExecutor, Rv64IsEqualModU16AdapterFiller,
    Rv64VecHeapAdapterAir, Rv64VecHeapAdapterFiller, Rv64VecHeapU16AdapterAir,
    Rv64VecHeapU16AdapterExecutor, Rv64VecHeapU16AdapterFiller,
};
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
    VmAirWrapper<Rv64VecHeapAdapterAir<2, BLOCKS, BLOCKS>, FieldExpressionCoreAir>;

pub type ModularExecutor<const BLOCKS: usize> = FieldExprVecHeapExecutor<BLOCKS, false>;

pub type ModularU16Air<const BLOCKS: usize> =
    VmAirWrapper<Rv64VecHeapU16AdapterAir<2, BLOCKS, BLOCKS>, FieldExpressionCoreAir>;

pub type ModularU16Executor<const BLOCKS: usize> =
    FieldExprVecHeapExecutor<BLOCKS, false, Rv64VecHeapU16AdapterExecutor<2, BLOCKS, BLOCKS>>;

pub type ModularChip<F, const BLOCKS: usize> =
    VmChipWrapper<F, FieldExpressionFiller<Rv64VecHeapAdapterFiller<2, BLOCKS, BLOCKS>>>;

pub type ModularU16Chip<F, const BLOCKS: usize> =
    VmChipWrapper<F, FieldExpressionFiller<Rv64VecHeapU16AdapterFiller<2, BLOCKS, BLOCKS>, u16>>;

/// U16-shaped is_eq wrapper: two heap operands, one BLOCK_FE_WIDTH-cell register write.
pub type ModularIsEqualU16Air<const NUM_LANES: usize, const TOTAL_LIMBS: usize> = VmAirWrapper<
    Rv64IsEqualModU16AdapterAir<2, NUM_LANES, TOTAL_LIMBS>,
    ModularIsEqualCoreAir<TOTAL_LIMBS, BLOCK_FE_WIDTH, U16_BITS>,
>;

#[derive(Clone, PreflightExecutor)]
pub struct VmModularIsEqualU16Executor<const NUM_LANES: usize, const TOTAL_LIMBS: usize>(
    ModularIsEqualExecutor<
        Rv64IsEqualModU16AdapterExecutor<2, NUM_LANES, TOTAL_LIMBS>,
        TOTAL_LIMBS,
        BLOCK_FE_WIDTH,
        U16_BITS,
    >,
);

pub type ModularIsEqualU16Chip<F, const NUM_LANES: usize, const TOTAL_LIMBS: usize> = VmChipWrapper<
    F,
    ModularIsEqualFiller<
        Rv64IsEqualModU16AdapterFiller<2, NUM_LANES>,
        TOTAL_LIMBS,
        BLOCK_FE_WIDTH,
        U16_BITS,
    >,
>;
