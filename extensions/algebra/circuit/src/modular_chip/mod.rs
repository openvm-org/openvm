use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_circuit_derive::PreflightExecutor;
use openvm_instructions::riscv::{RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionFiller};
use openvm_riscv_adapters::{
    Rv64IsEqualModAdapterAir, Rv64IsEqualModAdapterExecutor, Rv64IsEqualModAdapterFiller,
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

pub type ModularAir<const BLOCKS: usize> =
    VmAirWrapper<Rv64VecHeapAdapterAir<2, BLOCKS, BLOCKS>, FieldExpressionCoreAir>;

pub type ModularExecutor<const BLOCKS: usize> = FieldExprVecHeapExecutor<BLOCKS, false>;

pub type ModularChip<F, const BLOCKS: usize> =
    VmChipWrapper<F, FieldExpressionFiller<Rv64VecHeapAdapterFiller<2, BLOCKS, BLOCKS>>>;

// Must have TOTAL_LIMBS = NUM_LANES * MEMORY_BLOCK_BYTES.
pub type ModularIsEqualAir<const NUM_LANES: usize, const TOTAL_LIMBS: usize> = VmAirWrapper<
    Rv64IsEqualModAdapterAir<2, NUM_LANES, TOTAL_LIMBS>,
    ModularIsEqualCoreAir<TOTAL_LIMBS, RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>,
>;

#[derive(Clone, PreflightExecutor)]
pub struct VmModularIsEqualExecutor<const NUM_LANES: usize, const TOTAL_LIMBS: usize>(
    ModularIsEqualExecutor<
        Rv64IsEqualModAdapterExecutor<2, NUM_LANES, TOTAL_LIMBS>,
        TOTAL_LIMBS,
        RV64_REGISTER_NUM_LIMBS,
        RV64_CELL_BITS,
    >,
);

pub type ModularIsEqualChip<F, const NUM_LANES: usize, const TOTAL_LIMBS: usize> = VmChipWrapper<
    F,
    ModularIsEqualFiller<
        Rv64IsEqualModAdapterFiller<2, NUM_LANES>,
        TOTAL_LIMBS,
        RV64_REGISTER_NUM_LIMBS,
        RV64_CELL_BITS,
    >,
>;
