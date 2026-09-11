use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionFiller};
use openvm_riscv_adapters::{VecHeapAdapterAir, VecHeapAdapterFiller};

use crate::FieldExprVecHeapExecutor;

mod addsub;
pub use addsub::*;
mod muldiv;
pub use muldiv::*;

pub type Fp2Air<const BLOCKS: usize> =
    VmAirWrapper<VecHeapAdapterAir<2, BLOCKS, BLOCKS>, FieldExpressionCoreAir>;

pub type Fp2Executor<const BLOCKS: usize> = FieldExprVecHeapExecutor<BLOCKS, true>;

pub type Fp2Chip<F, const BLOCKS: usize> =
    VmChipWrapper<F, FieldExpressionFiller<VecHeapAdapterFiller<2, BLOCKS, BLOCKS>>>;

#[cfg(test)]
mod tests;
