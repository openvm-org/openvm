use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionFiller};
use openvm_riscv_adapters::{Rv64VecHeapAdapterAir, Rv64VecHeapAdapterFiller};

use crate::FieldExprVecHeapExecutor;

mod addsub;
pub use addsub::*;
mod muldiv;
pub use muldiv::*;

pub type Fp2Air<const BLOCKS: usize, const BLOCK_SIZE: usize> = VmAirWrapper<
    Rv64VecHeapAdapterAir<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

pub type Fp2Executor<const BLOCKS: usize, const BLOCK_SIZE: usize> =
    FieldExprVecHeapExecutor<BLOCKS, BLOCK_SIZE, true>;

pub type Fp2Chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> = VmChipWrapper<
    F,
    FieldExpressionFiller<Rv64VecHeapAdapterFiller<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>,
>;

#[cfg(test)]
mod tests;
