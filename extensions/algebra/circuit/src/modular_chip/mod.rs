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

#[cfg(test)]
mod tests;

pub(crate) type ModularAir<const BLOCKS: usize, const BLOCK_SIZE: usize> = VmAirWrapper<
    Rv32VecHeapAdapterAir<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

pub(crate) type ModularStep<const BLOCKS: usize, const BLOCK_SIZE: usize> =
    FieldExprVecHeapStep<2, BLOCKS, BLOCK_SIZE>;

pub(crate) type ModularChip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> = NewVmChipWrapper<
    F,
    ModularAir<BLOCKS, BLOCK_SIZE>,
    ModularStep<BLOCKS, BLOCK_SIZE>,
    MatrixRecordArena<F>,
>;

#[cfg(test)]
pub(crate) type ModularDenseChip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> =
    NewVmChipWrapper<
        F,
        ModularAir<BLOCKS, BLOCK_SIZE>,
        ModularStep<BLOCKS, BLOCK_SIZE>,
        openvm_circuit::arch::DenseRecordArena,
    >;
