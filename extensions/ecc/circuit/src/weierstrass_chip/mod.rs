mod add_ne;
mod double;

pub use add_ne::*;
pub use double::*;

#[cfg(test)]
mod tests;


use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};

use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionStep};
use openvm_rv32_adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterStep};

pub(crate) type WeierstrassAir<const BLOCKS: usize, const BLOCK_SIZE: usize> = VmAirWrapper<
    Rv32VecHeapAdapterAir<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

pub(crate) type WeierstrassStep<const BLOCKS: usize, const BLOCK_SIZE: usize> =
    FieldExpressionStep<Rv32VecHeapAdapterStep<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>;

pub(crate) type WeierstrassChip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> =
    NewVmChipWrapper<F, WeierstrassAir<BLOCKS, BLOCK_SIZE>, WeierstrassStep<BLOCKS, BLOCK_SIZE>>;
