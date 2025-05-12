mod addsub;
pub use addsub::*;

mod muldiv;
pub use muldiv::*;
use openvm_circuit::arch::{NewVmChipWrapper, VmAirWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionStep};
use openvm_rv32_adapters::{Rv32VecHeapAdapterStep, Rv32VecHeapAdapterAir};

pub(crate) type Fp2Air<const BLOCKS: usize, const BLOCK_SIZE: usize> = VmAirWrapper<
    Rv32VecHeapAdapterAir<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

pub(crate) type Fp2Step<const BLOCKS: usize, const BLOCK_SIZE: usize> =
    FieldExpressionStep<Rv32VecHeapAdapterStep<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>;

pub(crate) type Fp2Chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> =
    NewVmChipWrapper<F, Fp2Air<BLOCKS, BLOCK_SIZE>, Fp2Step<BLOCKS, BLOCK_SIZE>>;
