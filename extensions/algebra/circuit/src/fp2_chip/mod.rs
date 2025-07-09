use openvm_circuit::arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper};
use openvm_mod_circuit_builder::FieldExpressionCoreAir;
use openvm_rv32_adapters::Rv32VecHeapAdapterAir;

use crate::FieldExprVecHeapStep;

mod addsub;
pub use addsub::*;

mod muldiv;
pub use muldiv::*;

pub(crate) type Fp2Air<const BLOCKS: usize, const BLOCK_SIZE: usize> = VmAirWrapper<
    Rv32VecHeapAdapterAir<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

pub(crate) type Fp2Step<const BLOCKS: usize, const BLOCK_SIZE: usize> =
    FieldExprVecHeapStep<2, BLOCKS, BLOCK_SIZE>;

pub(crate) type Fp2Chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> = NewVmChipWrapper<
    F,
    Fp2Air<BLOCKS, BLOCK_SIZE>,
    Fp2Step<BLOCKS, BLOCK_SIZE>,
    MatrixRecordArena<F>,
>;
