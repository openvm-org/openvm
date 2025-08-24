#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]

use derive_more::derive::{Deref, DerefMut};
use openvm_circuit_derive::PreflightExecutor;
use openvm_mod_circuit_builder::FieldExpressionExecutor;
use openvm_rv32_adapters::Rv32VecHeapAdapterExecutor;
#[cfg(feature = "cuda")]
use {
    openvm_mod_circuit_builder::FieldExpressionCoreRecordMut,
    openvm_rv32_adapters::Rv32VecHeapAdapterRecord,
};

pub mod fp2_chip;
pub mod modular_chip;

mod execution;
mod fp2;
pub use fp2::*;
mod extension;
pub use extension::*;
pub mod fields;

#[derive(Clone, PreflightExecutor, Deref, DerefMut)]
pub struct FieldExprVecHeapExecutor<
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
>(FieldExpressionExecutor<Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>);

#[cfg(feature = "cuda")]
pub(crate) type AlgebraRecord<
    'a,
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = (
    &'a mut Rv32VecHeapAdapterRecord<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreRecordMut<'a>,
);
