#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]

use std::ops::{Deref, DerefMut};

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
mod preflight;

use fields::{get_field_type, get_fp2_field_type, FieldType};

// Note: PreflightExecutor is implemented manually in preflight.rs with fast native arithmetic
#[derive(Clone)]
pub struct FieldExprVecHeapExecutor<
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
> {
    inner: FieldExpressionExecutor<
        Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    >,
    pub(crate) cached_field_type: Option<FieldType>,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize, const IS_FP2: bool>
    FieldExprVecHeapExecutor<BLOCKS, BLOCK_SIZE, IS_FP2>
{
    pub fn new(
        inner: FieldExpressionExecutor<
            Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >,
    ) -> Self {
        let cached_field_type = if IS_FP2 {
            get_fp2_field_type(&inner.expr.prime)
        } else {
            get_field_type(&inner.expr.prime)
        };
        Self {
            inner,
            cached_field_type,
        }
    }
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize, const IS_FP2: bool> Deref
    for FieldExprVecHeapExecutor<BLOCKS, BLOCK_SIZE, IS_FP2>
{
    type Target = FieldExpressionExecutor<
        Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    >;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize, const IS_FP2: bool> DerefMut
    for FieldExprVecHeapExecutor<BLOCKS, BLOCK_SIZE, IS_FP2>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

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
