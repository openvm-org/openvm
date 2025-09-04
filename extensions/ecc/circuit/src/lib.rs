#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]
#[cfg(feature = "cuda")]
use {
    openvm_mod_circuit_builder::FieldExpressionCoreRecordMut,
    openvm_rv32_adapters::Rv32VecHeapAdapterRecord,
};

mod extension;
mod weierstrass_chip;

pub use extension::*;
pub use weierstrass_chip::*;

#[cfg(feature = "cuda")]
pub(crate) type EccRecord<
    'a,
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = (
    &'a mut Rv32VecHeapAdapterRecord<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreRecordMut<'a>,
);
