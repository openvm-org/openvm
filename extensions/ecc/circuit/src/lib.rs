#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]
use openvm_circuit::arch::DEFAULT_BLOCK_SIZE;
#[cfg(feature = "cuda")]
use {
    openvm_mod_circuit_builder::FieldExpressionCoreRecordMut,
    openvm_rv32_adapters::Rv32VecHeapAdapterRecord,
};

mod extension;
mod weierstrass_chip;

pub use extension::*;
// Re-export limb constants from algebra for consistency
pub use openvm_algebra_circuit::{NUM_LIMBS_32, NUM_LIMBS_48};
pub use weierstrass_chip::*;

// Blocks per ECC operation (3 projective coordinates per point: X, Y, Z)
/// Blocks for ECC with 32-limb coordinates: 3 * (32 / 4) = 24 blocks
pub const ECC_BLOCKS_32: usize = 3 * (NUM_LIMBS_32 / DEFAULT_BLOCK_SIZE);
/// Blocks for ECC with 48-limb coordinates: 3 * (48 / 4) = 36 blocks
pub const ECC_BLOCKS_48: usize = 3 * (NUM_LIMBS_48 / DEFAULT_BLOCK_SIZE);

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
