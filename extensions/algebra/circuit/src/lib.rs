#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]

use derive_more::derive::{Deref, DerefMut};
use openvm_circuit::arch::CONST_BLOCK_SIZE;
use openvm_circuit_derive::PreflightExecutor;
use openvm_mod_circuit_builder::FieldExpressionExecutor;
use openvm_rv32_adapters::Rv32VecHeapAdapterExecutor;
#[cfg(feature = "cuda")]
use {
    openvm_mod_circuit_builder::FieldExpressionCoreRecordMut,
    openvm_rv32_adapters::Rv32VecHeapAdapterRecord,
};

// Number of limbs for different modulus sizes (bytes)
/// Number of limbs for 256-bit (32-byte) moduli
pub const NUM_LIMBS_32: usize = 32;
/// Number of limbs for 384-bit (48-byte) moduli
pub const NUM_LIMBS_48: usize = 48;

// Blocks per operation for modular arithmetic (single field element)
/// Blocks for 32-limb modular operations: 32 / 4 = 8 blocks
pub const MODULAR_BLOCKS_32: usize = NUM_LIMBS_32 / CONST_BLOCK_SIZE;
/// Blocks for 48-limb modular operations: 48 / 4 = 12 blocks
pub const MODULAR_BLOCKS_48: usize = NUM_LIMBS_48 / CONST_BLOCK_SIZE;

// Blocks per operation for Fp2 (two field elements)
/// Blocks for Fp2 with 32-limb base field: 2 * 8 = 16 blocks
pub const FP2_BLOCKS_32: usize = 2 * MODULAR_BLOCKS_32;
/// Blocks for Fp2 with 48-limb base field: 2 * 12 = 24 blocks
pub const FP2_BLOCKS_48: usize = 2 * MODULAR_BLOCKS_48;

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
