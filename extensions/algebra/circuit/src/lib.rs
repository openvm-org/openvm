#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]

use derive_more::derive::{Deref, DerefMut};
use openvm_circuit_derive::PreflightExecutor;
use openvm_mod_circuit_builder::FieldExpressionExecutor;
use openvm_rv32_adapters::Rv32VecHeapAdapterExecutor;

pub mod fp2_chip;
pub mod modular_chip;

mod execution;
mod fp2;
pub use fp2::*;
mod modular_extension;
pub use modular_extension::*;
mod fp2_extension;
pub use fp2_extension::*;
mod config;
pub use config::*;
pub mod fields;

pub struct AlgebraCpuProverExt;

#[derive(Clone, PreflightExecutor, Deref, DerefMut)]
pub struct FieldExprVecHeapExecutor<
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const IS_FP2: bool,
>(FieldExpressionExecutor<Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>);
