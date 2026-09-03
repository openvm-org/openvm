use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use super::adapters::{
    BaseAluImmU16AdapterAir, BaseAluWImmU16AdapterAir, U16_BITS, WORD_U16_LIMBS,
};

mod core;
mod execution;
pub use core::*;

pub(crate) mod trace;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type AddIWCoreAir = AddICoreAir<WORD_U16_LIMBS, U16_BITS, false>;
pub type AddIWFiller = AddIFiller;

pub type AddIAir =
    VmAirWrapper<BaseAluImmU16AdapterAir, AddICoreAir<BLOCK_FE_WIDTH, U16_BITS, true>>;
pub type AddIExecutor = AddICoreExecutor<BLOCK_FE_WIDTH, U16_BITS>;
pub type AddIChip<F> = VmChipWrapper<F, AddIFiller>;

pub type AddIWAir = VmAirWrapper<BaseAluWImmU16AdapterAir, AddIWCoreAir>;
pub type AddIWExecutor = AddICoreExecutor<WORD_U16_LIMBS, U16_BITS>;
pub type AddIWChip<F> = VmChipWrapper<F, AddIWFiller>;
