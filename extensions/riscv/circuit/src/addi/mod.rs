use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use super::adapters::{
    Rv64BaseAluImmU16AdapterAir, Rv64BaseAluWImmU16AdapterAir, RV64_WORD_U16_LIMBS, U16_BITS,
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

pub type AddIWCoreAir = AddICoreAir<RV64_WORD_U16_LIMBS, U16_BITS, false>;
pub type AddIWFiller = AddIFiller;

pub type Rv64AddIAir =
    VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<BLOCK_FE_WIDTH, U16_BITS, true>>;
pub type Rv64AddIExecutor = AddIExecutor<BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64AddIChip<F> = VmChipWrapper<F, AddIFiller>;

pub type Rv64AddIWAir = VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddIWCoreAir>;
pub type Rv64AddIWExecutor = AddIExecutor<RV64_WORD_U16_LIMBS, U16_BITS>;
pub type Rv64AddIWChip<F> = VmChipWrapper<F, AddIWFiller>;
