use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use super::adapters::{
    Rv64BaseAluImmU16AdapterAir, Rv64BaseAluImmU16AdapterExecutor, Rv64BaseAluImmU16AdapterFiller,
    Rv64BaseAluWImmU16AdapterAir, Rv64BaseAluWImmU16AdapterExecutor,
    Rv64BaseAluWImmU16AdapterFiller, RV64_WORD_U16_LIMBS, U16_BITS,
};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type AddIWCoreAir = AddICoreAir<RV64_WORD_U16_LIMBS, U16_BITS, false>;
pub type AddIWFiller<A> = AddIFiller<A, RV64_WORD_U16_LIMBS, U16_BITS, false>;

pub type Rv64AddIAir =
    VmAirWrapper<Rv64BaseAluImmU16AdapterAir, AddICoreAir<BLOCK_FE_WIDTH, U16_BITS, true>>;
pub type Rv64AddIExecutor =
    AddIExecutor<Rv64BaseAluImmU16AdapterExecutor, BLOCK_FE_WIDTH, U16_BITS>;
pub type Rv64AddIChip<F> =
    VmChipWrapper<F, AddIFiller<Rv64BaseAluImmU16AdapterFiller, BLOCK_FE_WIDTH, U16_BITS, true>>;

pub type Rv64AddIWAir = VmAirWrapper<Rv64BaseAluWImmU16AdapterAir, AddIWCoreAir>;
pub type Rv64AddIWExecutor =
    AddIExecutor<Rv64BaseAluWImmU16AdapterExecutor, RV64_WORD_U16_LIMBS, U16_BITS>;
pub type Rv64AddIWChip<F> = VmChipWrapper<F, AddIWFiller<Rv64BaseAluWImmU16AdapterFiller>>;
