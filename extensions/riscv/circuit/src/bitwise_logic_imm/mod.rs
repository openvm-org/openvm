use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{
    Rv64BaseAluImmAdapterAir, Rv64BaseAluImmAdapterExecutor, Rv64BaseAluImmAdapterFiller,
    RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS,
};

mod core;
mod execution;
pub use core::*;

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

// Immediate-only bitwise chip with one register read.
pub type Rv64BitwiseLogicImmAir = VmAirWrapper<
    Rv64BaseAluImmAdapterAir,
    BitwiseLogicImmCoreAir<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
>;
pub type Rv64BitwiseLogicImmExecutor =
    BitwiseLogicImmExecutor<Rv64BaseAluImmAdapterExecutor, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>;
pub type Rv64BitwiseLogicImmChip<F> = VmChipWrapper<
    F,
    BitwiseLogicImmFiller<Rv64BaseAluImmAdapterFiller, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
>;
