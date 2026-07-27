use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{Rv64BaseAluImmAdapterAir, RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS};

mod core;
mod execution;
pub use core::*;

pub(crate) mod trace;

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
    BitwiseLogicImmExecutor<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>;
pub type Rv64BitwiseLogicImmChip<F> =
    VmChipWrapper<F, BitwiseLogicImmFiller<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>>;
