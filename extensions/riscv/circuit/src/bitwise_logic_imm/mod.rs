use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::adapters::{BaseAluImmAdapterAir, BYTE_BITS, REGISTER_NUM_LIMBS};

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
pub type BitwiseLogicImmAir =
    VmAirWrapper<BaseAluImmAdapterAir, BitwiseLogicImmCoreAir<REGISTER_NUM_LIMBS, BYTE_BITS>>;
pub type BitwiseLogicImmExecutor = BitwiseLogicImmCoreExecutor<REGISTER_NUM_LIMBS, BYTE_BITS>;
pub type BitwiseLogicImmChip<F> =
    VmChipWrapper<F, BitwiseLogicImmFiller<REGISTER_NUM_LIMBS, BYTE_BITS>>;
