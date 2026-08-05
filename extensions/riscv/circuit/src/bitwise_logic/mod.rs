use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{BaseAluRegAdapterAir, BYTE_BITS, REGISTER_NUM_LIMBS};

mod core;
mod execution;
pub(crate) mod trace;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type BitwiseLogicAir =
    VmAirWrapper<BaseAluRegAdapterAir, BitwiseLogicCoreAir<REGISTER_NUM_LIMBS, BYTE_BITS>>;
pub type BitwiseLogicExecutor = BitwiseLogicCoreExecutor<REGISTER_NUM_LIMBS, BYTE_BITS>;
pub type BitwiseLogicChip<F> = VmChipWrapper<F, BitwiseLogicFiller<BYTE_BITS>>;
