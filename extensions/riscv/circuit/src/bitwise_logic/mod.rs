use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{
    Rv64BaseAluRegAdapterAir, Rv64BaseAluRegAdapterExecutor, Rv64BaseAluRegAdapterFiller,
    RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS,
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

pub type Rv64BitwiseLogicAir = VmAirWrapper<
    Rv64BaseAluRegAdapterAir,
    BitwiseLogicCoreAir<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
>;
pub type Rv64BitwiseLogicExecutor =
    BitwiseLogicExecutor<Rv64BaseAluRegAdapterExecutor, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>;
pub type Rv64BitwiseLogicChip<F> = VmChipWrapper<
    F,
    BitwiseLogicFiller<Rv64BaseAluRegAdapterFiller, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
>;
