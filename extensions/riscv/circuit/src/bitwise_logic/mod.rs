use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{
    Rv64BaseAluAdapterAir, Rv64BaseAluAdapterExecutor, Rv64BaseAluAdapterFiller, RV64_BYTE_BITS,
    RV64_REGISTER_NUM_LIMBS,
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
    Rv64BaseAluAdapterAir,
    BitwiseLogicCoreAir<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
>;
pub type Rv64BitwiseLogicExecutor = BitwiseLogicExecutor<
    Rv64BaseAluAdapterExecutor<RV64_BYTE_BITS>,
    RV64_REGISTER_NUM_LIMBS,
    RV64_BYTE_BITS,
>;
pub type Rv64BitwiseLogicChip<F> = VmChipWrapper<
    F,
    BitwiseLogicFiller<
        Rv64BaseAluAdapterFiller<RV64_BYTE_BITS>,
        RV64_REGISTER_NUM_LIMBS,
        RV64_BYTE_BITS,
    >,
>;
