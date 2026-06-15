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

pub type Rv64ShiftLeftAir =
    VmAirWrapper<Rv64BaseAluAdapterAir, ShiftLeftCoreAir<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>>;
pub type Rv64ShiftLeftExecutor = ShiftLeftExecutor<
    Rv64BaseAluAdapterExecutor<RV64_BYTE_BITS>,
    RV64_REGISTER_NUM_LIMBS,
    RV64_BYTE_BITS,
>;
pub type Rv64ShiftLeftChip<F> = VmChipWrapper<
    F,
    ShiftLeftFiller<
        Rv64BaseAluAdapterFiller<RV64_BYTE_BITS>,
        RV64_REGISTER_NUM_LIMBS,
        RV64_BYTE_BITS,
    >,
>;
