use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{
    Rv64BaseAluAdapterAir, Rv64BaseAluAdapterExecutor, Rv64BaseAluAdapterFiller, RV32_CELL_BITS,
    RV32_REGISTER_NUM_LIMBS,
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

pub type Rv64BaseAluAir =
    VmAirWrapper<Rv64BaseAluAdapterAir, BaseAluCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Rv64BaseAluExecutor = BaseAluExecutor<
    Rv64BaseAluAdapterExecutor<RV32_CELL_BITS>,
    RV32_REGISTER_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type Rv64BaseAluChip<F> = VmChipWrapper<
    F,
    BaseAluFiller<
        Rv64BaseAluAdapterFiller<RV32_CELL_BITS>,
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;
