use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{
    Rv64BaseAluAdapterAir, Rv64BaseAluAdapterExecutor, Rv64BaseAluAdapterFiller, RV64_CELL_BITS,
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

pub type Rv64LessThanAir =
    VmAirWrapper<Rv64BaseAluAdapterAir, LessThanCoreAir<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>>;
pub type Rv64LessThanExecutor = LessThanExecutor<
    Rv64BaseAluAdapterExecutor<RV64_CELL_BITS>,
    RV64_REGISTER_NUM_LIMBS,
    RV64_CELL_BITS,
>;
pub type Rv64LessThanChip<F> = VmChipWrapper<
    F,
    LessThanFiller<
        Rv64BaseAluAdapterFiller<RV64_CELL_BITS>,
        RV64_REGISTER_NUM_LIMBS,
        RV64_CELL_BITS,
    >,
>;
