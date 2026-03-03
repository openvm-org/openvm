use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::adapters::{Rv32MultAdapterAir, Rv32MultAdapterExecutor, Rv32MultAdapterFiller};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv32MultiplicationAir = VmAirWrapper<
    Rv32MultAdapterAir,
    MultiplicationCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Rv32MultiplicationExecutor =
    MultiplicationExecutor<Rv32MultAdapterExecutor, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>;
pub type Rv32MultiplicationChip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<Rv32MultAdapterFiller, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
