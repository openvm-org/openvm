use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{
    Rv64BaseAluWAdapterAir, Rv64BaseAluWAdapterExecutor, Rv64BaseAluWAdapterFiller, RV64_CELL_BITS,
    RV64_WORD_NUM_LIMBS,
};
use super::shift::{ShiftCoreAir, ShiftExecutor, ShiftFiller};

mod execution;

pub type ShiftWCoreAir = ShiftCoreAir<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;
pub type ShiftWExecutor<A> = ShiftExecutor<A, RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;
pub type ShiftWFiller<A> = ShiftFiller<A, RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64ShiftWAir = VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftWCoreAir>;
pub type Rv64ShiftWExecutor = ShiftWExecutor<Rv64BaseAluWAdapterExecutor>;
pub type Rv64ShiftWChip<F> = VmChipWrapper<F, ShiftWFiller<Rv64BaseAluWAdapterFiller>>;
