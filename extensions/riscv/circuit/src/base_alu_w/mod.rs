use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{
    Rv64BaseAluWAdapterAir, Rv64BaseAluWAdapterExecutor, Rv64BaseAluWAdapterFiller, RV64_CELL_BITS,
    RV64_WORD_NUM_LIMBS,
};
use super::base_alu::{BaseAluCoreAir, BaseAluExecutor, BaseAluFiller};

mod execution;
pub type BaseAluWCoreAir = BaseAluCoreAir<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;
pub type BaseAluWExecutor<A> = BaseAluExecutor<A, RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;
pub type BaseAluWFiller<A> = BaseAluFiller<A, RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64BaseAluWAir = VmAirWrapper<Rv64BaseAluWAdapterAir, BaseAluWCoreAir>;
pub type Rv64BaseAluWExecutor = BaseAluWExecutor<Rv64BaseAluWAdapterExecutor>;
pub type Rv64BaseAluWChip<F> = VmChipWrapper<F, BaseAluWFiller<Rv64BaseAluWAdapterFiller>>;
