use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{
    Rv64MultWAdapterAir, Rv64MultWAdapterExecutor, Rv64MultWAdapterFiller, RV64_CELL_BITS,
    RV64_WORD_NUM_LIMBS,
};
use super::mul::{MultiplicationCoreAir, MultiplicationExecutor, MultiplicationFiller};

mod execution;

pub type MulWCoreAir = MultiplicationCoreAir<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;
pub type MulWExecutor<A> = MultiplicationExecutor<A, RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;
pub type MulWFiller<A> = MultiplicationFiller<A, RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64MulWAir = VmAirWrapper<Rv64MultWAdapterAir, MulWCoreAir>;
pub type Rv64MulWExecutor = MulWExecutor<Rv64MultWAdapterExecutor>;
pub type Rv64MulWChip<F> = VmChipWrapper<F, MulWFiller<Rv64MultWAdapterFiller>>;
