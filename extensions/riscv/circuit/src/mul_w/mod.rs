use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::{
    adapters::{Rv64MultWAdapterAir, RV64_BYTE_BITS, RV64_WORD_NUM_LIMBS},
    mul::{MultiplicationCoreAir, MultiplicationExecutor, MultiplicationFiller},
};

mod execution;

pub type MulWCoreAir = MultiplicationCoreAir<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type MulWExecutor = MultiplicationExecutor<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type MulWFiller = MultiplicationFiller<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
pub(crate) mod trace;

pub type Rv64MulWAir = VmAirWrapper<Rv64MultWAdapterAir, MulWCoreAir>;
pub type Rv64MulWExecutor = MulWExecutor;
pub type Rv64MulWChip<F> = VmChipWrapper<F, MulWFiller>;
