use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::{
    adapters::{MultWAdapterAir, BYTE_BITS, WORD_NUM_LIMBS},
    mul::{MultiplicationCoreAir, MultiplicationCoreExecutor, MultiplicationFiller},
};

mod execution;

pub type MulWCoreAir = MultiplicationCoreAir<WORD_NUM_LIMBS, BYTE_BITS>;
pub type MulWCoreExecutor = MultiplicationCoreExecutor<WORD_NUM_LIMBS, BYTE_BITS>;
pub type MulWFiller = MultiplicationFiller<WORD_NUM_LIMBS, BYTE_BITS>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
pub(crate) mod trace;

pub type MulWAir = VmAirWrapper<MultWAdapterAir, MulWCoreAir>;
pub type MulWExecutor = MulWCoreExecutor;
pub type MulWChip<F> = VmChipWrapper<F, MulWFiller>;
