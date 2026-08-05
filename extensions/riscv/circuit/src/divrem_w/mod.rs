use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::{
    adapters::{MultWAdapterAir, MultWAdapterFiller, BYTE_BITS, WORD_NUM_LIMBS},
    divrem::{DivRemCoreAir, DivRemCoreExecutor, DivRemFiller},
};

mod execution;

pub type DivRemWCoreAir = DivRemCoreAir<WORD_NUM_LIMBS, BYTE_BITS>;
pub type DivRemWCoreExecutor = DivRemCoreExecutor<WORD_NUM_LIMBS, BYTE_BITS>;
pub type DivRemWFiller<A> = DivRemFiller<A, WORD_NUM_LIMBS, BYTE_BITS>;

pub(crate) mod trace;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type DivRemWAir = VmAirWrapper<MultWAdapterAir, DivRemWCoreAir>;
pub type DivRemWExecutor = DivRemWCoreExecutor;
pub type DivRemWChip<F> = VmChipWrapper<F, DivRemWFiller<MultWAdapterFiller>>;
