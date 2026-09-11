use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::{
    adapters::{BaseAluWRegU16AdapterAir, U16_BITS, WORD_U16_LIMBS},
    add_sub::{AddSubCoreAir, AddSubFiller},
};

mod execution;
pub(crate) mod trace;

pub type AddSubWCoreAir = AddSubCoreAir<WORD_U16_LIMBS, U16_BITS, false>;
pub type AddSubWFiller = AddSubFiller;

#[derive(Clone, Copy, derive_new::new)]
pub struct AddSubWCoreExecutor {
    pub offset: usize,
}

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type AddSubWAir = VmAirWrapper<BaseAluWRegU16AdapterAir, AddSubWCoreAir>;
pub type AddSubWExecutor = AddSubWCoreExecutor;
pub type AddSubWChip<F> = VmChipWrapper<F, AddSubWFiller>;
