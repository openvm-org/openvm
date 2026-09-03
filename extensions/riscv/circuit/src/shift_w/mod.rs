use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::{
    adapters::{BaseAluWRegU16AdapterAir, U16_BITS, WORD_U16_LIMBS},
    shift_logical::{ShiftLogicalCoreAir, ShiftLogicalFiller},
    shift_right_arithmetic::{
        ShiftRightArithmeticCoreAir, ShiftRightArithmeticCoreExecutor, ShiftRightArithmeticFiller,
    },
};

mod execution;
pub(crate) mod trace;

// SLLW/SRLW/SRAW all use the u16 shift cores over the W adapter (low 32-bit word in,
// sign-extended 64-bit write).
pub type ShiftWLogicalCoreAir = ShiftLogicalCoreAir<WORD_U16_LIMBS, U16_BITS>;
pub type ShiftWRightArithmeticCoreAir = ShiftRightArithmeticCoreAir<WORD_U16_LIMBS, U16_BITS>;
pub type ShiftWRightArithmeticCoreExecutor =
    ShiftRightArithmeticCoreExecutor<WORD_U16_LIMBS, U16_BITS>;
#[derive(Clone, Copy, derive_new::new)]
pub struct ShiftWLogicalCoreExecutor {
    pub offset: usize,
}
pub type ShiftWLogicalFiller = ShiftLogicalFiller;
pub type ShiftWRightArithmeticFiller = ShiftRightArithmeticFiller;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type ShiftWLogicalAir = VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftWLogicalCoreAir>;
pub type ShiftWRightArithmeticAir =
    VmAirWrapper<BaseAluWRegU16AdapterAir, ShiftWRightArithmeticCoreAir>;
pub type ShiftWLogicalExecutor = ShiftWLogicalCoreExecutor;
pub type ShiftWRightArithmeticExecutor = ShiftWRightArithmeticCoreExecutor;
pub type ShiftWLogicalChip<F> = VmChipWrapper<F, ShiftWLogicalFiller>;
pub type ShiftWRightArithmeticChip<F> = VmChipWrapper<F, ShiftWRightArithmeticFiller>;
