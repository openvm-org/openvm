use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::{
    adapters::{
        Rv64BaseAluWRegU16AdapterAir, Rv64BaseAluWRegU16AdapterExecutor,
        Rv64BaseAluWRegU16AdapterFiller, RV64_WORD_U16_LIMBS, U16_BITS,
    },
    shift_logical::{ShiftLogicalCoreAir, ShiftLogicalFiller},
    shift_right_arithmetic::{
        ShiftRightArithmeticCoreAir, ShiftRightArithmeticExecutor, ShiftRightArithmeticFiller,
    },
};

mod execution;
mod preflight;
pub use preflight::*;

// SLLW/SRLW/SRAW all use the u16 shift cores over the W adapter (low 32-bit word in,
// sign-extended 64-bit write).
pub type ShiftWLogicalCoreAir = ShiftLogicalCoreAir<RV64_WORD_U16_LIMBS, U16_BITS>;
pub type ShiftWRightArithmeticCoreAir = ShiftRightArithmeticCoreAir<RV64_WORD_U16_LIMBS, U16_BITS>;
pub type ShiftWRightArithmeticExecutor<A> =
    ShiftRightArithmeticExecutor<A, RV64_WORD_U16_LIMBS, U16_BITS>;
pub type ShiftWLogicalFiller<A> = ShiftLogicalFiller<A, RV64_WORD_U16_LIMBS, U16_BITS>;
pub type ShiftWRightArithmeticFiller<A> =
    ShiftRightArithmeticFiller<A, RV64_WORD_U16_LIMBS, U16_BITS>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64ShiftWLogicalAir = VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftWLogicalCoreAir>;
pub type Rv64ShiftWRightArithmeticAir =
    VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, ShiftWRightArithmeticCoreAir>;
pub type Rv64ShiftWLogicalExecutor = ShiftWLogicalExecutor<Rv64BaseAluWRegU16AdapterExecutor>;
pub type Rv64ShiftWRightArithmeticExecutor =
    ShiftWRightArithmeticExecutor<Rv64BaseAluWRegU16AdapterExecutor>;
pub type Rv64ShiftWLogicalChip<F> =
    VmChipWrapper<F, ShiftWLogicalFiller<Rv64BaseAluWRegU16AdapterFiller>>;
pub type Rv64ShiftWRightArithmeticChip<F> =
    VmChipWrapper<F, ShiftWRightArithmeticFiller<Rv64BaseAluWRegU16AdapterFiller>>;
