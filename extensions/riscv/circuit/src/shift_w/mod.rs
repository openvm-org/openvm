use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::{
    adapters::{
        Rv64BaseAluWAdapterAir, Rv64BaseAluWAdapterExecutor, Rv64BaseAluWAdapterFiller,
        Rv64BaseAluWU16AdapterAir, Rv64BaseAluWU16AdapterExecutor, Rv64BaseAluWU16AdapterFiller,
        RV64_BYTE_BITS, RV64_WORD_NUM_LIMBS, RV64_WORD_U16_LIMBS, U16_BITS,
    },
    shift_arithmetic_right::{
        ShiftArithmeticRightCoreAir, ShiftArithmeticRightExecutor, ShiftArithmeticRightFiller,
    },
    shift_logical::{ShiftLogicalCoreAir, ShiftLogicalFiller},
};

mod execution;
mod preflight;
pub use preflight::*;

// SLLW/SRLW use the u16 logical core over the W adapter (low 32-bit word in, sign-extended 64-bit
// write); SRAW uses the byte-shaped arithmetic-right core.
pub type ShiftWLogicalCoreAir = ShiftLogicalCoreAir<RV64_WORD_U16_LIMBS, U16_BITS>;
pub type ShiftWArithmeticRightCoreAir =
    ShiftArithmeticRightCoreAir<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type ShiftWArithmeticRightExecutor<A> =
    ShiftArithmeticRightExecutor<A, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type ShiftWLogicalFiller<A> = ShiftLogicalFiller<A, RV64_WORD_U16_LIMBS, U16_BITS>;
pub type ShiftWArithmeticRightFiller<A> =
    ShiftArithmeticRightFiller<A, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64ShiftWLogicalAir = VmAirWrapper<Rv64BaseAluWU16AdapterAir, ShiftWLogicalCoreAir>;
pub type Rv64ShiftWArithmeticRightAir =
    VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftWArithmeticRightCoreAir>;
pub type Rv64ShiftWLogicalExecutor = ShiftWLogicalExecutor<Rv64BaseAluWU16AdapterExecutor>;
pub type Rv64ShiftWArithmeticRightExecutor =
    ShiftWArithmeticRightExecutor<Rv64BaseAluWAdapterExecutor>;
pub type Rv64ShiftWLogicalChip<F> =
    VmChipWrapper<F, ShiftWLogicalFiller<Rv64BaseAluWU16AdapterFiller>>;
pub type Rv64ShiftWArithmeticRightChip<F> =
    VmChipWrapper<F, ShiftWArithmeticRightFiller<Rv64BaseAluWAdapterFiller>>;
