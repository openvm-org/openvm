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
    shift_logical::{ShiftLogicalU16CoreAir, ShiftLogicalU16Filler},
};

mod execution;
mod preflight;
pub use preflight::*;

// SLLW/SRLW reuse the u16 logical core (2 limbs of 16 bits) over the W adapter, which exposes the
// low 32-bit word to the core and rebuilds the sign-extended 64-bit write. SRAW keeps the byte-
// shaped arithmetic-right core, whose AIR relies on byte-level carry/shift structure.
pub type ShiftWLogicalCoreAir = ShiftLogicalU16CoreAir<RV64_WORD_U16_LIMBS, U16_BITS>;
pub type ShiftWArithmeticRightCoreAir =
    ShiftArithmeticRightCoreAir<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type ShiftWArithmeticRightExecutor<A> =
    ShiftArithmeticRightExecutor<A, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type ShiftWLogicalFiller<A> = ShiftLogicalU16Filler<A, RV64_WORD_U16_LIMBS, U16_BITS>;
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
