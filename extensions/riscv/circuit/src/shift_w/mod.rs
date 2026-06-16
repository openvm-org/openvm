use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::{
    adapters::{
        Rv64BaseAluWAdapterAir, Rv64BaseAluWAdapterExecutor, Rv64BaseAluWAdapterFiller,
        RV64_BYTE_BITS, RV64_WORD_NUM_LIMBS,
    },
    shift_arithmetic_right::{
        ShiftArithmeticRightCoreAir, ShiftArithmeticRightExecutor, ShiftArithmeticRightFiller,
    },
    shift_logical::{ShiftLogicalCoreAir, ShiftLogicalExecutor, ShiftLogicalFiller},
};

mod execution;

pub type ShiftWLogicalCoreAir = ShiftLogicalCoreAir<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type ShiftWArithmeticRightCoreAir =
    ShiftArithmeticRightCoreAir<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type ShiftWLogicalExecutor<A> = ShiftLogicalExecutor<A, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type ShiftWArithmeticRightExecutor<A> =
    ShiftArithmeticRightExecutor<A, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type ShiftWLogicalFiller<A> = ShiftLogicalFiller<A, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type ShiftWArithmeticRightFiller<A> =
    ShiftArithmeticRightFiller<A, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64ShiftWLogicalAir = VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftWLogicalCoreAir>;
pub type Rv64ShiftWArithmeticRightAir =
    VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftWArithmeticRightCoreAir>;
pub type Rv64ShiftWLogicalExecutor = ShiftWLogicalExecutor<Rv64BaseAluWAdapterExecutor>;
pub type Rv64ShiftWArithmeticRightExecutor =
    ShiftWArithmeticRightExecutor<Rv64BaseAluWAdapterExecutor>;
pub type Rv64ShiftWLogicalChip<F> =
    VmChipWrapper<F, ShiftWLogicalFiller<Rv64BaseAluWAdapterFiller>>;
pub type Rv64ShiftWArithmeticRightChip<F> =
    VmChipWrapper<F, ShiftWArithmeticRightFiller<Rv64BaseAluWAdapterFiller>>;
