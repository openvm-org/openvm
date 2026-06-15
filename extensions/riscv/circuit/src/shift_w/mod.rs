use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::{
    adapters::{
        Rv64BaseAluWAdapterAir, Rv64BaseAluWAdapterExecutor, Rv64BaseAluWAdapterFiller,
        RV64_BYTE_BITS, RV64_WORD_NUM_LIMBS,
    },
    shift_left::{ShiftLeftCoreAir, ShiftLeftExecutor, ShiftLeftFiller},
    shift_right::{ShiftRightCoreAir, ShiftRightExecutor, ShiftRightFiller},
};

mod execution;

pub type ShiftWLeftCoreAir = ShiftLeftCoreAir<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type ShiftWRightCoreAir = ShiftRightCoreAir<RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type ShiftWLeftExecutor<A> = ShiftLeftExecutor<A, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type ShiftWRightExecutor<A> = ShiftRightExecutor<A, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type ShiftWLeftFiller<A> = ShiftLeftFiller<A, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;
pub type ShiftWRightFiller<A> = ShiftRightFiller<A, RV64_WORD_NUM_LIMBS, RV64_BYTE_BITS>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64ShiftWLeftAir = VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftWLeftCoreAir>;
pub type Rv64ShiftWRightAir = VmAirWrapper<Rv64BaseAluWAdapterAir, ShiftWRightCoreAir>;
pub type Rv64ShiftWLeftExecutor = ShiftWLeftExecutor<Rv64BaseAluWAdapterExecutor>;
pub type Rv64ShiftWRightExecutor = ShiftWRightExecutor<Rv64BaseAluWAdapterExecutor>;
pub type Rv64ShiftWLeftChip<F> = VmChipWrapper<F, ShiftWLeftFiller<Rv64BaseAluWAdapterFiller>>;
pub type Rv64ShiftWRightChip<F> = VmChipWrapper<F, ShiftWRightFiller<Rv64BaseAluWAdapterFiller>>;
