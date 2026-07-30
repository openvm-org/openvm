use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{
    Rv64BaseAluImmAdapterAir, Rv64BaseAluImmAdapterExecutor, Rv64BaseAluImmAdapterFiller,
    Rv64BaseAluImmU16AdapterAir, Rv64BaseAluImmU16AdapterExecutor, Rv64BaseAluImmU16AdapterFiller,
    Rv64BaseAluRegAdapterAir, Rv64BaseAluRegAdapterExecutor, Rv64BaseAluRegAdapterFiller,
    Rv64BaseAluRegU16AdapterAir, Rv64BaseAluRegU16AdapterExecutor, Rv64BaseAluRegU16AdapterFiller,
    U16_BITS,
};

mod bitwise_inv;
mod byte_unary;
mod core;
mod execution;
mod min_max;
pub use core::*;

pub use bitwise_inv::*;
pub use byte_unary::*;
pub use min_max::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64BitManipShAddAir = VmAirWrapper<Rv64BaseAluRegU16AdapterAir, BitManipShAddCoreAir>;
pub type Rv64BitManipShAddExecutor = BitManipShAddExecutor<Rv64BaseAluRegU16AdapterExecutor>;
pub type Rv64BitManipShAddChip<F> =
    VmChipWrapper<F, BitManipShAddFiller<Rv64BaseAluRegU16AdapterFiller>>;

pub type Rv64BitManipSlliUwAir = VmAirWrapper<Rv64BaseAluImmU16AdapterAir, BitManipSlliUwCoreAir>;
pub type Rv64BitManipSlliUwExecutor = BitManipSlliUwExecutor<Rv64BaseAluImmU16AdapterExecutor>;
pub type Rv64BitManipSlliUwChip<F> =
    VmChipWrapper<F, BitManipSlliUwFiller<Rv64BaseAluImmU16AdapterFiller>>;

pub type Rv64BitManipRegAir = VmAirWrapper<Rv64BaseAluRegU16AdapterAir, BitManipRegCoreAir>;
pub type Rv64BitManipRegExecutor = BitManipRegExecutor<Rv64BaseAluRegU16AdapterExecutor>;
pub type Rv64BitManipRegChip<F> =
    VmChipWrapper<F, BitManipRegFiller<Rv64BaseAluRegU16AdapterFiller>>;

pub type Rv64BitManipImmAir = VmAirWrapper<Rv64BaseAluImmU16AdapterAir, BitManipImmCoreAir>;
pub type Rv64BitManipImmExecutor = BitManipImmExecutor<Rv64BaseAluImmU16AdapterExecutor>;
pub type Rv64BitManipImmChip<F> =
    VmChipWrapper<F, BitManipImmFiller<Rv64BaseAluImmU16AdapterFiller>>;

pub type Rv64BitManipBitwiseInvAir =
    VmAirWrapper<Rv64BaseAluRegAdapterAir, BitManipBitwiseInvCoreAir>;
pub type Rv64BitManipBitwiseInvExecutor = BitManipBitwiseInvExecutor<Rv64BaseAluRegAdapterExecutor>;
pub type Rv64BitManipBitwiseInvChip<F> =
    VmChipWrapper<F, BitManipBitwiseInvFiller<Rv64BaseAluRegAdapterFiller>>;

pub type Rv64BitManipMinMaxAir = VmAirWrapper<Rv64BaseAluRegU16AdapterAir, BitManipMinMaxCoreAir>;
pub type Rv64BitManipMinMaxExecutor = BitManipMinMaxExecutor<Rv64BaseAluRegU16AdapterExecutor>;
pub type Rv64BitManipMinMaxChip<F> =
    VmChipWrapper<F, BitManipMinMaxFiller<Rv64BaseAluRegU16AdapterFiller>>;

pub type Rv64BitManipByteUnaryAir =
    VmAirWrapper<Rv64BaseAluImmAdapterAir, BitManipByteUnaryCoreAir>;
pub type Rv64BitManipByteUnaryExecutor = BitManipByteUnaryExecutor<Rv64BaseAluImmAdapterExecutor>;
pub type Rv64BitManipByteUnaryChip<F> =
    VmChipWrapper<F, BitManipByteUnaryFiller<Rv64BaseAluImmAdapterFiller>>;

pub(crate) const BITMANIP_NUM_LIMBS: usize = BLOCK_FE_WIDTH;
pub(crate) const BITMANIP_LIMB_BITS: usize = U16_BITS;
pub(crate) const BITMANIP_NUM_BITS: usize = BITMANIP_NUM_LIMBS * BITMANIP_LIMB_BITS;
