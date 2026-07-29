use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper, BLOCK_FE_WIDTH};

use crate::adapters::{
    Rv64BaseAluImmU16AdapterAir, Rv64BaseAluImmU16AdapterExecutor, Rv64BaseAluImmU16AdapterFiller,
    Rv64BaseAluRegU16AdapterAir, Rv64BaseAluRegU16AdapterExecutor, Rv64BaseAluRegU16AdapterFiller,
    U16_BITS,
};

mod core;
mod execution;
pub use core::*;

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

pub(crate) const BITMANIP_NUM_LIMBS: usize = BLOCK_FE_WIDTH;
pub(crate) const BITMANIP_LIMB_BITS: usize = U16_BITS;
pub(crate) const BITMANIP_NUM_BITS: usize = BITMANIP_NUM_LIMBS * BITMANIP_LIMB_BITS;
