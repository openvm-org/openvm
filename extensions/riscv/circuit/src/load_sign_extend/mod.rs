use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_instructions::riscv::{RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS};

use crate::adapters::{Rv64LoadStoreAdapterAir, Rv64LoadStoreAdapterExecutor};

mod core;
mod execution;
pub use core::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;
#[cfg(feature = "aot")]
mod aot;

#[cfg(all(test, any()))] // TODO: port tests to RV64
mod tests;

pub type Rv64LoadSignExtendAir = VmAirWrapper<
    Rv64LoadStoreAdapterAir,
    LoadSignExtendCoreAir<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>,
>;
pub type Rv64LoadSignExtendExecutor =
    LoadSignExtendExecutor<Rv64LoadStoreAdapterExecutor, RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>;
pub type Rv64LoadSignExtendChip<F> = VmChipWrapper<
    F,
    LoadSignExtendFiller<
        crate::adapters::Rv64LoadStoreAdapterFiller,
        RV64_REGISTER_NUM_LIMBS,
        RV64_CELL_BITS,
    >,
>;
