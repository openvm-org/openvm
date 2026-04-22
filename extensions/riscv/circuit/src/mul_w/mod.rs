use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::{
    adapters::{
        Rv64MultWAdapterAir, Rv64MultWAdapterExecutor, Rv64MultWAdapterFiller, RV64_CELL_BITS,
        RV64_WORD_NUM_LIMBS,
    },
    mul::{MultiplicationCoreAir, MultiplicationExecutor, MultiplicationFiller},
};

mod execution;

pub type MulWCoreAir = MultiplicationCoreAir<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;
pub type MulWExecutor<A> = MultiplicationExecutor<A, RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;
pub type MulWFiller<A> = MultiplicationFiller<A, RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;

#[cfg(all(feature = "cuda", not(feature = "cuda")))] // TODO: RV64 GPU port
mod cuda;
#[cfg(all(feature = "cuda", not(feature = "cuda")))] // TODO: RV64 GPU port
pub use cuda::*;

#[cfg(all(test, any()))] // TODO: port tests to RV64
mod tests;

pub type Rv64MulWAir = VmAirWrapper<Rv64MultWAdapterAir, MulWCoreAir>;
pub type Rv64MulWExecutor = MulWExecutor<Rv64MultWAdapterExecutor>;
pub type Rv64MulWChip<F> = VmChipWrapper<F, MulWFiller<Rv64MultWAdapterFiller>>;
