use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::{
    adapters::{
        Rv64MultWAdapterAir, Rv64MultWAdapterExecutor, Rv64MultWAdapterFiller, RV64_CELL_BITS,
        RV64_WORD_NUM_LIMBS,
    },
    divrem::{DivRemCoreAir, DivRemExecutor, DivRemFiller},
};

mod execution;

pub type DivRemWCoreAir = DivRemCoreAir<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;
pub type DivRemWExecutor<A> = DivRemExecutor<A, RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;
pub type DivRemWFiller<A> = DivRemFiller<A, RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;

#[cfg(all(feature = "cuda"))]
mod cuda;
#[cfg(all(feature = "cuda"))]
pub use cuda::*;

#[cfg(all(test))]
mod tests;

pub type Rv64DivRemWAir = VmAirWrapper<Rv64MultWAdapterAir, DivRemWCoreAir>;
pub type Rv64DivRemWExecutor = DivRemWExecutor<Rv64MultWAdapterExecutor>;
pub type Rv64DivRemWChip<F> = VmChipWrapper<F, DivRemWFiller<Rv64MultWAdapterFiller>>;
