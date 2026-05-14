use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::{
    adapters::{
        Rv64BaseAluWAdapterAir, Rv64BaseAluWAdapterExecutor, Rv64BaseAluWAdapterFiller,
        RV64_CELL_BITS, RV64_WORD_NUM_LIMBS,
    },
    add_sub::{AddSubCoreAir, AddSubExecutor, AddSubFiller},
};

mod execution;

pub type AddSubWCoreAir = AddSubCoreAir<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;
pub type AddSubWExecutor<A> = AddSubExecutor<A, RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;
pub type AddSubWFiller<A> = AddSubFiller<A, RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64AddSubWAir = VmAirWrapper<Rv64BaseAluWAdapterAir, AddSubWCoreAir>;
pub type Rv64AddSubWExecutor = AddSubWExecutor<Rv64BaseAluWAdapterExecutor>;
pub type Rv64AddSubWChip<F> = VmChipWrapper<F, AddSubWFiller<Rv64BaseAluWAdapterFiller>>;
