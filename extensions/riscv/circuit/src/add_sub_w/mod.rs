use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::{
    adapters::{
        Rv64BaseAluWU16AdapterAir, Rv64BaseAluWU16AdapterExecutor, Rv64BaseAluWU16AdapterFiller,
        RV64_WORD_U16_LIMBS, U16_BITS,
    },
    add_sub::{AddSubCoreAir, AddSubFiller},
};

mod execution;
mod preflight;
pub use preflight::*;

pub type AddSubWCoreAir = AddSubCoreAir<RV64_WORD_U16_LIMBS, U16_BITS, false>;
pub type AddSubWFiller<A> = AddSubFiller<A, RV64_WORD_U16_LIMBS, U16_BITS, false>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64AddSubWAir = VmAirWrapper<Rv64BaseAluWU16AdapterAir, AddSubWCoreAir>;
pub type Rv64AddSubWExecutor = AddSubWExecutor<Rv64BaseAluWU16AdapterExecutor>;
pub type Rv64AddSubWChip<F> = VmChipWrapper<F, AddSubWFiller<Rv64BaseAluWU16AdapterFiller>>;
