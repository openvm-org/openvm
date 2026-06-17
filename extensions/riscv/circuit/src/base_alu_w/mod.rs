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

pub type BaseAluWCoreAir = AddSubCoreAir<RV64_WORD_U16_LIMBS, U16_BITS>;
pub type BaseAluWFiller<A> = AddSubFiller<A, RV64_WORD_U16_LIMBS, U16_BITS>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub type Rv64BaseAluWAir = VmAirWrapper<Rv64BaseAluWU16AdapterAir, BaseAluWCoreAir>;
pub type Rv64BaseAluWExecutor = BaseAluWExecutor<Rv64BaseAluWU16AdapterExecutor>;
pub type Rv64BaseAluWChip<F> = VmChipWrapper<F, BaseAluWFiller<Rv64BaseAluWU16AdapterFiller>>;
