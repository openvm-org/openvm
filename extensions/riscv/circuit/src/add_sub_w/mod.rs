use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::{
    adapters::{
        Rv64BaseAluWRegU16AdapterAir, Rv64BaseAluWRegU16AdapterExecutor,
        Rv64BaseAluWRegU16AdapterFiller, RV64_WORD_U16_LIMBS, U16_BITS,
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

pub type Rv64AddSubWAir = VmAirWrapper<Rv64BaseAluWRegU16AdapterAir, AddSubWCoreAir>;
pub type Rv64AddSubWExecutor = AddSubWExecutor<Rv64BaseAluWRegU16AdapterExecutor>;
pub type Rv64AddSubWChip<F> = VmChipWrapper<F, AddSubWFiller<Rv64BaseAluWRegU16AdapterFiller>>;
