use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadAdapterAir, Rv64LoadAdapterExecutor},
    load_sign_extend::{
        common::{LoadSignExtendExecutor, KIND_HALFWORD},
        width_aligned::core::{
            LoadSignExtendWidthAlignedCoreAir, LoadSignExtendWidthAlignedFiller,
        },
    },
};

pub const LOAD_SIGN_EXTEND_HALFWORD_CASES: usize = 4;
pub const LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH: usize = 2;

pub type LoadSignExtendHalfwordCoreAir = LoadSignExtendWidthAlignedCoreAir<
    KIND_HALFWORD,
    LOAD_SIGN_EXTEND_HALFWORD_CASES,
    LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH,
>;
pub type LoadSignExtendHalfwordFiller = LoadSignExtendWidthAlignedFiller<
    crate::adapters::Rv64LoadAdapterFiller,
    KIND_HALFWORD,
    LOAD_SIGN_EXTEND_HALFWORD_CASES,
    LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH,
>;

pub type Rv64LoadSignExtendHalfwordAir =
    VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendHalfwordCoreAir>;
pub type Rv64LoadSignExtendHalfwordExecutor =
    LoadSignExtendExecutor<Rv64LoadAdapterExecutor, KIND_HALFWORD>;
pub type Rv64LoadSignExtendHalfwordChip<F> = VmChipWrapper<F, LoadSignExtendHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
