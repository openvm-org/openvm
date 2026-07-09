use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadAdapterAir, Rv64LoadAdapterExecutor, LOAD_WIDTH_HALFWORD},
    load_sign_extend::{
        common::LoadSignExtendExecutor,
        core::{LoadSignExtendWidthAlignedCoreAir, LoadSignExtendWidthAlignedFiller},
    },
};

pub const LOAD_SIGN_EXTEND_HALFWORD_CASES: usize = 4;
pub const LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH: usize = 2;

pub type LoadSignExtendHalfwordCoreAir = LoadSignExtendWidthAlignedCoreAir<
    LOAD_WIDTH_HALFWORD,
    LOAD_SIGN_EXTEND_HALFWORD_CASES,
    LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH,
>;
pub type LoadSignExtendHalfwordFiller = LoadSignExtendWidthAlignedFiller<
    crate::adapters::Rv64LoadAdapterFiller,
    LOAD_WIDTH_HALFWORD,
    LOAD_SIGN_EXTEND_HALFWORD_CASES,
    LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH,
>;

pub type Rv64LoadSignExtendHalfwordAir =
    VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendHalfwordCoreAir>;
pub type Rv64LoadSignExtendHalfwordExecutor =
    LoadSignExtendExecutor<Rv64LoadAdapterExecutor, LOAD_WIDTH_HALFWORD>;
pub type Rv64LoadSignExtendHalfwordChip<F> = VmChipWrapper<F, LoadSignExtendHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
