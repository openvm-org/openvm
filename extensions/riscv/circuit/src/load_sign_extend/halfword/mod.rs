use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64LoadAdapterAir, Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller, LOAD_WIDTH_HALFWORD,
    },
    load_sign_extend::{
        common::LoadSignExtendExecutor,
        core::{LoadSignExtendCoreAir, LoadSignExtendFiller},
    },
};

pub const LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH: usize = 3;
/// Cells loaded by a halfword load: `LOAD_WIDTH_HALFWORD / 2`.
pub const LOAD_SIGN_EXTEND_HALFWORD_LOADED_CELLS: usize = 1;

pub type LoadSignExtendHalfwordCoreAir = LoadSignExtendCoreAir<
    LOAD_WIDTH_HALFWORD,
    LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH,
    LOAD_SIGN_EXTEND_HALFWORD_LOADED_CELLS,
>;
pub type LoadSignExtendHalfwordFiller = LoadSignExtendFiller<
    Rv64LoadAdapterFiller,
    LOAD_WIDTH_HALFWORD,
    LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH,
    LOAD_SIGN_EXTEND_HALFWORD_LOADED_CELLS,
>;

pub type Rv64LoadSignExtendHalfwordAir =
    VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendHalfwordCoreAir>;
pub type Rv64LoadSignExtendHalfwordExecutor =
    LoadSignExtendExecutor<Rv64LoadAdapterExecutor<LOAD_WIDTH_HALFWORD>, LOAD_WIDTH_HALFWORD>;
pub type Rv64LoadSignExtendHalfwordChip<F> = VmChipWrapper<F, LoadSignExtendHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
