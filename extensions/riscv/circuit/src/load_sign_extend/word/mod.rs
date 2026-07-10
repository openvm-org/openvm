use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64LoadAdapterAir, Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller, LOAD_WIDTH_WORD,
    },
    load_sign_extend::{
        common::LoadSignExtendExecutor,
        core::{LoadSignExtendCoreAir, LoadSignExtendFiller},
    },
};

pub const LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH: usize = 3;
/// Cells loaded by a word load: `LOAD_WIDTH_WORD / 2`.
pub const LOAD_SIGN_EXTEND_WORD_LOADED_CELLS: usize = 2;

pub type LoadSignExtendWordCoreAir = LoadSignExtendCoreAir<
    LOAD_WIDTH_WORD,
    LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH,
    LOAD_SIGN_EXTEND_WORD_LOADED_CELLS,
>;
pub type LoadSignExtendWordFiller = LoadSignExtendFiller<
    Rv64LoadAdapterFiller,
    LOAD_WIDTH_WORD,
    LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH,
    LOAD_SIGN_EXTEND_WORD_LOADED_CELLS,
>;

pub type Rv64LoadSignExtendWordAir = VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendWordCoreAir>;
pub type Rv64LoadSignExtendWordExecutor =
    LoadSignExtendExecutor<Rv64LoadAdapterExecutor<LOAD_WIDTH_WORD>, LOAD_WIDTH_WORD>;
pub type Rv64LoadSignExtendWordChip<F> = VmChipWrapper<F, LoadSignExtendWordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
