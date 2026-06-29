use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadAdapterAir, Rv64LoadAdapterExecutor},
    load_sign_extend::{
        common::{LoadSignExtendExecutor, KIND_WORD},
        width_aligned::core::{
            LoadSignExtendWidthAlignedCoreAir, LoadSignExtendWidthAlignedFiller,
        },
    },
};

pub const LOAD_SIGN_EXTEND_WORD_CASES: usize = 2;
pub const LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH: usize = 1;

pub type LoadSignExtendWordCoreAir = LoadSignExtendWidthAlignedCoreAir<
    KIND_WORD,
    LOAD_SIGN_EXTEND_WORD_CASES,
    LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH,
>;
pub type LoadSignExtendWordFiller = LoadSignExtendWidthAlignedFiller<
    crate::adapters::Rv64LoadAdapterFiller,
    KIND_WORD,
    LOAD_SIGN_EXTEND_WORD_CASES,
    LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH,
>;

pub type Rv64LoadSignExtendWordAir = VmAirWrapper<Rv64LoadAdapterAir, LoadSignExtendWordCoreAir>;
pub type Rv64LoadSignExtendWordExecutor =
    LoadSignExtendExecutor<Rv64LoadAdapterExecutor, KIND_WORD>;
pub type Rv64LoadSignExtendWordChip<F> = VmChipWrapper<F, LoadSignExtendWordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
