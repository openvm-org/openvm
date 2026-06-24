use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadStoreAdapterAir, Rv64LoadStoreAdapterExecutor},
    load_sign_extend::aligned::core::{LoadSignExtendAlignedCoreAir, LoadSignExtendAlignedFiller},
    loadstore::common::{LoadStoreExecutor, KIND_WORD},
};

pub const LOAD_SIGN_EXTEND_WORD_CASES: usize = 2;
pub const LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH: usize = 1;

pub type LoadSignExtendWordCoreAir = LoadSignExtendAlignedCoreAir<
    KIND_WORD,
    LOAD_SIGN_EXTEND_WORD_CASES,
    LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH,
>;
pub type LoadSignExtendWordFiller = LoadSignExtendAlignedFiller<
    crate::adapters::Rv64LoadStoreAdapterFiller,
    KIND_WORD,
    LOAD_SIGN_EXTEND_WORD_CASES,
    LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH,
>;

pub type Rv64LoadSignExtendWordAir =
    VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendWordCoreAir>;
pub type Rv64LoadSignExtendWordExecutor =
    LoadStoreExecutor<Rv64LoadStoreAdapterExecutor, KIND_WORD>;
pub type Rv64LoadSignExtendWordChip<F> = VmChipWrapper<F, LoadSignExtendWordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
