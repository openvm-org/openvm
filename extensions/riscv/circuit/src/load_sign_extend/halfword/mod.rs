use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{Rv64LoadStoreAdapterAir, Rv64LoadStoreAdapterExecutor},
    load_sign_extend::aligned::core::{LoadSignExtendAlignedCoreAir, LoadSignExtendAlignedFiller},
    loadstore::common::{LoadStoreExecutor, KIND_HALFWORD},
};

pub const LOAD_SIGN_EXTEND_HALFWORD_CASES: usize = 4;
pub const LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH: usize = 2;

pub type LoadSignExtendHalfwordCoreAir = LoadSignExtendAlignedCoreAir<
    KIND_HALFWORD,
    LOAD_SIGN_EXTEND_HALFWORD_CASES,
    LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH,
>;
pub type LoadSignExtendHalfwordFiller = LoadSignExtendAlignedFiller<
    crate::adapters::Rv64LoadStoreAdapterFiller,
    KIND_HALFWORD,
    LOAD_SIGN_EXTEND_HALFWORD_CASES,
    LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH,
>;

pub type Rv64LoadSignExtendHalfwordAir =
    VmAirWrapper<Rv64LoadStoreAdapterAir, LoadSignExtendHalfwordCoreAir>;
pub type Rv64LoadSignExtendHalfwordExecutor =
    LoadStoreExecutor<Rv64LoadStoreAdapterExecutor, KIND_HALFWORD>;
pub type Rv64LoadSignExtendHalfwordChip<F> = VmChipWrapper<F, LoadSignExtendHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;
