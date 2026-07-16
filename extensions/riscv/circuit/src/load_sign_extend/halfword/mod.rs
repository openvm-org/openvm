use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64LoadMultiByteAdapterAir, Rv64LoadMultiByteAdapterExecutor,
        Rv64LoadMultiByteAdapterFiller, HALFWORD_ACCESS_WIDTH,
    },
    load_sign_extend::{
        common::LoadSignExtendExecutor,
        core::{LoadSignExtendCoreAir, LoadSignExtendFiller},
    },
};

/// Cells overlapped by an odd-shift halfword load: `HALFWORD_ACCESS_WIDTH / 2 + 1`.
pub const LOAD_SIGN_EXTEND_HALFWORD_OVERLAP_CELLS: usize = HALFWORD_ACCESS_WIDTH / 2 + 1;

pub type LoadSignExtendHalfwordCoreAir =
    LoadSignExtendCoreAir<HALFWORD_ACCESS_WIDTH, LOAD_SIGN_EXTEND_HALFWORD_OVERLAP_CELLS>;
pub type LoadSignExtendHalfwordFiller = LoadSignExtendFiller<
    Rv64LoadMultiByteAdapterFiller,
    HALFWORD_ACCESS_WIDTH,
    LOAD_SIGN_EXTEND_HALFWORD_OVERLAP_CELLS,
>;

pub type Rv64LoadSignExtendHalfwordAir =
    VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendHalfwordCoreAir>;
pub type Rv64LoadSignExtendHalfwordExecutor = LoadSignExtendExecutor<
    Rv64LoadMultiByteAdapterExecutor<HALFWORD_ACCESS_WIDTH>,
    HALFWORD_ACCESS_WIDTH,
>;
pub type Rv64LoadSignExtendHalfwordChip<F> = VmChipWrapper<F, LoadSignExtendHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
