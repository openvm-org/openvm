use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64LoadMultiByteAdapterAir, Rv64LoadMultiByteAdapterExecutor,
        Rv64LoadMultiByteAdapterFiller, WORD_ACCESS_WIDTH,
    },
    load_sign_extend::{
        common::LoadSignExtendExecutor,
        core::{LoadSignExtendCoreAir, LoadSignExtendFiller},
    },
};

/// Cells overlapped by an odd-shift word load: `WORD_ACCESS_WIDTH / 2 + 1`.
pub const LOAD_SIGN_EXTEND_WORD_OVERLAP_CELLS: usize = WORD_ACCESS_WIDTH / 2 + 1;

pub type LoadSignExtendWordCoreAir =
    LoadSignExtendCoreAir<WORD_ACCESS_WIDTH, LOAD_SIGN_EXTEND_WORD_OVERLAP_CELLS>;
pub type LoadSignExtendWordFiller = LoadSignExtendFiller<
    Rv64LoadMultiByteAdapterFiller,
    WORD_ACCESS_WIDTH,
    LOAD_SIGN_EXTEND_WORD_OVERLAP_CELLS,
>;

pub type Rv64LoadSignExtendWordAir =
    VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendWordCoreAir>;
pub type Rv64LoadSignExtendWordExecutor =
    LoadSignExtendExecutor<Rv64LoadMultiByteAdapterExecutor<WORD_ACCESS_WIDTH>, WORD_ACCESS_WIDTH>;
pub type Rv64LoadSignExtendWordChip<F> = VmChipWrapper<F, LoadSignExtendWordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
