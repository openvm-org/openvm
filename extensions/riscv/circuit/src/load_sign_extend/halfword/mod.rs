use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::{
        Rv64LoadMultiByteAdapterAir, Rv64LoadMultiByteAdapterExecutor,
        Rv64LoadMultiByteAdapterFiller, BYTE_SHIFT_SELECTOR_WIDTH, LOAD_WIDTH_HALFWORD,
    },
    load_sign_extend::{
        common::LoadSignExtendExecutor,
        core::{LoadSignExtendCoreAir, LoadSignExtendFiller},
    },
};

/// Cells overlapped by an odd-shift halfword load: `LOAD_WIDTH_HALFWORD / 2 + 1`.
pub const LOAD_SIGN_EXTEND_HALFWORD_OVERLAP_CELLS: usize = 2;

pub type LoadSignExtendHalfwordCoreAir = LoadSignExtendCoreAir<
    LOAD_WIDTH_HALFWORD,
    BYTE_SHIFT_SELECTOR_WIDTH,
    LOAD_SIGN_EXTEND_HALFWORD_OVERLAP_CELLS,
>;
pub type LoadSignExtendHalfwordFiller = LoadSignExtendFiller<
    Rv64LoadMultiByteAdapterFiller,
    LOAD_WIDTH_HALFWORD,
    BYTE_SHIFT_SELECTOR_WIDTH,
    LOAD_SIGN_EXTEND_HALFWORD_OVERLAP_CELLS,
>;

pub type Rv64LoadSignExtendHalfwordAir =
    VmAirWrapper<Rv64LoadMultiByteAdapterAir, LoadSignExtendHalfwordCoreAir>;
pub type Rv64LoadSignExtendHalfwordExecutor = LoadSignExtendExecutor<
    Rv64LoadMultiByteAdapterExecutor<LOAD_WIDTH_HALFWORD>,
    LOAD_WIDTH_HALFWORD,
>;
pub type Rv64LoadSignExtendHalfwordChip<F> = VmChipWrapper<F, LoadSignExtendHalfwordFiller>;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;
