//! WARNING: the order of fields in the structs is important, do not change it

use openvm_circuit::{
    arch::ExecutionState,
    system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_sha_air::{
    ShaDigestCols, ShaDigestColsRef, ShaDigestColsRefMut, ShaRoundCols, ShaRoundColsRef,
    ShaRoundColsRefMut,
};
use openvm_sha_macros::ColsRef;

use super::{SHA_REGISTER_READS, SHA_WRITE_SIZE};
use crate::ShaChipConfig;

/// the first C::ROUND_ROWS rows of every SHA block will be of type ShaVmRoundCols and the last row will be of type ShaVmDigestCols
#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(ShaChipConfig)]
pub struct ShaVmRoundCols<
    T,
    const WORD_BITS: usize,
    const WORD_U8S: usize,
    const WORD_U16S: usize,
    const ROUNDS_PER_ROW: usize,
    const ROUNDS_PER_ROW_MINUS_ONE: usize,
    const ROW_VAR_CNT: usize,
> {
    pub control: ShaVmControlCols<T>,
    pub inner: ShaRoundCols<
        T,
        WORD_BITS,
        WORD_U8S,
        WORD_U16S,
        ROUNDS_PER_ROW,
        ROUNDS_PER_ROW_MINUS_ONE,
        ROW_VAR_CNT,
    >,
    #[plain]
    pub read_aux: MemoryReadAuxCols<T>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(ShaChipConfig)]
pub struct ShaVmDigestCols<
    T,
    const WORD_BITS: usize,
    const WORD_U8S: usize,
    const WORD_U16S: usize,
    const HASH_WORDS: usize,
    const ROUNDS_PER_ROW: usize,
    const ROUNDS_PER_ROW_MINUS_ONE: usize,
    const ROW_VAR_CNT: usize,
    const WRITE_SIZE: usize,
> {
    pub control: ShaVmControlCols<T>,
    pub inner: ShaDigestCols<
        T,
        WORD_BITS,
        WORD_U8S,
        WORD_U16S,
        HASH_WORDS,
        ROUNDS_PER_ROW,
        ROUNDS_PER_ROW_MINUS_ONE,
        ROW_VAR_CNT,
    >,
    #[plain]
    pub from_state: ExecutionState<T>,
    /// It is counter intuitive, but we will constrain the register reads on the very last row of every message
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs2_ptr: T,
    #[plain]
    pub dst_ptr: [T; RV32_REGISTER_NUM_LIMBS],
    pub src_ptr: [T; RV32_REGISTER_NUM_LIMBS],
    pub len_data: [T; RV32_REGISTER_NUM_LIMBS],
    #[plain]
    pub register_reads_aux: [MemoryReadAuxCols<T>; SHA_REGISTER_READS],
    #[plain]
    pub writes_aux: MemoryWriteAuxCols<T, SHA_WRITE_SIZE>,
}

/// These are the columns that are used on both round and digest rows
#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(ShaChipConfig)]
pub struct ShaVmControlCols<T> {
    /// Note: We will use the buffer in `inner.message_schedule` as the message data
    /// This is the length of the entire message
    pub len: T,
    /// Need to keep timestamp and read_ptr since block reads don't have the necessary information
    pub cur_timestamp: T,
    pub read_ptr: T,
    /// Padding flags which will be used to encode the the number of non-padding cells in the current row
    pub pad_flags: [T; 6],
    /// A boolean flag that indicates whether a padding already occurred
    pub padding_occurred: T,
}
