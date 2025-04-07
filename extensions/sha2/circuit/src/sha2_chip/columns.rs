//! WARNING: the order of fields in the structs is important, do not change it

use openvm_circuit::{
    arch::ExecutionState,
    system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};
use openvm_circuit_primitives::AlignedBorrow;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_sha256_air::{Sha256DigestCols, Sha256RoundCols};

use super::{SHA256_REGISTER_READS, SHA256_WRITE_SIZE};

/// the first 16 rows of every SHA256 block will be of type Sha256VmRoundCols and the last row will
/// be of type Sha256VmDigestCols
#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct Sha256VmRoundCols<T> {
    pub control: Sha256VmControlCols<T>,
    pub inner: Sha256RoundCols<T>,
    pub read_aux: MemoryReadAuxCols<T>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct Sha256VmDigestCols<T> {
    pub control: Sha256VmControlCols<T>,
    pub inner: Sha256DigestCols<T>,

    pub from_state: ExecutionState<T>,
    /// It is counter intuitive, but we will constrain the register reads on the very last row of
    /// every message
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs2_ptr: T,
    pub dst_ptr: [T; RV32_REGISTER_NUM_LIMBS],
    pub src_ptr: [T; RV32_REGISTER_NUM_LIMBS],
    pub len_data: [T; RV32_REGISTER_NUM_LIMBS],
    pub register_reads_aux: [MemoryReadAuxCols<T>; SHA256_REGISTER_READS],
    pub writes_aux: MemoryWriteAuxCols<T, SHA256_WRITE_SIZE>,
}

/// These are the columns that are used on both round and digest rows
#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct Sha256VmControlCols<T> {
    /// Note: We will use the buffer in `inner.message_schedule` as the message data
    /// This is the length of the entire message in bytes
    pub len: T,
    /// Need to keep timestamp and read_ptr since block reads don't have the necessary information
    pub cur_timestamp: T,
    pub read_ptr: T,
    /// Padding flags which will be used to encode the the number of non-padding cells in the
    /// current row
    pub pad_flags: [T; 6],
    /// A boolean flag that indicates whether a padding already occurred
    pub padding_occurred: T,
}

/// Width of the Sha256VmControlCols
pub const SHA256VM_CONTROL_WIDTH: usize = Sha256VmControlCols::<u8>::width();
/// Width of the Sha256VmRoundCols
pub const SHA256VM_ROUND_WIDTH: usize = Sha256VmRoundCols::<u8>::width();
/// Width of the Sha256VmDigestCols
pub const SHA256VM_DIGEST_WIDTH: usize = Sha256VmDigestCols::<u8>::width();
/// Width of the Sha256Cols
pub const SHA256VM_WIDTH: usize = if SHA256VM_ROUND_WIDTH > SHA256VM_DIGEST_WIDTH {
    SHA256VM_ROUND_WIDTH
} else {
    SHA256VM_DIGEST_WIDTH
};
