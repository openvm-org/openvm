use openvm_circuit::{
    arch::{ExecutionState, BLOCK_FE_WIDTH},
    system::memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};
use openvm_circuit_primitives::ColsRef;
use openvm_instructions::riscv::RV64_WORD_NUM_LIMBS;

use crate::{Sha2MainChipConfig, SHA2_REGISTER_READS, SHA2_WRITE_SIZE};

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(Sha2MainChipConfig)]
pub struct Sha2Cols<
    T,
    const BLOCK_BYTES: usize,
    const STATE_BYTES: usize,
    const STATE_U16S: usize,
> {
    pub block: Sha2BlockCols<T, BLOCK_BYTES, STATE_BYTES, STATE_U16S>,
    pub instruction: Sha2InstructionCols<T>,
    pub mem: Sha2MemoryCols<T, BLOCK_BYTES, STATE_BYTES, SHA2_WRITE_SIZE>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(Sha2MainChipConfig)]
pub struct Sha2BlockCols<
    T,
    const BLOCK_BYTES: usize,
    const STATE_BYTES: usize,
    const STATE_U16S: usize,
> {
    /// Identifier of this row in the interactions between the two chips
    pub request_id: T,
    /// Input bytes for this block
    pub message_bytes: [T; BLOCK_BYTES],
    /// Previous state of the SHA-2 hasher object, as little-endian 16-bit cells. Each cell
    /// holds two consecutive message bytes packed `lo | hi << 8` to match the bus interaction
    /// with the block hasher.
    pub prev_state: [T; STATE_U16S],
    /// New state of the SHA-2 hasher object after processing this block, as little-endian
    /// **bytes**. The block hasher receives `final_hash` byte-by-byte, so this column stays
    /// byte-shaped.
    pub new_state: [T; STATE_BYTES],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(Sha2MainChipConfig)]
pub struct Sha2InstructionCols<T> {
    /// True for all rows that are part of opcode execution.
    /// False on dummy rows only used to pad the height.
    pub is_enabled: T,
    #[aligned_borrow]
    pub from_state: ExecutionState<T>,
    /// Pointer to address space 1 `dst` register
    pub dst_reg_ptr: T,
    /// Pointer to address space 1 `state` register
    pub state_reg_ptr: T,
    /// Pointer to address space 1 `input` register
    pub input_reg_ptr: T,
    /// Low 4 bytes of \[dst_reg_ptr:8\]_1
    pub dst_ptr_limbs: [T; RV64_WORD_NUM_LIMBS],
    /// Low 4 bytes of \[state_reg_ptr:8\]_1
    pub state_ptr_limbs: [T; RV64_WORD_NUM_LIMBS],
    /// Low 4 bytes of \[input_reg_ptr:8\]_1
    pub input_ptr_limbs: [T; RV64_WORD_NUM_LIMBS],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(Sha2MainChipConfig)]
pub struct Sha2MemoryCols<
    T,
    const BLOCK_READS: usize,
    const STATE_READS: usize,
    const STATE_WRITES: usize,
> {
    #[aligned_borrow]
    pub register_aux: [MemoryReadAuxCols<T>; SHA2_REGISTER_READS],
    #[aligned_borrow]
    pub input_reads: [MemoryReadAuxCols<T>; BLOCK_READS],
    #[aligned_borrow]
    pub state_reads: [MemoryReadAuxCols<T>; STATE_READS],
    #[aligned_borrow]
    pub write_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; STATE_WRITES],
}
