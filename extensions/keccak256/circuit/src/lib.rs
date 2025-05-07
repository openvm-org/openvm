//! Stateful keccak256 hasher. Handles full keccak sponge (padding, absorb, keccak-f) on
//! variable length inputs read from VM memory.
use std::{
    array::from_fn,
    cmp::min,
    sync::{Arc, Mutex},
};

use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use tiny_keccak::{Hasher, Keccak};
use utils::num_keccak_f;

pub mod air;
pub mod columns;
pub mod trace;
pub mod utils;

mod extension;
pub use extension::*;

#[cfg(test)]
mod tests;

pub use air::KeccakVmAir;
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionBus, ExecutionError, ExecutionState, InstructionExecutor},
    system::{
        memory::{offline_checker::MemoryBridge, MemoryController, OfflineMemory, RecordId},
        program::ProgramBus,
    },
};
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_NUM_LIMBS, LocalOpcode,
};
use openvm_keccak256_transpiler::Rv32KeccakOpcode;
use openvm_rv32im_circuit::adapters::read_rv32_register;

// ==== Constants for register/memory adapter ====
/// Register reads to get dst, src, len
const KECCAK_REGISTER_READS: usize = 3;
/// Number of cells to read/write in a single memory access
const KECCAK_WORD_SIZE: usize = 4;
/// Memory reads for absorb per row
const KECCAK_ABSORB_READS: usize = KECCAK_RATE_BYTES / KECCAK_WORD_SIZE;
/// Memory writes for digest per row
const KECCAK_DIGEST_WRITES: usize = KECCAK_DIGEST_BYTES / KECCAK_WORD_SIZE;

// ==== Do not change these constants! ====
/// Total number of sponge bytes: number of rate bytes + number of capacity
/// bytes.
pub const KECCAK_WIDTH_BYTES: usize = 200;
/// Total number of 16-bit limbs in the sponge.
pub const KECCAK_WIDTH_U16S: usize = KECCAK_WIDTH_BYTES / 2;
/// Number of rate bytes.
pub const KECCAK_RATE_BYTES: usize = 136;
/// Number of 16-bit rate limbs.
pub const KECCAK_RATE_U16S: usize = KECCAK_RATE_BYTES / 2;
/// Number of absorb rounds, equal to rate in u64s.
pub const NUM_ABSORB_ROUNDS: usize = KECCAK_RATE_BYTES / 8;
/// Number of capacity bytes.
pub const KECCAK_CAPACITY_BYTES: usize = 64;
/// Number of 16-bit capacity limbs.
pub const KECCAK_CAPACITY_U16S: usize = KECCAK_CAPACITY_BYTES / 2;
/// Number of output digest bytes used during the squeezing phase.
pub const KECCAK_DIGEST_BYTES: usize = 32;
/// Number of 64-bit digest limbs.
pub const KECCAK_DIGEST_U64S: usize = KECCAK_DIGEST_BYTES / 8;

pub struct KeccakVmChip<F: PrimeField32> {
    pub air: KeccakVmAir,
    /// IO and memory data necessary for each opcode call
    pub records: Vec<KeccakRecord<F>>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,

    offset: usize,

    offline_memory: Arc<Mutex<OfflineMemory<F>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KeccakRecord<F> {
    pub pc: F,
    pub dst_read: RecordId,
    pub src_read: RecordId,
    pub len_read: RecordId,
    pub input_blocks: Vec<KeccakInputBlock>,
    pub digest_writes: [RecordId; KECCAK_DIGEST_WRITES],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KeccakInputBlock {
    /// Memory reads for non-padding bytes in this block. Length is at most [KECCAK_RATE_BYTES /
    /// KECCAK_WORD_SIZE].
    pub reads: Vec<RecordId>,
    /// Index in `reads` of the memory read for < KECCAK_WORD_SIZE bytes, if any.
    pub partial_read_idx: Option<usize>,
    /// Bytes with padding. Can be derived from `bytes_read` but we store for convenience.
    #[serde(with = "BigArray")]
    pub padded_bytes: [u8; KECCAK_RATE_BYTES],
    pub remaining_len: usize,
    pub src: usize,
    pub is_new_start: bool,
}

impl<F: PrimeField32> KeccakVmChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_bridge: MemoryBridge,
        address_bits: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
        offset: usize,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    ) -> Self {
        Self {
            air: KeccakVmAir::new(
                ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                bitwise_lookup_chip.bus(),
                address_bits,
                offset,
            ),
            bitwise_lookup_chip,
            records: Vec::new(),
            offset,
            offline_memory,
        }
    }
}

impl<F: PrimeField32> InstructionExecutor<F> for KeccakVmChip<F> {
    fn execute(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;
        let local_opcode = Rv32KeccakOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        debug_assert_eq!(local_opcode, Rv32KeccakOpcode::KECCAK256);

        let mut timestamp_delta = 3;
        let (dst_read, dst) = read_rv32_register(memory, d, a);
        let (src_read, src) = read_rv32_register(memory, d, b);
        let (len_read, len) = read_rv32_register(memory, d, c);
        #[cfg(debug_assertions)]
        {
            assert!(dst < (1 << self.air.ptr_max_bits));
            assert!(src < (1 << self.air.ptr_max_bits));
            assert!(len < (1 << self.air.ptr_max_bits));
        }

        let mut remaining_len = len as usize;
        let num_blocks = num_keccak_f(remaining_len);
        let mut input_blocks = Vec::with_capacity(num_blocks);
        let mut hasher = Keccak::v256();
        let mut src = src as usize;

        for block_idx in 0..num_blocks {
            if block_idx != 0 {
                memory.increment_timestamp_by(KECCAK_REGISTER_READS as u32);
                timestamp_delta += KECCAK_REGISTER_READS as u32;
            }
            let mut reads = Vec::with_capacity(KECCAK_RATE_BYTES);

            let mut partial_read_idx = None;
            let mut bytes = [0u8; KECCAK_RATE_BYTES];
            for i in (0..KECCAK_RATE_BYTES).step_by(KECCAK_WORD_SIZE) {
                if i < remaining_len {
                    let read = memory
                        .read::<u8, RV32_REGISTER_NUM_LIMBS>(e, F::from_canonical_usize(src + i));

                    let chunk = read.1;
                    let copy_len = min(KECCAK_WORD_SIZE, remaining_len - i);
                    if copy_len != KECCAK_WORD_SIZE {
                        partial_read_idx = Some(reads.len());
                    }
                    bytes[i..i + copy_len].copy_from_slice(&chunk[..copy_len]);
                    reads.push(read.0);
                } else {
                    memory.increment_timestamp();
                }
                timestamp_delta += 1;
            }

            let mut block = KeccakInputBlock {
                reads,
                partial_read_idx,
                padded_bytes: bytes,
                remaining_len,
                src,
                is_new_start: block_idx == 0,
            };
            if block_idx != num_blocks - 1 {
                src += KECCAK_RATE_BYTES;
                remaining_len -= KECCAK_RATE_BYTES;
                hasher.update(&block.padded_bytes);
            } else {
                // handle padding here since it is convenient
                debug_assert!(remaining_len < KECCAK_RATE_BYTES);
                hasher.update(&block.padded_bytes[..remaining_len]);

                if remaining_len == KECCAK_RATE_BYTES - 1 {
                    block.padded_bytes[remaining_len] = 0b1000_0001;
                } else {
                    block.padded_bytes[remaining_len] = 0x01;
                    block.padded_bytes[KECCAK_RATE_BYTES - 1] = 0x80;
                }
            }
            input_blocks.push(block);
        }
        let mut output = [0u8; 32];
        hasher.finalize(&mut output);
        let dst = dst as usize;
        let digest_writes: [_; KECCAK_DIGEST_WRITES] = from_fn(|i| {
            timestamp_delta += 1;
            memory
                .write::<u8, KECCAK_WORD_SIZE>(
                    e,
                    F::from_canonical_usize(dst + i * KECCAK_WORD_SIZE),
                    &from_fn(|j| output[i * KECCAK_WORD_SIZE + j]),
                )
                .0
        });
        tracing::trace!("[runtime] keccak256 output: {:?}", output);

        let record = KeccakRecord {
            pc: F::from_canonical_u32(from_state.pc),
            dst_read,
            src_read,
            len_read,
            input_blocks,
            digest_writes,
        };

        // Add the events to chip state for later trace generation usage
        self.records.push(record);

        // NOTE: Check this is consistent with KeccakVmAir::timestamp_change (we don't use it to
        // avoid unnecessary conversions here)
        let total_timestamp_delta =
            len + (KECCAK_REGISTER_READS + KECCAK_ABSORB_READS + KECCAK_DIGEST_WRITES) as u32;
        memory.increment_timestamp_by(total_timestamp_delta - timestamp_delta);

        Ok(ExecutionState {
            pc: from_state.pc + DEFAULT_PC_STEP,
            timestamp: from_state.timestamp + total_timestamp_delta,
        })
    }

    fn get_opcode_name(&self, _: usize) -> String {
        "KECCAK256".to_string()
    }
}

impl Default for KeccakInputBlock {
    fn default() -> Self {
        // Padding for empty byte array so padding constraints still hold
        let mut padded_bytes = [0u8; KECCAK_RATE_BYTES];
        padded_bytes[0] = 0x01;
        *padded_bytes.last_mut().unwrap() = 0x80;
        Self {
            padded_bytes,
            partial_read_idx: None,
            remaining_len: 0,
            is_new_start: true,
            reads: Vec::new(),
            src: 0,
        }
    }
}
