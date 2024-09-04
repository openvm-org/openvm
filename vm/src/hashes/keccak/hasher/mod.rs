use std::{array::from_fn, sync::Arc};

use afs_primitives::xor::lookup::XorLookupChip;
use p3_field::PrimeField32;
use tiny_keccak::{Hasher, Keccak};
use utils::num_keccak_f;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;
pub mod utils;

#[cfg(test)]
mod tests;

pub use air::KeccakVmAir;

use crate::{
    arch::{
        bus::ExecutionBus, chips::InstructionExecutor, columns::ExecutionState,
        instructions::Opcode,
    },
    cpu::trace::Instruction,
    memory::manager::{MemoryChipRef, MemoryReadRecord, MemoryWriteRecord},
};

/// Memory reads to get dst, src, len
const KECCAK_EXECUTION_READS: usize = 3;
// TODO[jpw]: adjust for batch read
/// Memory reads for absorb per row
const KECCAK_ABSORB_READS: usize = KECCAK_RATE_BYTES;
// TODO[jpw]: adjust for batch write
/// Memory writes for digest per row
const KECCAK_DIGEST_WRITES: usize = KECCAK_DIGEST_U16S;

/// Total number of sponge bytes: number of rate bytes + number of capacity
/// bytes.
pub const KECCAK_WIDTH_BYTES: usize = 200;
/// Total number of 16-bit limbs in the sponge.
pub const KECCAK_WIDTH_U16S: usize = KECCAK_WIDTH_BYTES / 2;
/// Number of non-digest bytes.
pub const KECCAK_WIDTH_MINUS_DIGEST_U16S: usize = (KECCAK_WIDTH_BYTES - KECCAK_DIGEST_BYTES) / 2;
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
/// Number of 16-bit digest limbs.
pub const KECCAK_DIGEST_U16S: usize = KECCAK_DIGEST_BYTES / 2;

#[derive(Clone, Debug)]
pub struct KeccakVmChip<F: PrimeField32> {
    pub air: KeccakVmAir,
    /// IO and memory data necessary for each opcode call
    pub records: Vec<KeccakRecord<F>>,
    pub memory_chip: MemoryChipRef<F>,
    pub byte_xor_chip: Arc<XorLookupChip<8>>,
}

#[derive(Clone, Debug)]
pub struct KeccakRecord<F> {
    pub pc: F,
    pub dst_read: MemoryReadRecord<1, F>,
    pub src_read: MemoryReadRecord<1, F>,
    pub len_read: MemoryReadRecord<1, F>,
    pub input_blocks: Vec<KeccakInputBlock<F>>,
    pub digest_writes: [MemoryWriteRecord<1, F>; KECCAK_DIGEST_WRITES],
}

#[derive(Clone, Debug)]
pub struct KeccakInputBlock<F> {
    /// Memory reads for non-padding bytes in this block. Length is at most [KECCAK_RATE_BYTES].
    pub bytes_read: Vec<MemoryReadRecord<1, F>>,
    /// Bytes with padding. Can be derived from `bytes_read` but we store for convenience.
    pub padded_bytes: [u8; KECCAK_RATE_BYTES],
    pub remaining_len: usize,
    pub is_new_start: bool,
}

impl<F: PrimeField32> KeccakVmChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        memory_chip: MemoryChipRef<F>,
        byte_xor_chip: Arc<XorLookupChip<8>>,
    ) -> Self {
        let mem_oc = memory_chip.borrow().make_offline_checker();
        Self {
            air: KeccakVmAir::new(execution_bus, mem_oc, byte_xor_chip.bus()),
            memory_chip,
            byte_xor_chip,
            records: Vec::new(),
        }
    }
}

impl<F: PrimeField32> InstructionExecutor<F> for KeccakVmChip<F> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        let Instruction {
            opcode,
            op_a: a,
            op_b: b,
            op_c: c,
            d,
            e,
            op_f: f,
            ..
        } = instruction;
        debug_assert_eq!(opcode, Opcode::KECCAK256);

        let mut memory = self.memory_chip.borrow_mut();
        debug_assert_eq!(
            from_state.timestamp,
            memory.timestamp().as_canonical_u32() as usize
        );

        let dst_read = memory.read(d, a);
        let src_read = memory.read(d, b);
        let len_read = memory.read(f, c);

        let dst = dst_read.value();
        let mut src = src_read.value();
        let len = len_read.value();
        let byte_len = len.as_canonical_u32() as usize;

        let num_blocks = num_keccak_f(byte_len);
        let mut input_blocks = Vec::with_capacity(num_blocks);
        let mut remaining_len = byte_len;
        let mut hasher = Keccak::v256();

        for block_idx in 0..num_blocks {
            if block_idx != 0 {
                memory.increment_timestamp_by(F::from_canonical_usize(KECCAK_EXECUTION_READS));
            }
            let mut bytes_read = Vec::with_capacity(KECCAK_RATE_BYTES);
            let bytes: [_; KECCAK_RATE_BYTES] = from_fn(|i| {
                if i < remaining_len {
                    let byte_read = memory.read(e, src + F::from_canonical_usize(i));
                    let byte = byte_read
                        .value()
                        .as_canonical_u32()
                        .try_into()
                        .expect("Memory cell not a byte");
                    bytes_read.push(byte_read);
                    byte
                } else {
                    memory.increment_timestamp();
                    0u8
                }
            });
            let mut block = KeccakInputBlock {
                bytes_read,
                padded_bytes: bytes,
                remaining_len,
                is_new_start: block_idx == 0,
            };
            if block_idx != num_blocks - 1 {
                src += F::from_canonical_usize(KECCAK_RATE_BYTES);
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
        let digest_writes: [_; KECCAK_DIGEST_U16S] = from_fn(|i| {
            let limb = output[2 * i] as u16 | (output[2 * i + 1] as u16) << 8;
            memory.write(
                e,
                dst + F::from_canonical_usize(i),
                [F::from_canonical_u16(limb)],
            )
        });
        tracing::trace!("[runtime] keccak256 output: {:?}", output);

        let record = KeccakRecord {
            pc: F::from_canonical_usize(from_state.pc),
            dst_read,
            src_read,
            len_read,
            input_blocks,
            digest_writes,
        };

        // Add the events to chip state for later trace generation usage
        self.records.push(record);

        let timestamp_change = KeccakVmAir::timestamp_change::<F>(len).as_canonical_u32() as usize;
        let to_timestamp = from_state.timestamp + timestamp_change;
        memory.jump_timestamp(F::from_canonical_usize(to_timestamp));

        ExecutionState {
            pc: from_state.pc + 1,
            timestamp: to_timestamp,
        }
    }
}

impl<F: PrimeField32> Default for KeccakInputBlock<F> {
    fn default() -> Self {
        // Padding for empty byte array so padding constraints still hold
        let mut padded_bytes = [0u8; KECCAK_RATE_BYTES];
        padded_bytes[0] = 0x01;
        *padded_bytes.last_mut().unwrap() = 0x80;
        Self {
            padded_bytes,
            remaining_len: 0,
            is_new_start: true,
            bytes_read: Vec::new(),
        }
    }
}

impl<F: Copy> KeccakRecord<F> {
    pub fn operands(&self) -> [F; 6] {
        let a = self.dst_read.pointer;
        let b = self.src_read.pointer;
        let c = self.len_read.pointer;
        let d = self.dst_read.address_space;
        let e = self.digest_writes[0].address_space;
        let f = self.len_read.address_space;
        [a, b, c, d, e, f]
    }

    pub fn start_timestamp(&self) -> F {
        self.dst_read.timestamp
    }

    pub fn src(&self) -> F {
        self.src_read.value()
    }

    pub fn dst(&self) -> F {
        self.dst_read.value()
    }

    pub fn len(&self) -> F {
        self.len_read.value()
    }
}
