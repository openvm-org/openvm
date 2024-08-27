use std::{array::from_fn, sync::Arc};

use afs_primitives::xor_lookup::XorLookupChip;
use bridge::{BLOCK_MEMORY_ACCESSES, TIMESTAMP_OFFSET_FOR_OPCODE};
use columns::KeccakOpcodeCols;
use p3_field::PrimeField32;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;
pub mod utils;

pub use air::KeccakVmAir;
use tiny_keccak::{Hasher, Keccak};
use utils::num_keccak_f;

use crate::vm::ExecutionSegment;

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
    pub byte_xor_chip: Arc<XorLookupChip<8>>,
    /// IO and memory data necessary for each opcode call
    pub requests: Vec<(KeccakOpcodeCols<F>, Vec<KeccakInputBlock<F>>)>,
}

#[derive(Clone, Debug)]
pub struct KeccakInputBlock<F: PrimeField32> {
    pub padded_bytes: [u8; KECCAK_RATE_BYTES],
    pub remaining_len: usize,
    pub is_new_start: bool,
    pub src: F,
}

impl<F: PrimeField32> KeccakVmChip<F> {
    #[allow(clippy::new_without_default)]
    pub fn new(xor_bus_index: usize, byte_xor_chip: Arc<XorLookupChip<8>>) -> Self {
        Self {
            air: KeccakVmAir::new(xor_bus_index),
            byte_xor_chip,
            requests: Vec::new(),
        }
    }

    /// Returns the new timestamp after instruction has finished execution
    // TODO: only WORD_SIZE=1 works right now
    pub fn execute<const WORD_SIZE: usize>(
        vm: &mut ExecutionSegment<WORD_SIZE, F>,
        start_timestamp: usize,
        instruction: Instruction<F>,
    ) -> usize {
        assert_eq!(WORD_SIZE, 1, "Only WORD_SIZE=1 supported for now");
        let Instruction {
            opcode,
            op_a,
            op_b,
            op_c,
            d,
            e,
            debug: _debug,
        } = instruction;
        debug_assert_eq!(opcode, OpCode::KECCAK256);

        let mut timestamp = start_timestamp;

        let dst = vm.memory_chip.read_elem(timestamp, d, op_a);
        let mut src = vm.memory_chip.read_elem(timestamp + 1, d, op_b);
        let len = vm.memory_chip.read_elem(timestamp + 2, d, op_c);

        let opcode = KeccakOpcodeCols::new(
            F::from_bool(true),
            F::from_canonical_usize(start_timestamp),
            op_a,
            op_b,
            op_c,
            d,
            e,
            dst,
            src,
            len,
        );
        let byte_len = len.as_canonical_u32() as usize;

        let num_blocks = num_keccak_f(byte_len);
        let mut input = Vec::with_capacity(num_blocks);
        let mut remaining_len = byte_len;
        let mut hasher = Keccak::v256();

        for block_idx in 0..num_blocks {
            let bytes: [_; KECCAK_RATE_BYTES] = from_fn(|i| {
                if i < remaining_len {
                    vm.memory_chip
                        .read_elem(
                            timestamp + TIMESTAMP_OFFSET_FOR_OPCODE + i,
                            e,
                            src + F::from_canonical_usize(i),
                        )
                        .as_canonical_u32() as u8
                } else {
                    0u8
                }
            });
            let mut block = KeccakInputBlock {
                padded_bytes: bytes,
                remaining_len,
                is_new_start: block_idx == 0,
                src,
            };
            timestamp += TIMESTAMP_OFFSET_FOR_OPCODE + BLOCK_MEMORY_ACCESSES;
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
            input.push(block);
        }
        let mut output = [0u8; 32];
        hasher.finalize(&mut output);
        for i in 0..KECCAK_DIGEST_U16S {
            let limb = output[2 * i] as u16 | (output[2 * i + 1] as u16) << 8;
            vm.memory_chip.write_elem(
                timestamp + i,
                e,
                dst + F::from_canonical_usize(i),
                F::from_canonical_u16(limb),
            );
        }
        tracing::trace!("[runtime] keccak256 output: {:?}", output);

        // Add the events to chip state for later trace generation usage
        vm.keccak_chip.requests.push((opcode, input));

        timestamp + KECCAK_DIGEST_U16S
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
            src: F::zero(),
        }
    }
}
