use columns::KeccakOpcodeCols;
use p3_field::PrimeField32;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub use air::KeccakVmAir;

use crate::{
    cpu::{trace::Instruction, OpCode},
    vm::ExecutionSegment,
};

/// Number of u64 elements in a Keccak hash.
pub const NUM_U64_HASH_ELEMS: usize = 4;
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
    pub requests: Vec<KeccakOpcodeCols<F>>,
    /// The input state of each keccak-f permutation corresponding to `requests`.
    /// Must have same length as `requests`.
    pub inputs: Vec<[u64; 25]>,
}

impl<F: PrimeField32> KeccakVmChip<F> {
    #[allow(clippy::new_without_default)]
    pub fn new(xor_bus_index: usize) -> Self {
        Self {
            air: KeccakVmAir::new(xor_bus_index),
            requests: Vec::new(),
            inputs: Vec::new(),
        }
    }

    /// Wrapper function for tiny-keccak's keccak-f permutation.
    /// Returns the new state after permutation.
    pub fn keccak_f(mut input: [u64; 25]) -> [u64; 25] {
        tiny_keccak::keccakf(&mut input);
        input
    }

    // TODO: only WORD_SIZE=1 works right now
    pub fn execute<const WORD_SIZE: usize>(
        vm: &mut ExecutionSegment<WORD_SIZE, F>,
        start_timestamp: usize,
        instruction: Instruction<F>,
    ) {
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
        debug_assert_eq!(op_b, F::zero());

        let mut timestamp = start_timestamp;
        let mut read = |address_space, addr| {
            let val = vm.memory_chip.read_elem(timestamp, address_space, addr);
            timestamp += 1;
            val
        };

        let dst = read(d, op_a);
        let src = read(d, op_c);

        let io = KeccakPermuteIoCols::new(
            F::from_bool(true),
            F::from_canonical_usize(start_timestamp),
            op_a,
            op_c,
            d,
            e,
        );
        let aux = KeccakPermuteAuxCols::new(dst, src);

        // TODO: unoptimized, many conversions to/from Montgomery form
        let mut offset = 0;
        let input: [[F; U64_LIMBS]; 25] = [(); 25].map(|_| {
            [(); U64_LIMBS].map(|_| {
                let val = read(e, src + F::from_canonical_usize(offset));
                offset += 1;
                val
            })
        });
        // We need to compute the output to write into memory since runtime is serial
        let input_u64: [u64; 25] = input.map(|limbs| {
            let mut val = 0u64;
            for (i, limb) in limbs.into_iter().enumerate() {
                val |= limb.as_canonical_u64() << (i * 16);
            }
            val
        });
        let output_u64 = Self::keccak_f(input_u64);
        let output: [[F; U64_LIMBS]; 25] = output_u64
            .map(|val| core::array::from_fn(|i| F::from_canonical_u64((val >> (i * 16)) & 0xFFFF)));
        debug_assert_eq!(start_timestamp + Self::write_timestamp_offset(), timestamp);
        // TODO: again very unoptimized
        let mut write = |address_space, addr, val| {
            vm.memory_chip
                .write_elem(timestamp, address_space, addr, val);
            timestamp += 1;
        };
        for (offset, output) in output.into_iter().flatten().enumerate() {
            write(e, dst + F::from_canonical_usize(offset), output);
        }

        // Add the events to chip state for later trace generation usage
        vm.keccak_permute_chip.requests.push((io, aux));
        vm.keccak_permute_chip.inputs.push(input_u64);
    }

    /// The offset from `start_timestamp` when output is written to memory
    fn write_timestamp_offset() -> usize {
        2 + U64_LIMBS * 25
    }
}
