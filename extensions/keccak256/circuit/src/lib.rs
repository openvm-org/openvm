//! Stateful keccak256 hasher. Handles full keccak sponge (padding, absorb, keccak-f) on
//! variable length inputs read from VM memory.

use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;
use openvm_stark_backend::p3_field::PrimeField32;
use p3_keccak_air::NUM_ROUNDS;

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
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        ExecutionBridge, MatrixRecordArena, NewVmChipWrapper, Result, StepExecutorE1, VmStateMut,
    },
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_keccak256_transpiler::Rv32KeccakOpcode;
use openvm_rv32im_circuit::adapters::{
    memory_read_from_state, memory_write, memory_write_from_state, read_rv32_register_from_state,
};
use utils::num_keccak_f;

use crate::utils::keccak256;

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

pub type KeccakVmChip<F> = NewVmChipWrapper<F, KeccakVmAir, KeccakVmStep, MatrixRecordArena<F>>;

//#[derive(derive_new::new)]
pub struct KeccakVmStep {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub offset: usize,
    pub pointer_max_bits: usize,
}

impl KeccakVmStep {
    pub fn new(
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
        offset: usize,
        pointer_max_bits: usize,
    ) -> Self {
        Self {
            bitwise_lookup_chip,
            offset,
            pointer_max_bits,
        }
    }
}

impl<F: PrimeField32> StepExecutorE1<F> for KeccakVmStep {
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;

        debug_assert_eq!(opcode, Rv32KeccakOpcode::KECCAK256.global_opcode());
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        let dst = read_rv32_register_from_state(state, a.as_canonical_u32());
        let src = read_rv32_register_from_state(state, b.as_canonical_u32());
        let len = read_rv32_register_from_state(state, c.as_canonical_u32());

        let message = unsafe {
            state
                .memory
                .memory
                .get_slice((RV32_MEMORY_AS, src), len as usize)
        };

        let output = keccak256(&message);
        memory_write(state.memory, RV32_MEMORY_AS, dst, &output);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<F, GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;
        debug_assert_eq!(*opcode, Rv32KeccakOpcode::KECCAK256.global_opcode());
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        let dst = read_rv32_register_from_state(state, a.as_canonical_u32());
        let src = read_rv32_register_from_state(state, b.as_canonical_u32());
        let len = read_rv32_register_from_state(state, c.as_canonical_u32()) as usize;

        let num_blocks = num_keccak_f(len);

        debug_assert!(src as usize + len <= (1 << self.pointer_max_bits));
        debug_assert!(dst as usize + KECCAK_DIGEST_BYTES <= (1 << self.pointer_max_bits));
        // We don't support messages longer than 2^[pointer_max_bits] bytes
        debug_assert!(len < (1 << self.pointer_max_bits));

        let mut input = Vec::with_capacity(len);
        let num_reads = len.div_ceil(KECCAK_WORD_SIZE);
        for idx in 0..num_reads {
            let read: [u8; KECCAK_WORD_SIZE] = memory_read_from_state(
                state,
                RV32_MEMORY_AS,
                src + (idx * KECCAK_WORD_SIZE) as u32,
            );
            if idx < num_reads - 1 {
                input.extend_from_slice(&read);
            } else {
                let remaining_bytes = len - idx * KECCAK_WORD_SIZE;
                input.extend_from_slice(&read[..remaining_bytes]);
            }
        }

        let output = keccak256(&input);
        for (i, word) in output.chunks_exact(KECCAK_WORD_SIZE).enumerate() {
            memory_write_from_state::<F, _, KECCAK_WORD_SIZE>(
                state,
                RV32_MEMORY_AS,
                dst + (i * KECCAK_WORD_SIZE) as u32,
                word.try_into().unwrap(),
            );
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        state.ctx.trace_heights[chip_index] += (num_blocks * NUM_ROUNDS) as u32;

        Ok(())
    }
}
