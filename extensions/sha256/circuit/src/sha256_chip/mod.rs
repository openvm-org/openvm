//! Sha256 hasher. Handles full sha256 hashing with padding.
//! variable length inputs read from VM memory.

use std::cmp::min;

use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        MatrixRecordArena, NewVmChipWrapper, Result, StepExecutorE1, VmStateMut,
    },
    system::memory::online::GuestMemory,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, encoder::Encoder,
};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32im_circuit::adapters::{
    memory_read_from_state, memory_write, memory_write_from_state, read_rv32_register,
    read_rv32_register_from_state,
};
use openvm_sha256_air::{
    get_sha256_num_blocks, Sha256StepHelper, SHA256_BLOCK_BITS, SHA256_ROWS_PER_BLOCK,
};
use openvm_sha256_transpiler::Rv32Sha256Opcode;
use openvm_stark_backend::p3_field::PrimeField32;
use sha2::{Digest, Sha256};

mod air;
mod columns;
mod trace;

pub use air::*;
pub use columns::*;

#[cfg(test)]
mod tests;

// ==== Constants for register/memory adapter ====
/// Register reads to get dst, src, len
const SHA256_REGISTER_READS: usize = 3;
/// Number of cells to read in a single memory access
const SHA256_READ_SIZE: usize = 16;
/// Number of cells to write in a single memory access
const SHA256_WRITE_SIZE: usize = 32;
/// Number of rv32 cells read in a SHA256 block
pub const SHA256_BLOCK_CELLS: usize = SHA256_BLOCK_BITS / RV32_CELL_BITS;
/// Number of rows we will do a read on for each SHA256 block
pub const SHA256_NUM_READ_ROWS: usize = SHA256_BLOCK_CELLS / SHA256_READ_SIZE;
/// Maximum message length that this chip supports in bytes
pub const SHA256_MAX_MESSAGE_LEN: usize = 1 << 29;

pub type Sha256VmChip<F> = NewVmChipWrapper<F, Sha256VmAir, Sha256VmStep, MatrixRecordArena<F>>;

pub struct Sha256VmStep {
    pub inner: Sha256StepHelper,
    pub padding_encoder: Encoder,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pub offset: usize,
    pub pointer_max_bits: usize,
}

impl Sha256VmStep {
    pub fn new(
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        offset: usize,
        pointer_max_bits: usize,
    ) -> Self {
        Self {
            inner: Sha256StepHelper::new(),
            padding_encoder: Encoder::new(PaddingFlags::COUNT, 2, false),
            bitwise_lookup_chip,
            offset,
            pointer_max_bits,
        }
    }
}

impl<F: PrimeField32> StepExecutorE1<F> for Sha256VmStep {
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
        let local_opcode = opcode.local_opcode_idx(self.offset);
        debug_assert_eq!(local_opcode, Rv32Sha256Opcode::SHA256.local_usize());
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);
        let dst = read_rv32_register(state.memory, a.as_canonical_u32());
        let src = read_rv32_register(state.memory, b.as_canonical_u32());
        let len = read_rv32_register(state.memory, c.as_canonical_u32());

        debug_assert!(src + len <= (1 << self.pointer_max_bits));
        debug_assert!(dst < (1 << self.pointer_max_bits));

        let message = unsafe {
            state
                .memory
                .memory
                .get_slice::<u8>((RV32_MEMORY_AS, src), len as usize)
        };

        let output = sha256_solve(&message);
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
        debug_assert_eq!(*opcode, Rv32Sha256Opcode::SHA256.global_opcode());
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        let dst = read_rv32_register_from_state(state, a.as_canonical_u32());
        let src = read_rv32_register_from_state(state, b.as_canonical_u32());
        let len = read_rv32_register_from_state(state, c.as_canonical_u32());

        let num_blocks = get_sha256_num_blocks(len) as usize;

        // we will read [num_blocks] * [SHA256_BLOCK_CELLS] cells but only [len] cells will be used
        debug_assert!(
            src as usize + num_blocks * SHA256_BLOCK_CELLS <= (1 << self.pointer_max_bits)
        );
        debug_assert!(dst as usize + SHA256_WRITE_SIZE <= (1 << self.pointer_max_bits));
        // We don't support messages longer than 2^29 bytes
        debug_assert!(len < SHA256_MAX_MESSAGE_LEN as u32);

        let mut input = Vec::with_capacity(len as usize);
        for idx in 0..num_blocks * SHA256_NUM_READ_ROWS {
            let read: [u8; SHA256_READ_SIZE] = memory_read_from_state(
                state,
                RV32_MEMORY_AS,
                src + (idx * SHA256_READ_SIZE) as u32,
            );
            let offset = idx * SHA256_READ_SIZE;
            if offset < len as usize {
                let copy_len = min(len as usize - offset, SHA256_READ_SIZE);
                input.extend_from_slice(&read[..copy_len]);
            }
        }

        let output = sha256_solve(&input);
        memory_write_from_state(state, RV32_MEMORY_AS, dst, &output);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        state.ctx.trace_heights[chip_index] += (num_blocks * SHA256_ROWS_PER_BLOCK) as u32;
        Ok(())
    }
}

pub fn sha256_solve(input_message: &[u8]) -> [u8; SHA256_WRITE_SIZE] {
    let mut hasher = Sha256::new();
    hasher.update(input_message);
    let mut output = [0u8; SHA256_WRITE_SIZE];
    output.copy_from_slice(hasher.finalize().as_ref());
    output
}
