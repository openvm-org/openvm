//! Sha256 hasher. Handles full sha256 hashing with padding.
//! variable length inputs read from VM memory.

use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        NewVmChipWrapper, Result, StepExecutorE1, VmStateMut,
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
    memory_read_from_state, memory_write_from_state, new_read_rv32_register_from_state,
};
use openvm_sha256_air::{Sha256StepHelper, SHA256_BLOCK_BITS, SHA256_ROWS_PER_BLOCK};
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

pub type Sha256VmChip<F> = NewVmChipWrapper<F, Sha256VmAir, Sha256VmStep>;

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
        state: &mut VmStateMut<GuestMemory, Ctx>,
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
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();
        let local_opcode = opcode.local_opcode_idx(self.offset);
        debug_assert_eq!(local_opcode, Rv32Sha256Opcode::SHA256.local_usize());
        debug_assert_eq!(d, RV32_REGISTER_AS);
        debug_assert_eq!(e, RV32_MEMORY_AS);
        let dst = new_read_rv32_register_from_state(state, d, a.as_canonical_u32());
        let src = new_read_rv32_register_from_state(state, d, b.as_canonical_u32());
        let len = new_read_rv32_register_from_state(state, d, c.as_canonical_u32());

        debug_assert!(src + len <= (1 << self.pointer_max_bits));
        let mut hasher = Sha256::new();

        // TODO(ayush): read in a single call
        let mut message = Vec::with_capacity(len as usize);
        for offset in (0..len as usize).step_by(SHA256_READ_SIZE) {
            let read = memory_read_from_state::<_, SHA256_READ_SIZE>(state, e, src + offset as u32);
            let copy_len = std::cmp::min(SHA256_READ_SIZE, (len as usize) - offset);
            message.extend_from_slice(&read[..copy_len]);
        }
        hasher.update(&message);

        let output = hasher.finalize();
        memory_write_from_state(state, e, dst, output.as_ref());

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;

        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();
        let local_opcode = opcode.local_opcode_idx(self.offset);

        debug_assert_eq!(local_opcode, Rv32Sha256Opcode::SHA256.local_usize());
        debug_assert_eq!(d, RV32_REGISTER_AS);
        debug_assert_eq!(e, RV32_MEMORY_AS);

        let dst = new_read_rv32_register_from_state(state, d, a.as_canonical_u32());
        let src = new_read_rv32_register_from_state(state, d, b.as_canonical_u32());
        let len = new_read_rv32_register_from_state(state, d, c.as_canonical_u32());

        debug_assert!(src + len <= (1 << self.pointer_max_bits));

        let num_blocks = ((len << 3) as usize + 1 + 64).div_ceil(SHA256_BLOCK_BITS);

        let mut message = Vec::with_capacity(len as usize);
        for offset in (0..len as usize).step_by(SHA256_READ_SIZE) {
            let read = memory_read_from_state::<_, SHA256_READ_SIZE>(state, e, src + offset as u32);
            let copy_len = std::cmp::min(SHA256_READ_SIZE, (len as usize) - offset);
            message.extend_from_slice(&read[..copy_len]);
        }

        let mut hasher = Sha256::new();
        hasher.update(&message);

        let output = hasher.finalize();
        memory_write_from_state(state, e, dst, output.as_ref());

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
