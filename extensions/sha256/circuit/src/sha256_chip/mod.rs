//! Sha256 hasher. Handles full sha256 hashing with padding.
//! variable length inputs read from VM memory.

use openvm_circuit::{
    arch::{
        execution_mode::metered::MeteredCtx, NewVmChipWrapper, Result, StepExecutorE1, VmStateMut,
    },
    system::memory::online::GuestMemory,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, encoder::Encoder,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32im_circuit::adapters::{memory_write, new_read_rv32_register};
use openvm_sha256_air::{Sha256StepHelper, SHA256_BLOCK_BITS};
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
        &mut self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
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
        let dst = new_read_rv32_register(state.memory, d, a.as_canonical_u32());
        let src = new_read_rv32_register(state.memory, d, b.as_canonical_u32());
        let len = new_read_rv32_register(state.memory, d, c.as_canonical_u32());

        debug_assert!(src + len <= (1 << self.pointer_max_bits));
        let mut hasher = Sha256::new();

        let message: Vec<u8> = state
            .memory
            .memory
            .read_range_generic((e, src), len as usize);
        hasher.update(&message);

        memory_write(state.memory, e, dst, hasher.finalize().as_ref());
        Ok(())
    }

    fn execute_metered(
        &mut self,
        _state: &mut VmStateMut<GuestMemory, MeteredCtx>,
        _instruction: &Instruction<F>,
        _chip_index: usize,
    ) -> Result<()> {
        todo!()
    }
}

pub fn sha256_solve(input_message: &[u8]) -> [u8; SHA256_WRITE_SIZE] {
    let mut hasher = Sha256::new();
    hasher.update(input_message);
    let mut output = [0u8; SHA256_WRITE_SIZE];
    output.copy_from_slice(hasher.finalize().as_ref());
    output
}
