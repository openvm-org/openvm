//! Sha256 hasher. Handles full sha256 hashing with padding.
//! variable length inputs read from VM memory.

use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::{
    execution_mode::E1ExecutionCtx, MatrixRecordArena, NewVmChipWrapper, Result, StepExecutorE1,
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
use openvm_sha256_air::{Sha256StepHelper, SHA256_BLOCK_BITS};
use openvm_sha256_transpiler::Rv32Sha256Opcode;
use openvm_stark_backend::p3_field::PrimeField32;
use sha2::{Digest, Sha256};

mod air;
mod columns;
mod trace;

pub use air::*;
pub use columns::*;
use openvm_circuit::arch::{ExecuteFunc, ExecutionError::InvalidInstruction, VmSegmentState};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;

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

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct ShaPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<F: PrimeField32> StepExecutorE1<F> for Sha256VmStep {
    fn pre_compute_size(&self) -> usize {
        size_of::<ShaPreCompute>()
    }

    fn pre_compute_e1<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>>
    where
        Ctx: E1ExecutionCtx,
    {
        let data: &mut ShaPreCompute = data.borrow_mut();
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 != RV32_MEMORY_AS {
            return Err(InvalidInstruction(pc));
        }
        *data = ShaPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        assert_eq!(&Rv32Sha256Opcode::SHA256.global_opcode(), opcode);
        Ok(execute_impl::<_, _, true>)
    }
}

unsafe fn execute_impl<F: PrimeField32, CTX: E1ExecutionCtx, const IS_E1: bool>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &ShaPreCompute = pre_compute.borrow();
    let dst = vm_state.host_read(RV32_REGISTER_AS, pre_compute.a as u32);
    let src = vm_state.host_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let len = vm_state.host_read(RV32_REGISTER_AS, pre_compute.c as u32);
    let dst_u32 = u32::from_le_bytes(dst);
    let src_u32 = u32::from_le_bytes(src);
    let len_u32 = u32::from_le_bytes(len);

    let output = if IS_E1 {
        // SAFETY: RV32_MEMORY_AS is memory address space of type u8
        let message = vm_state.vm_read_slice(RV32_MEMORY_AS, src_u32, len_u32 as usize);
        sha256_solve(message)
    } else {
        unimplemented!()
    };
    vm_state.host_write(RV32_MEMORY_AS, dst_u32, &output);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

pub fn sha256_solve(input_message: &[u8]) -> [u8; SHA256_WRITE_SIZE] {
    let mut hasher = Sha256::new();
    hasher.update(input_message);
    let mut output = [0u8; SHA256_WRITE_SIZE];
    output.copy_from_slice(hasher.finalize().as_ref());
    output
}
