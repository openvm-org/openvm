//! Sha256 hasher. Handles full sha256 hashing with padding.
//! variable length inputs read from VM memory.

use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::{
    execution_mode::E1ExecutionCtx, E2PreCompute, MatrixRecordArena, NewVmChipWrapper, Result,
    StepExecutorE1, StepExecutorE2,
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
pub use trace::*;

use openvm_circuit::arch::{
    execution_mode::E2ExecutionCtx, ExecuteFunc, ExecutionError::InvalidInstruction, VmSegmentState,
};
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
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_impl::<_, _>)
    }
}
impl<F: PrimeField32> StepExecutorE2<F> for Sha256VmStep {
    fn e2_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<ShaPreCompute>>()
    }

    fn pre_compute_e2<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>>
    where
        Ctx: E2ExecutionCtx,
    {
        let data: &mut E2PreCompute<ShaPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_impl::<_, _>)
    }
}

unsafe fn execute_e12_impl<F: PrimeField32, CTX: E1ExecutionCtx, const IS_E1: bool>(
    pre_compute: &ShaPreCompute,
    vm_state: &mut VmSegmentState<F, CTX>,
) -> u32 {
    let dst = vm_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32);
    let src = vm_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let len = vm_state.vm_read(RV32_REGISTER_AS, pre_compute.c as u32);
    let dst_u32 = u32::from_le_bytes(dst);
    let src_u32 = u32::from_le_bytes(src);
    let len_u32 = u32::from_le_bytes(len);

    let (output, height) = if IS_E1 {
        // SAFETY: RV32_MEMORY_AS is memory address space of type u8
        let message = vm_state.vm_read_slice(RV32_MEMORY_AS, src_u32, len_u32 as usize);
        let output = sha256_solve(message);
        (output, 0)
    } else {
        let num_blocks = get_sha256_num_blocks(len_u32);
        let mut message = Vec::with_capacity(len_u32 as usize);
        for block_idx in 0..num_blocks as usize {
            // Reads happen on the first 4 rows of each block
            for row in 0..SHA256_NUM_READ_ROWS {
                let read_idx = block_idx * SHA256_NUM_READ_ROWS + row;
                let row_input: [u8; SHA256_READ_SIZE] = vm_state.vm_read(
                    RV32_MEMORY_AS,
                    src_u32 + (read_idx * SHA256_READ_SIZE) as u32,
                );
                message.extend_from_slice(&row_input);
            }
        }
        let output = sha256_solve(&message[..len_u32 as usize]);
        let height = num_blocks * SHA256_ROWS_PER_BLOCK as u32;
        (output, height)
    };
    vm_state.vm_write(RV32_MEMORY_AS, dst_u32, &output);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;

    height
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &ShaPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, true>(pre_compute, vm_state);
}
unsafe fn execute_e2_impl<F: PrimeField32, CTX: E2ExecutionCtx>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &E2PreCompute<ShaPreCompute> = pre_compute.borrow();
    let height = execute_e12_impl::<F, CTX, false>(&pre_compute.data, vm_state);
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}

impl Sha256VmStep {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut ShaPreCompute,
    ) -> Result<()> {
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
