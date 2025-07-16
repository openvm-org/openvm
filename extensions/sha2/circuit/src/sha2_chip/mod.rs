//! Sha256 hasher. Handles full sha256 hashing with padding.
//! variable length inputs read from VM memory.
use std::{
    borrow::{Borrow, BorrowMut},
    iter,
};

use openvm_circuit::arch::{
    execution_mode::{E1ExecutionCtx, E2ExecutionCtx},
    E2PreCompute, ExecuteFunc,
    ExecutionError::InvalidInstruction,
    MatrixRecordArena, NewVmChipWrapper, Result, StepExecutorE1, StepExecutorE2, VmSegmentState,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, encoder::Encoder,
};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_sha2_air::{Sha256Config, Sha2StepHelper, Sha2Variant, Sha384Config, Sha512Config};
use openvm_stark_backend::p3_field::PrimeField32;
use sha2::{Digest, Sha256, Sha384, Sha512};

mod air;
mod columns;
mod config;
mod trace;
mod utils;

pub use air::*;
pub use columns::*;
pub use config::*;
pub use utils::get_sha2_num_blocks;

#[cfg(test)]
mod tests;

pub type Sha2VmChip<F, C> = NewVmChipWrapper<F, Sha2VmAir<C>, Sha2VmStep<C>, MatrixRecordArena<F>>;

pub struct Sha2VmStep<C: Sha2ChipConfig> {
    pub inner: Sha2StepHelper<C>,
    pub padding_encoder: Encoder,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pub offset: usize,
    pub pointer_max_bits: usize,
}

impl<C: Sha2ChipConfig> Sha2VmStep<C> {
    pub fn new(
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        offset: usize,
        pointer_max_bits: usize,
    ) -> Self {
        Self {
            inner: Sha2StepHelper::<C>::new(),
            padding_encoder: Encoder::new(PaddingFlags::COUNT, 2, false),
            bitwise_lookup_chip,
            offset,
            pointer_max_bits,
        }
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct Sha2PreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<F: PrimeField32, C: Sha2ChipConfig> StepExecutorE1<F> for Sha2VmStep<C> {
    fn pre_compute_size(&self) -> usize {
        size_of::<Sha2PreCompute>()
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
        let data: &mut Sha2PreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_impl::<_, _, C>)
    }
}
impl<F: PrimeField32, C: Sha2ChipConfig> StepExecutorE2<F> for Sha2VmStep<C> {
    fn e2_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Sha2PreCompute>>()
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
        let data: &mut E2PreCompute<Sha2PreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_impl::<_, _, C>)
    }
}

unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    C: Sha2ChipConfig,
    const IS_E1: bool,
>(
    pre_compute: &Sha2PreCompute,
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
        let output = sha2_solve::<C>(message);
        (output, 0)
    } else {
        let num_blocks = get_sha2_num_blocks::<C>(len_u32);
        let mut message = Vec::with_capacity(len_u32 as usize);
        for block_idx in 0..num_blocks as usize {
            // Reads happen on the first 4 rows of each block
            for row in 0..C::NUM_READ_ROWS {
                let read_idx = block_idx * C::NUM_READ_ROWS + row;
                match C::VARIANT {
                    Sha2Variant::Sha256 => {
                        let row_input: [u8; Sha256Config::READ_SIZE] = vm_state
                            .vm_read(RV32_MEMORY_AS, src_u32 + (read_idx * C::READ_SIZE) as u32);
                        message.extend_from_slice(&row_input);
                    }
                    Sha2Variant::Sha512 => {
                        let row_input: [u8; Sha512Config::READ_SIZE] = vm_state
                            .vm_read(RV32_MEMORY_AS, src_u32 + (read_idx * C::READ_SIZE) as u32);
                        message.extend_from_slice(&row_input);
                    }
                    Sha2Variant::Sha384 => {
                        let row_input: [u8; Sha384Config::READ_SIZE] = vm_state
                            .vm_read(RV32_MEMORY_AS, src_u32 + (read_idx * C::READ_SIZE) as u32);
                        message.extend_from_slice(&row_input);
                    }
                }
            }
        }
        let output = sha2_solve::<C>(&message[..len_u32 as usize]);
        let height = num_blocks * C::ROWS_PER_BLOCK as u32;
        (output, height)
    };
    match C::VARIANT {
        Sha2Variant::Sha256 => {
            let output: [u8; Sha256Config::WRITE_SIZE] = output.try_into().unwrap();
            vm_state.vm_write(RV32_MEMORY_AS, dst_u32, &output);
        }
        Sha2Variant::Sha512 => {
            for i in 0..C::NUM_WRITES {
                let output: [u8; Sha512Config::WRITE_SIZE] = output
                    [i * Sha512Config::WRITE_SIZE..(i + 1) * Sha512Config::WRITE_SIZE]
                    .try_into()
                    .unwrap();
                vm_state.vm_write(
                    RV32_MEMORY_AS,
                    dst_u32 + (i * Sha512Config::WRITE_SIZE) as u32,
                    &output,
                );
            }
        }
        Sha2Variant::Sha384 => {
            // Pad the output with zeros to 64 bytes
            let output = output
                .into_iter()
                .chain(iter::repeat(0).take(16))
                .collect::<Vec<_>>();
            for i in 0..C::NUM_WRITES {
                let output: [u8; Sha384Config::WRITE_SIZE] = output
                    [i * Sha384Config::WRITE_SIZE..(i + 1) * Sha384Config::WRITE_SIZE]
                    .try_into()
                    .unwrap();
                vm_state.vm_write(
                    RV32_MEMORY_AS,
                    dst_u32 + (i * Sha384Config::WRITE_SIZE) as u32,
                    &output,
                );
            }
        }
    }

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;

    height
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx, C: Sha2ChipConfig>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &Sha2PreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, C, true>(pre_compute, vm_state);
}
unsafe fn execute_e2_impl<F: PrimeField32, CTX: E2ExecutionCtx, C: Sha2ChipConfig>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &E2PreCompute<Sha2PreCompute> = pre_compute.borrow();
    let height = execute_e12_impl::<F, CTX, C, false>(&pre_compute.data, vm_state);
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}

impl<C: Sha2ChipConfig> Sha2VmStep<C> {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Sha2PreCompute,
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
        *data = Sha2PreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        assert_eq!(&C::OPCODE.global_opcode(), opcode);
        Ok(())
    }
}

pub fn sha2_solve<C: Sha2ChipConfig>(input_message: &[u8]) -> Vec<u8> {
    match C::VARIANT {
        Sha2Variant::Sha256 => {
            let mut hasher = Sha256::new();
            hasher.update(input_message);
            let mut output = vec![0u8; C::DIGEST_SIZE];
            output.copy_from_slice(hasher.finalize().as_ref());
            output
        }
        Sha2Variant::Sha512 => {
            let mut hasher = Sha512::new();
            hasher.update(input_message);
            let mut output = vec![0u8; C::DIGEST_SIZE];
            output.copy_from_slice(hasher.finalize().as_ref());
            output
        }
        Sha2Variant::Sha384 => {
            let mut hasher = Sha384::new();
            hasher.update(input_message);
            let mut output = vec![0u8; C::DIGEST_SIZE];
            output.copy_from_slice(hasher.finalize().as_ref());
            output
        }
    }
}
