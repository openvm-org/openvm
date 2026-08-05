use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{MEMORY_AS, REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::{
    HintStoreOpcode,
    HintStoreOpcode::{HINT_BUFFER, HINT_STORED},
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{validate_hint_buffer_num_words, HintStoreExecutor};
use crate::adapters::{
    bytes_to_u32, validate_memory_block_byte_ptr, validate_memory_block_byte_span,
};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct HintStorePreCompute {
    c: u32,
    a: u8,
    b: u8,
}

impl HintStoreExecutor {
    #[inline(always)]
    fn pre_compute_impl(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut HintStorePreCompute,
    ) -> Result<HintStoreOpcode, StaticProgramError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        if d.as_u32() != REGISTER_AS || e.as_u32() != MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = {
            HintStorePreCompute {
                c: c.as_u32(),
                a: a.as_u32() as u8,
                b: b.as_u32() as u8,
            }
        };
        Ok(HintStoreOpcode::from_usize(
            opcode.local_opcode_idx(self.offset),
        ))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        match $local_opcode {
            HINT_STORED => Ok($execute_impl::<_, true>),
            HINT_BUFFER => Ok($execute_impl::<_, false>),
        }
    };
}

impl<F> InterpreterExecutor<F> for HintStoreExecutor
where
    F: PrimeField32,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        if opcode == HINT_STORED.global_opcode().as_usize() {
            String::from("HINT_STORED")
        } else if opcode == HINT_BUFFER.global_opcode().as_usize() {
            String::from("HINT_BUFFER")
        } else {
            unreachable!("unsupported opcode: {opcode}")
        }
    }

    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<HintStorePreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError> {
        let pre_compute: &mut HintStorePreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut HintStorePreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

impl<F> InterpreterMeteredExecutor<F> for HintStoreExecutor
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<HintStorePreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<HintStorePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<HintStorePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

/// Return the number of used rows.
#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, const IS_HINT_STORED: bool>(
    pre_compute: &HintStorePreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<u32, ExecutionError> {
    let pc = exec_state.pc();
    let mem_ptr_limbs =
        exec_state.vm_read_bytes::<REGISTER_NUM_LIMBS>(REGISTER_AS, pre_compute.b as u32);
    let mem_ptr = validate_memory_block_byte_ptr(pc, bytes_to_u32(mem_ptr_limbs))?;

    let num_words = if IS_HINT_STORED {
        exec_state.ctx.advance_timestamp(1);
        1u64
    } else {
        let num_words_limbs =
            exec_state.vm_read_bytes::<REGISTER_NUM_LIMBS>(REGISTER_AS, pre_compute.a as u32);
        u64::from_le_bytes(num_words_limbs)
    };
    let num_words = u32::from(validate_hint_buffer_num_words(pc, num_words)?);
    validate_memory_block_byte_span(pc, mem_ptr, num_words as usize)?;

    let num_bytes = REGISTER_NUM_LIMBS * num_words as usize;
    if exec_state.streams.hint_stream.remaining() < num_bytes {
        let err = ExecutionError::HintOutOfBounds { pc };
        return Err(err);
    }

    for word_index in 0..num_words {
        if word_index != 0 {
            exec_state.ctx.advance_timestamp(2);
        }
        let mut data = [0; REGISTER_NUM_LIMBS];
        exec_state.streams.hint_stream.copy_to_slice(&mut data);
        exec_state.vm_write_bytes(
            MEMORY_AS,
            mem_ptr + (REGISTER_NUM_LIMBS as u32 * word_index),
            &data,
        );
    }

    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
    Ok(num_words)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, const IS_HINT_STORED: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &HintStorePreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<HintStorePreCompute>()).borrow();
    execute_e12_impl::<CTX, IS_HINT_STORED>(pre_compute, exec_state)?;
    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, const IS_HINT_STORED: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<HintStorePreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<HintStorePreCompute>>())
            .borrow();
    let height_delta = execute_e12_impl::<CTX, IS_HINT_STORED>(&pre_compute.data, exec_state)?;
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height_delta);
    Ok(())
}
