use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::{
    Rv32HintStoreOpcode,
    Rv32HintStoreOpcode::{HINT_BUFFER, HINT_STOREW},
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::Rv32HintStoreExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct HintStorePreCompute {
    c: u32,
    a: u8,
    b: u8,
}

impl Rv32HintStoreExecutor {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut HintStorePreCompute,
    ) -> Result<Rv32HintStoreOpcode, StaticProgramError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        if d.as_canonical_u32() != RV32_REGISTER_AS || e.as_canonical_u32() != RV32_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = {
            HintStorePreCompute {
                c: c.as_canonical_u32(),
                a: a.as_canonical_u32() as u8,
                b: b.as_canonical_u32() as u8,
            }
        };
        Ok(Rv32HintStoreOpcode::from_usize(
            opcode.local_opcode_idx(self.offset),
        ))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        match $local_opcode {
            HINT_STOREW => Ok($execute_impl::<_, _, true>),
            HINT_BUFFER => Ok($execute_impl::<_, _, false>),
        }
    };
}

impl<F> Executor<F> for Rv32HintStoreExecutor
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<HintStorePreCompute>()
    }

    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut HintStorePreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_impl, local_opcode)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut HintStorePreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_tco_handler, local_opcode)
    }
}

impl<F> MeteredExecutor<F> for Rv32HintStoreExecutor
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<HintStorePreCompute>>()
    }

    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<HintStorePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_impl, local_opcode)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<HintStorePreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_tco_handler, local_opcode)
    }
}

/// Return the number of used rows.
#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_HINT_STOREW: bool>(
    pre_compute: &HintStorePreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32 {
    let mem_ptr_limbs = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let mem_ptr = u32::from_le_bytes(mem_ptr_limbs);

    let num_words = if IS_HINT_STOREW {
        1
    } else {
        let num_words_limbs = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32);
        u32::from_le_bytes(num_words_limbs)
    };
    debug_assert_ne!(num_words, 0);

    if vm_state.streams.hint_stream.len() < RV32_REGISTER_NUM_LIMBS * num_words as usize {
        vm_state.exit_code = Err(ExecutionError::HintOutOfBounds { pc: vm_state.pc });
        return 0;
    }

    for word_index in 0..num_words {
        let data: [u8; RV32_REGISTER_NUM_LIMBS] = std::array::from_fn(|_| {
            vm_state
                .streams
                .hint_stream
                .pop_front()
                .unwrap()
                .as_canonical_u32() as u8
        });
        vm_state.vm_write(
            RV32_MEMORY_AS,
            mem_ptr + (RV32_REGISTER_NUM_LIMBS as u32 * word_index),
            &data,
        );
    }

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
    num_words
}

#[create_tco_handler]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_HINT_STOREW: bool>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &HintStorePreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, IS_HINT_STOREW>(pre_compute, vm_state);
}

#[create_tco_handler]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_HINT_STOREW: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<HintStorePreCompute> = pre_compute.borrow();
    let height_delta = execute_e12_impl::<F, CTX, IS_HINT_STOREW>(&pre_compute.data, vm_state);
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height_delta);
}
