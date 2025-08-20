use std::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_IMM_AS, NATIVE_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::PublicValuesExecutor;
#[cfg(feature = "tco")]
use crate::arch::Handler;
use crate::{
    arch::{
        create_tco_handler,
        execution_mode::{ExecutionCtxTrait, MeteredExecutionCtxTrait},
        E2PreCompute, ExecuteFunc, Executor, MeteredExecutor, StaticProgramError, VmExecState,
    },
    system::memory::online::GuestMemory,
    utils::{transmute_field_to_u32, transmute_u32_to_field},
};

#[derive(AlignedBytesBorrow)]
#[repr(C)]
struct PublicValuesPreCompute {
    b_or_imm: u32,
    c_or_imm: u32,
}

impl<F, A> PublicValuesExecutor<F, A>
where
    F: PrimeField32,
{
    fn pre_compute_impl(
        &self,
        inst: &Instruction<F>,
        data: &mut PublicValuesPreCompute,
    ) -> (bool, bool) {
        let &Instruction { b, c, e, f, .. } = inst;

        let e = e.as_canonical_u32();
        let f = f.as_canonical_u32();

        let b_is_imm = e == RV32_IMM_AS;
        let c_is_imm = f == RV32_IMM_AS;

        let b_or_imm = if b_is_imm {
            transmute_field_to_u32(&b)
        } else {
            b.as_canonical_u32()
        };
        let c_or_imm = if c_is_imm {
            transmute_field_to_u32(&c)
        } else {
            c.as_canonical_u32()
        };

        *data = PublicValuesPreCompute { b_or_imm, c_or_imm };

        (b_is_imm, c_is_imm)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $b_is_imm:ident, $c_is_imm:ident) => {
        match ($b_is_imm, $c_is_imm) {
            (true, true) => Ok($execute_impl::<_, _, true, true>),
            (true, false) => Ok($execute_impl::<_, _, true, false>),
            (false, true) => Ok($execute_impl::<_, _, false, true>),
            (false, false) => Ok($execute_impl::<_, _, false, false>),
        }
    };
}

impl<F, A> Executor<F> for PublicValuesExecutor<F, A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<PublicValuesPreCompute>()
    }

    #[inline(always)]
    fn pre_compute<Ctx>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut PublicValuesPreCompute = data.borrow_mut();
        let (b_is_imm, c_is_imm) = self.pre_compute_impl(inst, data);

        dispatch!(execute_e1_impl, b_is_imm, c_is_imm)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut PublicValuesPreCompute = data.borrow_mut();
        let (b_is_imm, c_is_imm) = self.pre_compute_impl(inst, data);

        dispatch!(execute_e1_tco_handler, b_is_imm, c_is_imm)
    }
}

impl<F, A> MeteredExecutor<F> for PublicValuesExecutor<F, A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<PublicValuesPreCompute>>()
    }

    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<PublicValuesPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (b_is_imm, c_is_imm) = self.pre_compute_impl(inst, &mut data.data);

        dispatch!(execute_e2_impl, b_is_imm, c_is_imm)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<PublicValuesPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (b_is_imm, c_is_imm) = self.pre_compute_impl(inst, &mut data.data);

        dispatch!(execute_e2_tco_handler, b_is_imm, c_is_imm)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX, const B_IS_IMM: bool, const C_IS_IMM: bool>(
    pre_compute: &PublicValuesPreCompute,
    state: &mut VmExecState<F, GuestMemory, CTX>,
) where
    CTX: ExecutionCtxTrait,
{
    let value = if B_IS_IMM {
        transmute_u32_to_field(&pre_compute.b_or_imm)
    } else {
        state.vm_read::<F, 1>(NATIVE_AS, pre_compute.b_or_imm)[0]
    };
    let index = if C_IS_IMM {
        transmute_u32_to_field(&pre_compute.c_or_imm)
    } else {
        state.vm_read::<F, 1>(NATIVE_AS, pre_compute.c_or_imm)[0]
    };

    let idx: usize = index.as_canonical_u32() as usize;
    {
        let custom_pvs = &mut state.vm_state.custom_pvs;

        if custom_pvs[idx].is_none() {
            custom_pvs[idx] = Some(value);
        } else {
            // Not a hard constraint violation when publishing the same value twice but the
            // program should avoid that.
            panic!("Custom public value {} already set", idx);
        }
    }
    state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
    state.instret += 1;
}

#[create_tco_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX, const B_IS_IMM: bool, const C_IS_IMM: bool>(
    pre_compute: &[u8],
    state: &mut VmExecState<F, GuestMemory, CTX>,
) where
    CTX: ExecutionCtxTrait,
{
    let pre_compute: &PublicValuesPreCompute = pre_compute.borrow();
    execute_e12_impl::<_, _, B_IS_IMM, C_IS_IMM>(pre_compute, state);
}

#[create_tco_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX, const B_IS_IMM: bool, const C_IS_IMM: bool>(
    pre_compute: &[u8],
    state: &mut VmExecState<F, GuestMemory, CTX>,
) where
    CTX: MeteredExecutionCtxTrait,
{
    let pre_compute: &E2PreCompute<PublicValuesPreCompute> = pre_compute.borrow();
    state.ctx.on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<_, _, B_IS_IMM, C_IS_IMM>(&pre_compute.data, state);
}
