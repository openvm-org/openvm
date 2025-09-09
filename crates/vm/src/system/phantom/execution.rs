use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, PhantomDiscriminant, SysPhantom,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rand::rngs::StdRng;

#[cfg(not(feature = "tco"))]
use crate::arch::ExecuteFunc;
#[cfg(feature = "tco")]
use crate::arch::Handler;
use crate::{
    arch::{
        create_handler,
        execution_mode::{ExecutionCtxTrait, MeteredExecutionCtxTrait},
        E2PreCompute, ExecutionError, Executor, MeteredExecutor, PhantomSubExecutor,
        StaticProgramError, Streams, VmExecState,
    },
    system::{memory::online::GuestMemory, phantom::PhantomExecutor},
};

#[derive(Clone, AlignedBytesBorrow)]
#[repr(C)]
pub(super) struct PhantomOperands {
    pub(super) a: u32,
    pub(super) b: u32,
    pub(super) c: u32,
}

#[derive(Clone, AlignedBytesBorrow)]
#[repr(C)]
struct PhantomPreCompute<F> {
    operands: PhantomOperands,
    sub_executor: *const dyn PhantomSubExecutor<F>,
}

impl<F> Executor<F> for PhantomExecutor<F>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<PhantomPreCompute<F>>()
    }
    #[cfg(not(feature = "tco"))]
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
        let data: &mut PhantomPreCompute<F> = data.borrow_mut();
        self.pre_compute_impl(inst, data);
        Ok(execute_e1_handler)
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
        let data: &mut PhantomPreCompute<F> = data.borrow_mut();
        self.pre_compute_impl(inst, data);
        Ok(execute_e1_handler)
    }
}

pub(super) struct PhantomStateMut<'a, F> {
    pub(super) pc: &'a mut u32,
    pub(super) memory: &'a mut GuestMemory,
    pub(super) streams: &'a mut Streams<F>,
    pub(super) rng: &'a mut StdRng,
}

impl<F> PhantomExecutor<F>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_impl(&self, inst: &Instruction<F>, data: &mut PhantomPreCompute<F>) {
        let c = inst.c.as_canonical_u32();
        *data = PhantomPreCompute {
            operands: PhantomOperands {
                a: inst.a.as_canonical_u32(),
                b: inst.b.as_canonical_u32(),
                c,
            },
            sub_executor: self
                .phantom_executors
                .get(&PhantomDiscriminant(c as u16))
                .unwrap_or_else(|| panic!("Phantom executor not found for insn {inst:?}"))
                .as_ref(),
        };
    }
}

impl<F> MeteredExecutor<F> for PhantomExecutor<F>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<PhantomPreCompute<F>>>()
    }

    #[cfg(not(feature = "tco"))]
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
        let e2_data: &mut E2PreCompute<PhantomPreCompute<F>> = data.borrow_mut();
        e2_data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(inst, &mut e2_data.data);
        Ok(execute_e2_handler)
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
        let e2_data: &mut E2PreCompute<PhantomPreCompute<F>> = data.borrow_mut();
        e2_data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(inst, &mut e2_data.data);
        Ok(execute_e2_handler)
    }
}

#[inline(always)]
fn execute_impl<F>(
    state: PhantomStateMut<F>,
    operands: &PhantomOperands,
    sub_executor: &dyn PhantomSubExecutor<F>,
) -> Result<(), ExecutionError> {
    let &PhantomOperands { a, b, c } = operands;

    let discriminant = PhantomDiscriminant(c as u16);
    // SysPhantom::{CtStart, CtEnd} are only handled in Preflight Execution, so the only SysPhantom
    // to handle here is DebugPanic.
    if let Some(discr) = SysPhantom::from_repr(discriminant.0) {
        if discr == SysPhantom::DebugPanic {
            return Err(ExecutionError::Fail {
                pc: *state.pc,
                msg: "DebugPanic",
            });
        }
    }
    sub_executor
        .phantom_execute(
            state.memory,
            state.streams,
            state.rng,
            discriminant,
            a,
            b,
            (c >> 16) as u16,
        )
        .map_err(|e| ExecutionError::Phantom {
            pc: *state.pc,
            discriminant,
            inner: e,
        })?;

    Ok(())
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &PhantomPreCompute<F>,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let sub_executor = &*pre_compute.sub_executor;
    execute_impl(
        PhantomStateMut {
            pc,
            memory: &mut exec_state.vm_state.memory,
            streams: &mut exec_state.vm_state.streams,
            rng: &mut exec_state.vm_state.rng,
        },
        &pre_compute.operands,
        sub_executor,
    )?;
    *pc += DEFAULT_PC_STEP;
    *instret += 1;

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &PhantomPreCompute<F> = pre_compute.borrow();
    execute_e12_impl(pre_compute, instret, pc, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<PhantomPreCompute<F>> = pre_compute.borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl(&pre_compute.data, instret, pc, exec_state)
}
