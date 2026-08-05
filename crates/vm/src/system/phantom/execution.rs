use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::{Instruction, InstructionOperand},
    program::DEFAULT_PC_STEP,
    PhantomDiscriminant, SysPhantom, SystemOpcode,
};
use rand::rngs::StdRng;

#[cfg(not(feature = "tco"))]
use crate::arch::ExecuteFunc;
#[cfg(feature = "tco")]
use crate::arch::Handler;
use crate::{
    arch::{
        create_handler,
        execution_mode::{ExecutionCtxTrait, MeteredExecutionCtxTrait},
        E2PreCompute, ExecutionError, InterpreterExecutor, InterpreterMeteredExecutor,
        PhantomSubExecutor, StaticProgramError, Streams, VmExecState,
    },
    system::{memory::online::GuestMemory, phantom::PhantomExecutor},
};

#[derive(Clone, AlignedBytesBorrow)]
#[repr(C)]
pub(super) struct PhantomOperands {
    pub(super) a: u32,
    pub(super) b: u32,
    pub(super) c: u32,
    pub(super) d: u32,
}

#[derive(Clone, AlignedBytesBorrow)]
#[repr(C)]
struct PhantomPreCompute {
    operands: PhantomOperands,
    sub_executor: *const dyn PhantomSubExecutor,
}

impl InterpreterExecutor for PhantomExecutor {
    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", SystemOpcode::PHANTOM)
    }

    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<PhantomPreCompute>()
    }
    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut PhantomPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler::<_>)
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
        let data: &mut PhantomPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler::<_>)
    }
}

pub(super) struct PhantomStateMut<'a> {
    pub(super) pc: u32,
    pub(super) memory: &'a mut GuestMemory,
    pub(super) streams: &'a mut Streams,
    pub(super) rng: &'a mut StdRng,
}

impl PhantomExecutor {
    #[inline(always)]
    fn pre_compute_impl(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut PhantomPreCompute,
    ) -> Result<(), StaticProgramError> {
        if [inst.e, inst.f, inst.g]
            .into_iter()
            .any(|operand| !operand.is_zero())
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let to_u32 = |operand: InstructionOperand| {
            operand
                .checked_as_u32()
                .ok_or(StaticProgramError::InvalidInstruction(pc))
        };
        let a = to_u32(inst.a)?;
        let b = to_u32(inst.b)?;
        let c = to_u32(inst.c)?;
        let d = to_u32(inst.d)?;
        let discriminant = u16::try_from(c)
            .map(PhantomDiscriminant)
            .map_err(|_| StaticProgramError::InvalidInstruction(pc))?;
        let _c_upper = u16::try_from(d).map_err(|_| StaticProgramError::InvalidInstruction(pc))?;
        let sub_executor = self
            .phantom_executors
            .get(&discriminant)
            .ok_or(StaticProgramError::InvalidInstruction(pc))?
            .as_ref();
        *data = PhantomPreCompute {
            operands: PhantomOperands { a, b, c, d },
            sub_executor,
        };
        Ok(())
    }
}

impl InterpreterMeteredExecutor for PhantomExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<PhantomPreCompute>>()
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
        let e2_data: &mut E2PreCompute<PhantomPreCompute> = data.borrow_mut();
        e2_data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut e2_data.data)?;
        Ok(execute_e2_handler::<_>)
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
        let e2_data: &mut E2PreCompute<PhantomPreCompute> = data.borrow_mut();
        e2_data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut e2_data.data)?;
        Ok(execute_e2_handler::<_>)
    }
}

#[inline(always)]
fn execute_impl(
    state: PhantomStateMut,
    operands: &PhantomOperands,
    sub_executor: &dyn PhantomSubExecutor,
) -> Result<(), ExecutionError> {
    let &PhantomOperands { a, b, c, d } = operands;

    let discriminant = PhantomDiscriminant(c as u16);
    // SysPhantom::{CtStart, CtEnd} are only handled in Preflight Execution, so the only SysPhantom
    // to handle here is DebugPanic.
    if let Some(discr) = SysPhantom::from_repr(discriminant.0) {
        if discr == SysPhantom::DebugPanic {
            return Err(ExecutionError::Fail {
                pc: state.pc,
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
            d as u16,
        )
        .map_err(|e| ExecutionError::Phantom {
            pc: state.pc,
            discriminant,
            inner: e,
        })?;

    Ok(())
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait>(
    pre_compute: &PhantomPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let sub_executor = &*pre_compute.sub_executor;
    let pc = exec_state.pc();
    let discriminant = PhantomDiscriminant(pre_compute.operands.c as u16);
    if let Some(phantom) = SysPhantom::from_repr(discriminant.0) {
        CTX::on_system_phantom(exec_state, pc, phantom);
    }
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
    exec_state.ctx.advance_timestamp(1);
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &PhantomPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<PhantomPreCompute>()).borrow();
    execute_e12_impl(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<PhantomPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<PhantomPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl(&pre_compute.data, exec_state)
}
