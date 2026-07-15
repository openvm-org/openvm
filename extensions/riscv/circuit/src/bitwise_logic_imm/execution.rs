use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_IMM_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::BitwiseImmOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::BitwiseLogicImmExecutor;
use crate::adapters::{imm_to_rv64_u64, is_canonical_i12};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct BitwiseLogicImmPreCompute {
    imm: u64,
    rd_ptr: u8,
    rs1_ptr: u8,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    BitwiseLogicImmExecutor<A, NUM_LIMBS, LIMB_BITS>
{
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut BitwiseLogicImmPreCompute,
    ) -> Result<BitwiseImmOpcode, StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let c = c.as_canonical_u32();
        if d.as_canonical_u32() != RV64_REGISTER_AS
            || e.as_canonical_u32() != RV64_IMM_AS
            || !is_canonical_i12(c)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = BitwiseLogicImmPreCompute {
            imm: imm_to_rv64_u64(c),
            rd_ptr: a.as_canonical_u32() as u8,
            rs1_ptr: b.as_canonical_u32() as u8,
        };
        Ok(BitwiseImmOpcode::from_usize(
            opcode.local_opcode_idx(self.offset),
        ))
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for BitwiseLogicImmExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<BitwiseLogicImmPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut BitwiseLogicImmPreCompute = data.borrow_mut();
        let opcode = self.pre_compute_impl(pc, inst, data)?;
        Ok(match opcode {
            BitwiseImmOpcode::XORI => execute_e1_handler::<Ctx, XorOp>,
            BitwiseImmOpcode::ORI => execute_e1_handler::<Ctx, OrOp>,
            BitwiseImmOpcode::ANDI => execute_e1_handler::<Ctx, AndOp>,
        })
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut BitwiseLogicImmPreCompute = data.borrow_mut();
        let opcode = self.pre_compute_impl(pc, inst, data)?;
        Ok(match opcode {
            BitwiseImmOpcode::XORI => execute_e1_handler::<Ctx, XorOp>,
            BitwiseImmOpcode::ORI => execute_e1_handler::<Ctx, OrOp>,
            BitwiseImmOpcode::ANDI => execute_e1_handler::<Ctx, AndOp>,
        })
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for BitwiseLogicImmExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BitwiseLogicImmPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<BitwiseLogicImmPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(match opcode {
            BitwiseImmOpcode::XORI => execute_e2_handler::<Ctx, XorOp>,
            BitwiseImmOpcode::ORI => execute_e2_handler::<Ctx, OrOp>,
            BitwiseImmOpcode::ANDI => execute_e2_handler::<Ctx, AndOp>,
        })
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<BitwiseLogicImmPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(match opcode {
            BitwiseImmOpcode::XORI => execute_e2_handler::<Ctx, XorOp>,
            BitwiseImmOpcode::ORI => execute_e2_handler::<Ctx, OrOp>,
            BitwiseImmOpcode::ANDI => execute_e2_handler::<Ctx, AndOp>,
        })
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, OP: ImmOp>(
    pre_compute: &BitwiseLogicImmPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let rs1 = exec_state
        .vm_read_bytes::<RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.rs1_ptr as u32);
    let rs1 = u64::from_le_bytes(rs1);
    let rd = <OP as ImmOp>::compute(rs1, pre_compute.imm);
    exec_state.vm_write_bytes::<RV64_REGISTER_NUM_LIMBS>(
        RV64_REGISTER_AS,
        pre_compute.rd_ptr as u32,
        &rd.to_le_bytes(),
    );
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, OP: ImmOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &BitwiseLogicImmPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<BitwiseLogicImmPreCompute>()).borrow();
    execute_e12_impl::<CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, OP: ImmOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<BitwiseLogicImmPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<BitwiseLogicImmPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, OP>(&pre_compute.data, exec_state);
}

trait ImmOp {
    fn compute(rs1: u64, imm: u64) -> u64;
}
struct XorOp;
impl ImmOp for XorOp {
    #[inline(always)]
    fn compute(rs1: u64, imm: u64) -> u64 {
        rs1 ^ imm
    }
}
struct OrOp;
impl ImmOp for OrOp {
    #[inline(always)]
    fn compute(rs1: u64, imm: u64) -> u64 {
        rs1 | imm
    }
}
struct AndOp;
impl ImmOp for AndOp {
    #[inline(always)]
    fn compute(rs1: u64, imm: u64) -> u64 {
        rs1 & imm
    }
}
