use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::{
        E2PreCompute, ExecuteFunc, ExecutionCtxTrait, Executor, MeteredExecutionCtxTrait,
        MeteredExecutor, StaticProgramError, VmExecState,
    },
    system::memory::online::GuestMemory,
};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
};
use openvm_rv32im_transpiler::BranchLessThanOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::BranchLessThanExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct BranchLePreCompute {
    imm: isize,
    a: u8,
    b: u8,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    BranchLessThanExecutor<A, NUM_LIMBS, LIMB_BITS>
{
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut BranchLePreCompute,
    ) -> Result<BranchLessThanOpcode, StaticProgramError> {
        let &Instruction {
            opcode, a, b, c, d, ..
        } = inst;
        let local_opcode = BranchLessThanOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let c = c.as_canonical_u32();
        let imm = if F::ORDER_U32 - c < c {
            -((F::ORDER_U32 - c) as isize)
        } else {
            c as isize
        };
        if d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = BranchLePreCompute {
            imm,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok(local_opcode)
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> Executor<F>
    for BranchLessThanExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<BranchLePreCompute>()
    }

    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut BranchLePreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        let fn_ptr = match local_opcode {
            BranchLessThanOpcode::BLT => execute_e1_impl::<_, _, BltOp>,
            BranchLessThanOpcode::BLTU => execute_e1_impl::<_, _, BltuOp>,
            BranchLessThanOpcode::BGE => execute_e1_impl::<_, _, BgeOp>,
            BranchLessThanOpcode::BGEU => execute_e1_impl::<_, _, BgeuOp>,
        };
        Ok(fn_ptr)
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> MeteredExecutor<F>
    for BranchLessThanExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BranchLePreCompute>>()
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
        let data: &mut E2PreCompute<BranchLePreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        let fn_ptr = match local_opcode {
            BranchLessThanOpcode::BLT => execute_e2_impl::<_, _, BltOp>,
            BranchLessThanOpcode::BLTU => execute_e2_impl::<_, _, BltuOp>,
            BranchLessThanOpcode::BGE => execute_e2_impl::<_, _, BgeOp>,
            BranchLessThanOpcode::BGEU => execute_e2_impl::<_, _, BgeuOp>,
        };
        Ok(fn_ptr)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: BranchLessThanOp>(
    pre_compute: &BranchLePreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32);
    let rs2 = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let jmp = <OP as BranchLessThanOp>::compute(rs1, rs2);
    if jmp {
        vm_state.pc = (vm_state.pc as isize + pre_compute.imm) as u32;
    } else {
        vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    };
    vm_state.instret += 1;
}

unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: BranchLessThanOp>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &BranchLePreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, OP>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, OP: BranchLessThanOp>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<BranchLePreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP>(&pre_compute.data, vm_state);
}

trait BranchLessThanOp {
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> bool;
}
struct BltOp;
struct BltuOp;
struct BgeOp;
struct BgeuOp;

impl BranchLessThanOp for BltOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> bool {
        let rs1 = i32::from_le_bytes(rs1);
        let rs2 = i32::from_le_bytes(rs2);
        rs1 < rs2
    }
}
impl BranchLessThanOp for BltuOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> bool {
        let rs1 = u32::from_le_bytes(rs1);
        let rs2 = u32::from_le_bytes(rs2);
        rs1 < rs2
    }
}
impl BranchLessThanOp for BgeOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> bool {
        let rs1 = i32::from_le_bytes(rs1);
        let rs2 = i32::from_le_bytes(rs2);
        rs1 >= rs2
    }
}
impl BranchLessThanOp for BgeuOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> bool {
        let rs1 = u32::from_le_bytes(rs1);
        let rs2 = u32::from_le_bytes(rs2);
        rs1 >= rs2
    }
}
