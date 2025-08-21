use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{adapters::imm_to_bytes, BaseAluExecutor};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct BaseAluPreCompute {
    c: u32,
    a: u8,
    b: u8,
}

impl<A, const LIMB_BITS: usize> BaseAluExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS> {
    /// Return `is_imm`, true if `e` is RV32_IMM_AS.
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut BaseAluPreCompute,
    ) -> Result<bool, StaticProgramError> {
        let Instruction { a, b, c, d, e, .. } = inst;
        let e_u32 = e.as_canonical_u32();
        if (d.as_canonical_u32() != RV32_REGISTER_AS)
            || !(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = BaseAluPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok(is_imm)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $opcode:expr, $offset:expr) => {
        Ok(
            match (
                $is_imm,
                BaseAluOpcode::from_usize($opcode.local_opcode_idx($offset)),
            ) {
                (true, BaseAluOpcode::ADD) => $execute_impl::<_, _, true, AddOp>,
                (false, BaseAluOpcode::ADD) => $execute_impl::<_, _, false, AddOp>,
                (true, BaseAluOpcode::SUB) => $execute_impl::<_, _, true, SubOp>,
                (false, BaseAluOpcode::SUB) => $execute_impl::<_, _, false, SubOp>,
                (true, BaseAluOpcode::XOR) => $execute_impl::<_, _, true, XorOp>,
                (false, BaseAluOpcode::XOR) => $execute_impl::<_, _, false, XorOp>,
                (true, BaseAluOpcode::OR) => $execute_impl::<_, _, true, OrOp>,
                (false, BaseAluOpcode::OR) => $execute_impl::<_, _, false, OrOp>,
                (true, BaseAluOpcode::AND) => $execute_impl::<_, _, true, AndOp>,
                (false, BaseAluOpcode::AND) => $execute_impl::<_, _, false, AndOp>,
            },
        )
    };
}

impl<F, A, const LIMB_BITS: usize> Executor<F>
    for BaseAluExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<BaseAluPreCompute>()
    }

    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut BaseAluPreCompute = data.borrow_mut();
        let is_imm = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_impl, is_imm, inst.opcode, self.offset)
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
        let data: &mut BaseAluPreCompute = data.borrow_mut();
        let is_imm = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_tco_handler, is_imm, inst.opcode, self.offset)
    }
}

impl<F, A, const LIMB_BITS: usize> MeteredExecutor<F>
    for BaseAluExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BaseAluPreCompute>>()
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
        let data: &mut E2PreCompute<BaseAluPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let is_imm = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_impl, is_imm, inst.opcode, self.offset)
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
        let data: &mut E2PreCompute<BaseAluPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let is_imm = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_tco_handler, is_imm, inst.opcode, self.offset)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_IMM: bool,
    OP: AluOp,
>(
    pre_compute: &BaseAluPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2 = if IS_IMM {
        pre_compute.c.to_le_bytes()
    } else {
        vm_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.c)
    };
    let rs1 = u32::from_le_bytes(rs1);
    let rs2 = u32::from_le_bytes(rs2);
    let rd = <OP as AluOp>::compute(rs1, rs2);
    let rd = rd.to_le_bytes();
    vm_state.vm_write::<u8, 4>(RV32_REGISTER_AS, pre_compute.a as u32, &rd);
    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

#[create_tco_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_IMM: bool,
    OP: AluOp,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &BaseAluPreCompute = pre_compute.borrow();
    execute_e12_impl::<F, CTX, IS_IMM, OP>(pre_compute, vm_state);
}

#[create_tco_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_IMM: bool,
    OP: AluOp,
>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<BaseAluPreCompute> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_IMM, OP>(&pre_compute.data, vm_state);
}

trait AluOp {
    fn compute(rs1: u32, rs2: u32) -> u32;
}
struct AddOp;
struct SubOp;
struct XorOp;
struct OrOp;
struct AndOp;
impl AluOp for AddOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1.wrapping_add(rs2)
    }
}
impl AluOp for SubOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1.wrapping_sub(rs2)
    }
}
impl AluOp for XorOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1 ^ rs2
    }
}
impl AluOp for OrOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1 | rs2
    }
}
impl AluOp for AndOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1 & rs2
    }
}
