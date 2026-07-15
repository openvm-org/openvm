use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::BitwiseLogicExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct BitwiseLogicPreCompute {
    c: u64,
    a: u8,
    b: u8,
}

impl<A, const LIMB_BITS: usize> BitwiseLogicExecutor<A, { RV64_REGISTER_NUM_LIMBS }, LIMB_BITS> {
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut BitwiseLogicPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction { a, b, c, d, e, .. } = inst;
        if d.as_canonical_u32() != RV64_REGISTER_AS || e.as_canonical_u32() != RV64_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = BitwiseLogicPreCompute {
            c: c.as_canonical_u32() as u64,
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok(())
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $opcode:expr, $offset:expr) => {
        Ok(
            match BaseAluOpcode::from_usize($opcode.local_opcode_idx($offset)) {
                BaseAluOpcode::XOR => $execute_impl::<_, XorOp>,
                BaseAluOpcode::OR => $execute_impl::<_, OrOp>,
                BaseAluOpcode::AND => $execute_impl::<_, AndOp>,
                _ => unreachable!("BitwiseLogicExecutor received non-XOR/OR/AND opcode"),
            },
        )
    };
}

impl<F, A, const LIMB_BITS: usize> InterpreterExecutor<F>
    for BitwiseLogicExecutor<A, { RV64_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<BitwiseLogicPreCompute>()
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
        let data: &mut BitwiseLogicPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, inst.opcode, self.offset)
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
        let data: &mut BitwiseLogicPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, inst.opcode, self.offset)
    }
}

impl<F, A, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for BitwiseLogicExecutor<A, { RV64_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BitwiseLogicPreCompute>>()
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
        let data: &mut E2PreCompute<BitwiseLogicPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, inst.opcode, self.offset)
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
        let data: &mut E2PreCompute<BitwiseLogicPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, inst.opcode, self.offset)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, OP: AluOp>(
    pre_compute: &BitwiseLogicPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let rs1 = exec_state.vm_read_bytes::<8>(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs2: [u8; 8] = exec_state.vm_read_bytes(RV64_REGISTER_AS, pre_compute.c as u32);
    let rs1 = u64::from_le_bytes(rs1);
    let rs2 = u64::from_le_bytes(rs2);
    let rd = <OP as AluOp>::compute(rs1, rs2);
    let rd = rd.to_le_bytes();
    exec_state.vm_write_bytes::<8>(RV64_REGISTER_AS, pre_compute.a as u32, &rd);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, OP: AluOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &BitwiseLogicPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<BitwiseLogicPreCompute>()).borrow();
    execute_e12_impl::<CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, OP: AluOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<BitwiseLogicPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<BitwiseLogicPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, OP>(&pre_compute.data, exec_state);
}

trait AluOp {
    fn compute(rs1: u64, rs2: u64) -> u64;
}
struct XorOp;
struct OrOp;
struct AndOp;
impl AluOp for XorOp {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        rs1 ^ rs2
    }
}
impl AluOp for OrOp {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        rs1 | rs2
    }
}
impl AluOp for AndOp {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        rs1 & rs2
    }
}
