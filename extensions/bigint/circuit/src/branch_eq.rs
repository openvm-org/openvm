use std::borrow::{Borrow, BorrowMut};

use openvm_bigint_transpiler::BranchEqual256Opcode;
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{MEMORY_AS, REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_circuit::adapters::{
    bytes_to_u32, decode_signed_instruction_imm, RV_B_TYPE_IMM_BITS,
};
use openvm_riscv_transpiler::BranchEqualOpcode;

use crate::{
    common::{bytes_to_u64_array, read_int256},
    BranchEqual256Executor, INT256_NUM_U64_LIMBS, INT256_NUM_U8_LIMBS,
};

impl BranchEqual256Executor {
    pub fn new() -> Self {
        Self
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct BranchEqPreCompute {
    imm: i32,
    a: u8,
    b: u8,
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        match $local_opcode {
            BranchEqualOpcode::BEQ => Ok($execute_impl::<_, false>),
            BranchEqualOpcode::BNE => Ok($execute_impl::<_, true>),
        }
    };
}

impl InterpreterExecutor for BranchEqual256Executor {
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            BranchEqualOpcode::from_usize(opcode - BranchEqual256Opcode::CLASS_OFFSET)
        )
    }

    fn pre_compute_size(&self) -> usize {
        size_of::<BranchEqPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut BranchEqPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
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
        let data: &mut BranchEqPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

impl InterpreterMeteredExecutor for BranchEqual256Executor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BranchEqPreCompute>>()
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
        let data: &mut E2PreCompute<BranchEqPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
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
        let data: &mut E2PreCompute<BranchEqPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<CTX: ExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: &BranchEqPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let mut pc = exec_state.pc();
    let rs1_ptr = exec_state.vm_read_bytes::<REGISTER_NUM_LIMBS>(REGISTER_AS, pre_compute.a as u32);
    let rs2_ptr = exec_state.vm_read_bytes::<REGISTER_NUM_LIMBS>(REGISTER_AS, pre_compute.b as u32);
    let rs1 = read_int256(exec_state, MEMORY_AS, bytes_to_u32(rs1_ptr))?;
    let rs2 = read_int256(exec_state, MEMORY_AS, bytes_to_u32(rs2_ptr))?;
    let cmp_result = u256_eq(rs1, rs2);
    if cmp_result ^ IS_NE {
        pc = pc.wrapping_add_signed(pre_compute.imm);
    } else {
        pc = pc.wrapping_add(DEFAULT_PC_STEP);
    }
    exec_state.set_pc(pc);
    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<CTX: ExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &BranchEqPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<BranchEqPreCompute>()).borrow();
    execute_e12_impl::<CTX, IS_NE>(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<CTX: MeteredExecutionCtxTrait, const IS_NE: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &E2PreCompute<BranchEqPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<BranchEqPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<CTX, IS_NE>(&pre_compute.data, exec_state)
}

impl BranchEqual256Executor {
    fn pre_compute_impl(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut BranchEqPreCompute,
    ) -> Result<BranchEqualOpcode, StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let imm = decode_signed_instruction_imm(*c, RV_B_TYPE_IMM_BITS)
            .ok_or(StaticProgramError::InvalidInstruction(pc))?;
        let e_u32 = e.as_u32();
        if d.as_u32() != REGISTER_AS || e_u32 != MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = BranchEqPreCompute {
            imm,
            a: a.as_u32() as u8,
            b: b.as_u32() as u8,
        };
        let local_opcode = BranchEqualOpcode::from_usize(
            opcode.local_opcode_idx(BranchEqual256Opcode::CLASS_OFFSET),
        );
        Ok(local_opcode)
    }
}

fn u256_eq(rs1: [u8; INT256_NUM_U8_LIMBS], rs2: [u8; INT256_NUM_U8_LIMBS]) -> bool {
    let rs1_u64 = bytes_to_u64_array(rs1);
    let rs2_u64 = bytes_to_u64_array(rs2);
    for i in 0..INT256_NUM_U64_LIMBS {
        if rs1_u64[i] != rs2_u64[i] {
            return false;
        }
    }
    true
}
