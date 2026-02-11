use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::*,
    system::memory::online::{GuestMemory, TracingMemory},
};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv64im_transpiler::Rv64ShiftWOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::base_alu::imm24_sign_extend_to_u64;
use crate::base_alu_w::sign_extend_32_to_64;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct Rv64ShiftWPreCompute {
    c: u64,
    a: u8,
    b: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64ShiftWExecutor {
    pub offset: usize,
}

impl Rv64ShiftWExecutor {
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Rv64ShiftWPreCompute,
    ) -> Result<(bool, Rv64ShiftWOpcode), StaticProgramError> {
        let Instruction {
            opcode, a, b, c, e, ..
        } = inst;
        let shift_opcode = Rv64ShiftWOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let e_u32 = e.as_canonical_u32();
        if inst.d.as_canonical_u32() != RV32_REGISTER_AS
            || !(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = Rv64ShiftWPreCompute {
            c: if is_imm {
                imm24_sign_extend_to_u64(c_u32)
            } else {
                c_u32 as u64
            },
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok((is_imm, shift_opcode))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $shift_opcode:ident) => {
        match ($is_imm, $shift_opcode) {
            (true, Rv64ShiftWOpcode::SLLW) => Ok($execute_impl::<_, _, true, SllwOp>),
            (false, Rv64ShiftWOpcode::SLLW) => Ok($execute_impl::<_, _, false, SllwOp>),
            (true, Rv64ShiftWOpcode::SRLW) => Ok($execute_impl::<_, _, true, SrlwOp>),
            (false, Rv64ShiftWOpcode::SRLW) => Ok($execute_impl::<_, _, false, SrlwOp>),
            (true, Rv64ShiftWOpcode::SRAW) => Ok($execute_impl::<_, _, true, SrawOp>),
            (false, Rv64ShiftWOpcode::SRAW) => Ok($execute_impl::<_, _, false, SrawOp>),
        }
    };
}

impl<F: PrimeField32> InterpreterExecutor<F> for Rv64ShiftWExecutor {
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<Rv64ShiftWPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut Rv64ShiftWPreCompute = data.borrow_mut();
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, is_imm, shift_opcode)
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
        let data: &mut Rv64ShiftWPreCompute = data.borrow_mut();
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, is_imm, shift_opcode)
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for Rv64ShiftWExecutor {
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Rv64ShiftWPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
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
        let data: &mut E2PreCompute<Rv64ShiftWPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, is_imm, shift_opcode)
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
        let data: &mut E2PreCompute<Rv64ShiftWPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, is_imm, shift_opcode)
    }
}

impl<F: PrimeField32, RA> PreflightExecutor<F, RA> for Rv64ShiftWExecutor {
    fn get_opcode_name(&self, _opcode: usize) -> String {
        panic!("not yet implemented")
    }

    fn execute(
        &self,
        _state: VmStateMut<F, TracingMemory, RA>,
        _instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        panic!("not yet implemented")
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotExecutor<F> for Rv64ShiftWExecutor {
    fn is_aot_supported(&self, _instruction: &Instruction<F>) -> bool {
        false
    }

    fn generate_x86_asm(&self, _inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        Err(AotError::Other("not yet implemented".to_string()))
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for Rv64ShiftWExecutor {
    fn is_aot_metered_supported(&self, _inst: &Instruction<F>) -> bool {
        false
    }

    fn generate_x86_metered_asm(
        &self,
        _inst: &Instruction<F>,
        _pc: u32,
        _chip_idx: usize,
        _config: &SystemConfig,
    ) -> Result<String, AotError> {
        Err(AotError::Other("not yet implemented".to_string()))
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_IMM: bool,
    OP: ShiftWOp,
>(
    pre_compute: &Rv64ShiftWPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = exec_state.vm_read::<u8, 8>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2 = if IS_IMM {
        pre_compute.c.to_le_bytes()
    } else {
        exec_state.vm_read::<u8, 8>(RV32_REGISTER_AS, pre_compute.c as u32)
    };
    // Truncate to lower 32 bits
    let rs1_lo = u64::from_le_bytes(rs1) as u32;
    let rs2_lo = u64::from_le_bytes(rs2) as u32;
    // Perform 32-bit shift, then sign-extend result to 64 bits
    let result_32 = <OP as ShiftWOp>::compute(rs1_lo, rs2_lo);
    let rd = sign_extend_32_to_64(result_32);
    let rd = rd.to_le_bytes();
    exec_state.vm_write::<u8, 8>(RV32_REGISTER_AS, pre_compute.a as u32, &rd);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_IMM: bool,
    OP: ShiftWOp,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &Rv64ShiftWPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<Rv64ShiftWPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, IS_IMM, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_IMM: bool,
    OP: ShiftWOp,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<Rv64ShiftWPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<Rv64ShiftWPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_IMM, OP>(&pre_compute.data, exec_state);
}

trait ShiftWOp {
    fn compute(rs1: u32, rs2: u32) -> u32;
}
struct SllwOp;
struct SrlwOp;
struct SrawOp;
impl ShiftWOp for SllwOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1 << (rs2 & 0x1F)
    }
}
impl ShiftWOp for SrlwOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        rs1 >> (rs2 & 0x1F)
    }
}
impl ShiftWOp for SrawOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        ((rs1 as i32) >> (rs2 & 0x1F)) as u32
    }
}
