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
use openvm_rv64im_transpiler::Rv64BaseAluOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

/// Sign-extend a 24-bit immediate (stored in a u32) to a u64.
/// The immediate is encoded via `i12_to_u24` in the transpiler.
#[inline(always)]
pub(crate) fn imm24_sign_extend_to_u64(imm_u24: u32) -> u64 {
    ((imm_u24 as i32) << 8 >> 8) as i64 as u64
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct Rv64BaseAluPreCompute {
    c: u64,
    a: u8,
    b: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64BaseAluExecutor {
    pub offset: usize,
}

impl Rv64BaseAluExecutor {
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Rv64BaseAluPreCompute,
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
        *data = Rv64BaseAluPreCompute {
            c: if is_imm {
                imm24_sign_extend_to_u64(c_u32)
            } else {
                c_u32 as u64
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
                Rv64BaseAluOpcode::from_usize($opcode.local_opcode_idx($offset)),
            ) {
                (true, Rv64BaseAluOpcode::ADD) => $execute_impl::<_, _, true, AddOp64>,
                (false, Rv64BaseAluOpcode::ADD) => $execute_impl::<_, _, false, AddOp64>,
                (true, Rv64BaseAluOpcode::SUB) => $execute_impl::<_, _, true, SubOp64>,
                (false, Rv64BaseAluOpcode::SUB) => $execute_impl::<_, _, false, SubOp64>,
                (true, Rv64BaseAluOpcode::XOR) => $execute_impl::<_, _, true, XorOp64>,
                (false, Rv64BaseAluOpcode::XOR) => $execute_impl::<_, _, false, XorOp64>,
                (true, Rv64BaseAluOpcode::OR) => $execute_impl::<_, _, true, OrOp64>,
                (false, Rv64BaseAluOpcode::OR) => $execute_impl::<_, _, false, OrOp64>,
                (true, Rv64BaseAluOpcode::AND) => $execute_impl::<_, _, true, AndOp64>,
                (false, Rv64BaseAluOpcode::AND) => $execute_impl::<_, _, false, AndOp64>,
            },
        )
    };
}

impl<F: PrimeField32> InterpreterExecutor<F> for Rv64BaseAluExecutor {
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<Rv64BaseAluPreCompute>()
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
        let data: &mut Rv64BaseAluPreCompute = data.borrow_mut();
        let is_imm = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, is_imm, inst.opcode, self.offset)
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
        let data: &mut Rv64BaseAluPreCompute = data.borrow_mut();
        let is_imm = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, is_imm, inst.opcode, self.offset)
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for Rv64BaseAluExecutor {
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Rv64BaseAluPreCompute>>()
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
        let data: &mut E2PreCompute<Rv64BaseAluPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let is_imm = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, is_imm, inst.opcode, self.offset)
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
        let data: &mut E2PreCompute<Rv64BaseAluPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let is_imm = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, is_imm, inst.opcode, self.offset)
    }
}

impl<F: PrimeField32, RA> PreflightExecutor<F, RA> for Rv64BaseAluExecutor {
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
impl<F: PrimeField32> AotExecutor<F> for Rv64BaseAluExecutor {
    fn is_aot_supported(&self, _instruction: &Instruction<F>) -> bool {
        false
    }

    fn generate_x86_asm(&self, _inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        Err(AotError::Other("not yet implemented".to_string()))
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for Rv64BaseAluExecutor {
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
    OP: AluOp64,
>(
    pre_compute: &Rv64BaseAluPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = exec_state.vm_read::<u8, 8>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2 = if IS_IMM {
        pre_compute.c.to_le_bytes()
    } else {
        exec_state.vm_read::<u8, 8>(RV32_REGISTER_AS, pre_compute.c as u32)
    };
    let rs1 = u64::from_le_bytes(rs1);
    let rs2 = u64::from_le_bytes(rs2);
    let rd = <OP as AluOp64>::compute(rs1, rs2);
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
    OP: AluOp64,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &Rv64BaseAluPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<Rv64BaseAluPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, IS_IMM, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_IMM: bool,
    OP: AluOp64,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<Rv64BaseAluPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<Rv64BaseAluPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_IMM, OP>(&pre_compute.data, exec_state);
}

trait AluOp64 {
    fn compute(rs1: u64, rs2: u64) -> u64;
}
struct AddOp64;
struct SubOp64;
struct XorOp64;
struct OrOp64;
struct AndOp64;
impl AluOp64 for AddOp64 {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        rs1.wrapping_add(rs2)
    }
}
impl AluOp64 for SubOp64 {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        rs1.wrapping_sub(rs2)
    }
}
impl AluOp64 for XorOp64 {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        rs1 ^ rs2
    }
}
impl AluOp64 for OrOp64 {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        rs1 | rs2
    }
}
impl AluOp64 for AndOp64 {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        rs1 & rs2
    }
}
