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
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
};
use openvm_rv64im_transpiler::Rv64DivRemWOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::base_alu_w::sign_extend_32_to_64;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct Rv64DivRemWPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64DivRemWExecutor {
    pub offset: usize,
}

impl Rv64DivRemWExecutor {
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Rv64DivRemWPreCompute,
    ) -> Result<Rv64DivRemWOpcode, StaticProgramError> {
        let local_opcode = Rv64DivRemWOpcode::from_usize(inst.opcode.local_opcode_idx(self.offset));
        if inst.d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = Rv64DivRemWPreCompute {
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
            c: inst.c.as_canonical_u32() as u8,
        };
        Ok(local_opcode)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        Ok(match $local_opcode {
            Rv64DivRemWOpcode::DIVW => $execute_impl::<_, _, DivWOp>,
            Rv64DivRemWOpcode::DIVUW => $execute_impl::<_, _, DivuWOp>,
            Rv64DivRemWOpcode::REMW => $execute_impl::<_, _, RemWOp>,
            Rv64DivRemWOpcode::REMUW => $execute_impl::<_, _, RemuWOp>,
        })
    };
}

impl<F: PrimeField32> InterpreterExecutor<F> for Rv64DivRemWExecutor {
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<Rv64DivRemWPreCompute>()
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
        let data: &mut Rv64DivRemWPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
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
        let data: &mut Rv64DivRemWPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for Rv64DivRemWExecutor {
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Rv64DivRemWPreCompute>>()
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
        let data: &mut E2PreCompute<Rv64DivRemWPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
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
        let data: &mut E2PreCompute<Rv64DivRemWPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

impl<F: PrimeField32, RA> PreflightExecutor<F, RA> for Rv64DivRemWExecutor {
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
impl<F: PrimeField32> AotExecutor<F> for Rv64DivRemWExecutor {
    fn is_aot_supported(&self, _instruction: &Instruction<F>) -> bool {
        false
    }

    fn generate_x86_asm(&self, _inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        Err(AotError::Other("not yet implemented".to_string()))
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for Rv64DivRemWExecutor {
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
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: DivRemWOp>(
    pre_compute: &Rv64DivRemWPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = exec_state.vm_read::<u8, 8>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2 = exec_state.vm_read::<u8, 8>(RV32_REGISTER_AS, pre_compute.c as u32);
    let rs1_lo = u64::from_le_bytes(rs1) as u32;
    let rs2_lo = u64::from_le_bytes(rs2) as u32;
    let result_32 = <OP as DivRemWOp>::compute(rs1_lo, rs2_lo);
    let rd = sign_extend_32_to_64(result_32);
    exec_state.vm_write::<u8, 8>(RV32_REGISTER_AS, pre_compute.a as u32, &rd.to_le_bytes());
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: DivRemWOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &Rv64DivRemWPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<Rv64DivRemWPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, OP: DivRemWOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<Rv64DivRemWPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<Rv64DivRemWPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP>(&pre_compute.data, exec_state);
}

trait DivRemWOp {
    fn compute(rs1: u32, rs2: u32) -> u32;
}
struct DivWOp;
struct DivuWOp;
struct RemWOp;
struct RemuWOp;

impl DivRemWOp for DivWOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        let rs1 = rs1 as i32;
        let rs2 = rs2 as i32;
        match (rs1, rs2) {
            (_, 0) => u32::MAX,
            (i32::MIN, -1) => rs1 as u32,
            _ => (rs1 / rs2) as u32,
        }
    }
}

impl DivRemWOp for DivuWOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        if rs2 == 0 {
            u32::MAX
        } else {
            rs1 / rs2
        }
    }
}

impl DivRemWOp for RemWOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        let rs1 = rs1 as i32;
        let rs2 = rs2 as i32;
        match (rs1, rs2) {
            (_, 0) => rs1 as u32,
            (i32::MIN, -1) => 0,
            _ => (rs1 % rs2) as u32,
        }
    }
}

impl DivRemWOp for RemuWOp {
    #[inline(always)]
    fn compute(rs1: u32, rs2: u32) -> u32 {
        if rs2 == 0 {
            rs1
        } else {
            rs1 % rs2
        }
    }
}
