use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::{GuestMemory, TracingMemory}};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::RV32_REGISTER_AS,
    LocalOpcode,
};
use openvm_rv64im_transpiler::Rv64DivRemOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct Rv64DivRemPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64DivRemExecutor {
    pub offset: usize,
}

impl Rv64DivRemExecutor {
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Rv64DivRemPreCompute,
    ) -> Result<Rv64DivRemOpcode, StaticProgramError> {
        let local_opcode =
            Rv64DivRemOpcode::from_usize(inst.opcode.local_opcode_idx(self.offset));
        if inst.d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = Rv64DivRemPreCompute {
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
            c: inst.c.as_canonical_u32() as u8,
        };
        Ok(local_opcode)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        match $local_opcode {
            Rv64DivRemOpcode::DIV => Ok($execute_impl::<_, _, DivOp64>),
            Rv64DivRemOpcode::DIVU => Ok($execute_impl::<_, _, DivuOp64>),
            Rv64DivRemOpcode::REM => Ok($execute_impl::<_, _, RemOp64>),
            Rv64DivRemOpcode::REMU => Ok($execute_impl::<_, _, RemuOp64>),
        }
    };
}

impl<F: PrimeField32> InterpreterExecutor<F> for Rv64DivRemExecutor {
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<Rv64DivRemPreCompute>()
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
        let data: &mut Rv64DivRemPreCompute = data.borrow_mut();
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
        let data: &mut Rv64DivRemPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for Rv64DivRemExecutor {
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Rv64DivRemPreCompute>>()
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
        let data: &mut E2PreCompute<Rv64DivRemPreCompute> = data.borrow_mut();
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
        let data: &mut E2PreCompute<Rv64DivRemPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

impl<F: PrimeField32, RA> PreflightExecutor<F, RA> for Rv64DivRemExecutor {
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
impl<F: PrimeField32> AotExecutor<F> for Rv64DivRemExecutor {
    fn is_aot_supported(&self, _instruction: &Instruction<F>) -> bool {
        false
    }

    fn generate_x86_asm(&self, _inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        Err(AotError::Other("not yet implemented".to_string()))
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for Rv64DivRemExecutor {
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
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: DivRemOp64>(
    pre_compute: &Rv64DivRemPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = exec_state.vm_read::<u8, 8>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2 = exec_state.vm_read::<u8, 8>(RV32_REGISTER_AS, pre_compute.c as u32);
    let result = <OP as DivRemOp64>::compute(u64::from_le_bytes(rs1), u64::from_le_bytes(rs2));
    exec_state.vm_write::<u8, 8>(RV32_REGISTER_AS, pre_compute.a as u32, &result.to_le_bytes());
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: DivRemOp64>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &Rv64DivRemPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<Rv64DivRemPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, OP: DivRemOp64>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<Rv64DivRemPreCompute> = std::slice::from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<Rv64DivRemPreCompute>>(),
    )
    .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP>(&pre_compute.data, exec_state);
}

trait DivRemOp64 {
    fn compute(rs1: u64, rs2: u64) -> u64;
}
struct DivOp64;
struct DivuOp64;
struct RemOp64;
struct RemuOp64;

impl DivRemOp64 for DivOp64 {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        let rs1 = rs1 as i64;
        let rs2 = rs2 as i64;
        match (rs1, rs2) {
            (_, 0) => u64::MAX,
            (i64::MIN, -1) => rs1 as u64,
            _ => (rs1 / rs2) as u64,
        }
    }
}

impl DivRemOp64 for DivuOp64 {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        if rs2 == 0 {
            u64::MAX
        } else {
            rs1 / rs2
        }
    }
}

impl DivRemOp64 for RemOp64 {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        let rs1 = rs1 as i64;
        let rs2 = rs2 as i64;
        match (rs1, rs2) {
            (_, 0) => rs1 as u64,
            (i64::MIN, -1) => 0,
            _ => (rs1 % rs2) as u64,
        }
    }
}

impl DivRemOp64 for RemuOp64 {
    #[inline(always)]
    fn compute(rs1: u64, rs2: u64) -> u64 {
        if rs2 == 0 {
            rs1
        } else {
            rs1 % rs2
        }
    }
}
