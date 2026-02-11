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
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::base_alu_w::sign_extend_32_to_64;

const RV32_CELL_BITS: usize = 8;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct Rv64AuipcPreCompute {
    imm: u32,
    a: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64AuipcExecutor {
    pub offset: usize,
}

impl Rv64AuipcExecutor {
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut Rv64AuipcPreCompute,
    ) -> Result<(), StaticProgramError> {
        if inst.d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let imm = inst.c.as_canonical_u32();
        *data = Rv64AuipcPreCompute {
            imm,
            a: inst.a.as_canonical_u32() as u8,
        };
        Ok(())
    }
}

impl<F: PrimeField32> InterpreterExecutor<F> for Rv64AuipcExecutor {
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<Rv64AuipcPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut Rv64AuipcPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_impl)
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
        let data: &mut Rv64AuipcPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler)
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for Rv64AuipcExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Rv64AuipcPreCompute>>()
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
        let data: &mut E2PreCompute<Rv64AuipcPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_impl)
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
        let data: &mut E2PreCompute<Rv64AuipcPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler)
    }
}

impl<F: PrimeField32, RA> PreflightExecutor<F, RA> for Rv64AuipcExecutor {
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
impl<F: PrimeField32> AotExecutor<F> for Rv64AuipcExecutor {
    fn is_aot_supported(&self, _instruction: &Instruction<F>) -> bool {
        false
    }

    fn generate_x86_asm(&self, _inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        Err(AotError::Other("not yet implemented".to_string()))
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for Rv64AuipcExecutor {
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
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &Rv64AuipcPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pc = exec_state.pc();
    // imm is stored as (upper_20_bits << 12) >> 8 = upper_20_bits << 4
    // So imm << RV32_CELL_BITS reconstructs the original (upper_20_bits << 12)
    let rd32 = pc.wrapping_add(pre_compute.imm << RV32_CELL_BITS);
    let rd: [u8; 8] = sign_extend_32_to_64(rd32).to_le_bytes();
    exec_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &rd);
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &Rv64AuipcPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<Rv64AuipcPreCompute>()).borrow();
    execute_e12_impl(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<Rv64AuipcPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<Rv64AuipcPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl(&pre_compute.data, exec_state);
}
