use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32JalLuiOpcode::{self, JAL};
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::{get_signed_imm, Rv32JalLuiExecutor};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct JalLuiPreCompute {
    signed_imm: i32,
    a: u8,
}

impl<A> Rv32JalLuiExecutor<A> {
    /// Return (IS_JAL, ENABLED)
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        inst: &Instruction<F>,
        data: &mut JalLuiPreCompute,
    ) -> Result<(bool, bool), StaticProgramError> {
        let local_opcode = Rv32JalLuiOpcode::from_usize(
            inst.opcode.local_opcode_idx(Rv32JalLuiOpcode::CLASS_OFFSET),
        );
        let is_jal = local_opcode == JAL;
        let signed_imm = get_signed_imm(is_jal, inst.c);

        *data = JalLuiPreCompute {
            signed_imm,
            a: inst.a.as_canonical_u32() as u8,
        };
        let enabled = !inst.f.is_zero();
        Ok((is_jal, enabled))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_jal:ident, $enabled:ident) => {
        match ($is_jal, $enabled) {
            (true, true) => Ok($execute_impl::<_, _, true, true>),
            (true, false) => Ok($execute_impl::<_, _, true, false>),
            (false, true) => Ok($execute_impl::<_, _, false, true>),
            (false, false) => Ok($execute_impl::<_, _, false, false>),
        }
    };
}

impl<F, A> InterpreterExecutor<F> for Rv32JalLuiExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<JalLuiPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut JalLuiPreCompute = data.borrow_mut();
        let (is_jal, enabled) = self.pre_compute_impl(inst, data)?;
        dispatch!(execute_e1_handler, is_jal, enabled)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut JalLuiPreCompute = data.borrow_mut();
        let (is_jal, enabled) = self.pre_compute_impl(inst, data)?;
        dispatch!(execute_e1_handler, is_jal, enabled)
    }
}

#[cfg(feature = "aot")]
impl<F, A> AotExecutor<F> for Rv32JalLuiExecutor<A>
where
    F: PrimeField32,
{
    fn generate_x86_asm(&self, inst: &Instruction<F>, pc: u32) -> Result<String, AotError> {
        use crate::common::*;

        let local_opcode = Rv32JalLuiOpcode::from_usize(
            inst.opcode.local_opcode_idx(Rv32JalLuiOpcode::CLASS_OFFSET),
        );
        let is_jal = local_opcode == JAL;
        let signed_imm = get_signed_imm(is_jal, inst.c);
        let a = inst.a.as_canonical_u32() as u8;
        let enabled = !inst.f.is_zero();

        let mut asm_str = String::new();
        let a_reg = a / 4;

        let rd = if is_jal {
            pc + DEFAULT_PC_STEP
        } else {
            let imm = signed_imm as u32;
            imm << 12
        };

        if enabled {
            if let Some(override_reg) = RISCV_TO_X86_OVERRIDE_MAP[a_reg as usize] {
                asm_str += &format!("   mov {override_reg}, {rd}\n");
            } else {
                asm_str += &format!("   mov {REG_A_W}, {rd}\n");
                asm_str += &gpr_to_xmm(REG_A_W, a_reg);
            }
        }
        if is_jal {
            let next_pc = pc as i32 + signed_imm;
            debug_assert!(next_pc >= 0);
            asm_str += &format!("   jmp asm_execute_pc_{next_pc}\n");
        };

        Ok(asm_str)
    }

    fn is_aot_supported(&self, _inst: &Instruction<F>) -> bool {
        true
    }
}

impl<F, A> InterpreterMeteredExecutor<F> for Rv32JalLuiExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<JalLuiPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<JalLuiPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_jal, enabled) = self.pre_compute_impl(inst, &mut data.data)?;
        dispatch!(execute_e2_handler, is_jal, enabled)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<JalLuiPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_jal, enabled) = self.pre_compute_impl(inst, &mut data.data)?;
        dispatch!(execute_e2_handler, is_jal, enabled)
    }
}

#[cfg(feature = "aot")]
impl<F, A> AotMeteredExecutor<F> for Rv32JalLuiExecutor<A>
where
    F: PrimeField32,
{
    fn is_aot_metered_supported(&self, _inst: &Instruction<F>) -> bool {
        true
    }
    fn generate_x86_metered_asm(
        &self,
        inst: &Instruction<F>,
        pc: u32,
        chip_idx: usize,
        config: &SystemConfig,
    ) -> Result<String, AotError> {
        use crate::common::{update_adapter_heights_asm, update_height_change_asm};
        let enabled = !inst.f.is_zero();
        let mut asm_str = update_height_change_asm(chip_idx, 1)?;
        if enabled {
            // write [a:4]_1
            asm_str += &update_adapter_heights_asm(config, RV32_REGISTER_AS)?;
        }
        asm_str += &self.generate_x86_asm(inst, pc)?;
        Ok(asm_str)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_JAL: bool,
    const ENABLED: bool,
>(
    pre_compute: &JalLuiPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let JalLuiPreCompute { a, signed_imm } = *pre_compute;
    let mut pc = exec_state.pc();
    let rd = if IS_JAL {
        let rd_data = (pc + DEFAULT_PC_STEP).to_le_bytes();
        let next_pc = pc as i32 + signed_imm;
        debug_assert!(next_pc >= 0);
        pc = next_pc as u32;
        rd_data
    } else {
        let imm = signed_imm as u32;
        let rd = imm << 12;
        pc += DEFAULT_PC_STEP;
        rd.to_le_bytes()
    };

    if ENABLED {
        exec_state.vm_write(RV32_REGISTER_AS, a as u32, &rd);
    }
    exec_state.set_pc(pc);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_JAL: bool,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &JalLuiPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<JalLuiPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, IS_JAL, ENABLED>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_JAL: bool,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<JalLuiPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<JalLuiPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_JAL, ENABLED>(&pre_compute.data, exec_state);
}
