use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, PC_BITS},
    riscv::RV32_REGISTER_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::Rv32JalrExecutor;
#[cfg(feature = "aot")]
use crate::common::*;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct JalrPreCompute {
    imm_extended: u32,
    a: u8,
    b: u8,
}

impl<A> Rv32JalrExecutor<A> {
    /// Return true if enabled.
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut JalrPreCompute,
    ) -> Result<bool, StaticProgramError> {
        let imm_extended = inst.c.as_canonical_u32() + inst.g.as_canonical_u32() * 0xffff0000;
        if inst.d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = JalrPreCompute {
            imm_extended,
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
        };
        let enabled = !inst.f.is_zero();
        Ok(enabled)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $enabled:ident) => {
        if $enabled {
            Ok($execute_impl::<_, _, true>)
        } else {
            Ok($execute_impl::<_, _, false>)
        }
    };
}

impl<F, A> InterpreterExecutor<F> for Rv32JalrExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<JalrPreCompute>()
    }
    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut JalrPreCompute = data.borrow_mut();
        let enabled = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, enabled)
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
        let data: &mut JalrPreCompute = data.borrow_mut();
        let enabled = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, enabled)
    }
}

#[cfg(feature = "aot")]
impl<F, A> AotExecutor<F> for Rv32JalrExecutor<A>
where
    F: PrimeField32,
{
    fn is_aot_supported(&self, _inst: &Instruction<F>) -> bool {
        true
    }

    fn generate_x86_asm(&self, inst: &Instruction<F>, pc: u32) -> Result<String, AotError> {
        let mut asm_str = String::new();
        let to_i16 = |c: F| -> i16 {
            let c_u24 = (c.as_canonical_u64() & 0xFFFFFF) as u32;
            let c_i24 = ((c_u24 << 8) as i32) >> 8;
            c_i24 as i16
        };
        let a = to_i16(inst.a);
        let b = to_i16(inst.b);
        if a % 4 != 0 || b % 4 != 0 {
            return Err(AotError::InvalidInstruction);
        }
        let imm_extended = inst.c.as_canonical_u32() + inst.g.as_canonical_u32() * 0xffff0000;
        let write_rd = !inst.f.is_zero();

        let (gpr_reg_b, delta_b) = xmm_to_gpr((b / 4) as u8, REG_B_W, true);
        asm_str += &delta_b;
        asm_str += &format!("   add {gpr_reg_b}, {imm_extended}\n");
        asm_str += &format!("   and {gpr_reg_b}, -2\n"); // clear bit 0 per RISC-V jalr

        let gpr_reg_b_64 = convert_x86_reg(&gpr_reg_b, Width::W64).unwrap();

        if write_rd {
            let next_pc = pc.wrapping_add(DEFAULT_PC_STEP);
            asm_str += &format!("   mov {REG_A_W}, {next_pc}\n");
            asm_str += &gpr_to_xmm(REG_A_W, (a / 4) as u8);
        }

        asm_str += &format!("   lea {REG_C}, [rip + map_pc_base]\n");
        asm_str += &format!("   movsxd {REG_A}, [{REG_C} + {gpr_reg_b_64}]\n");
        asm_str += &format!("   add {REG_A}, {REG_C}\n");
        asm_str += &format!("   jmp {REG_A}\n");
        Ok(asm_str)
    }
}

impl<F, A> InterpreterMeteredExecutor<F> for Rv32JalrExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<JalrPreCompute>>()
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
        let data: &mut E2PreCompute<JalrPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let enabled = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, enabled)
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
        let data: &mut E2PreCompute<JalrPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let enabled = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, enabled)
    }
}

#[cfg(feature = "aot")]
impl<F, A> AotMeteredExecutor<F> for Rv32JalrExecutor<A>
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
        let enabled = !inst.f.is_zero();
        let mut asm_str = update_height_change_asm(chip_idx, 1)?;
        // read [b:4]_1
        asm_str += &update_adapter_heights_asm(config, RV32_REGISTER_AS)?;
        if enabled {
            // write [a:4]_1
            asm_str += &update_adapter_heights_asm(config, RV32_REGISTER_AS)?;
        }
        asm_str += &self.generate_x86_asm(inst, pc)?;
        Ok(asm_str)
    }
}
#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const ENABLED: bool>(
    pre_compute: &JalrPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pc = exec_state.pc();
    let rs1 = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs1 = u32::from_le_bytes(rs1);
    let to_pc = rs1.wrapping_add(pre_compute.imm_extended);
    let to_pc = to_pc - (to_pc & 1);
    debug_assert!(to_pc < (1 << PC_BITS));
    let rd = (pc + DEFAULT_PC_STEP).to_le_bytes();

    if ENABLED {
        exec_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &rd);
    }

    exec_state.set_pc(to_pc);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const ENABLED: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &JalrPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<JalrPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, ENABLED>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, const ENABLED: bool>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<JalrPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<JalrPreCompute>>()).borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, ENABLED>(&pre_compute.data, exec_state);
}
