use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{run_auipc, Rv32AuipcExecutor};
#[cfg(feature = "aot")]
use crate::common::*;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct AuiPcPreCompute {
    imm: u32,
    a: u8,
}

impl<A> Rv32AuipcExecutor<A> {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut AuiPcPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction { a, c: imm, d, .. } = inst;
        if d.as_canonical_u32() != RV32_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let imm = imm.as_canonical_u32();
        let data: &mut AuiPcPreCompute = data.borrow_mut();
        *data = AuiPcPreCompute {
            imm,
            a: a.as_canonical_u32() as u8,
        };
        Ok(())
    }
}

impl<F, A> InterpreterExecutor<F> for Rv32AuipcExecutor<A>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<AuiPcPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut AuiPcPreCompute = data.borrow_mut();
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
        let data: &mut AuiPcPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;
        Ok(execute_e1_handler)
    }
}

#[cfg(feature = "aot")]
impl<F, A> AotExecutor<F> for Rv32AuipcExecutor<A>
where
    F: PrimeField32,
{
    fn generate_x86_asm(&self, inst: &Instruction<F>, pc: u32) -> Result<String, AotError> {
        use openvm_instructions::riscv::RV32_CELL_BITS;

        let to_i16 = |c: F| -> i16 {
            let c_u24 = (c.as_canonical_u64() & 0xFFFFFF) as u32;
            let c_i24 = ((c_u24 << 8) as i32) >> 8;
            c_i24 as i16
        };
        let mut asm_str = String::new();
        let a: i16 = to_i16(inst.a);
        let c: i16 = to_i16(inst.c);
        let d: i16 = to_i16(inst.d);
        let rd = pc.wrapping_add((c as u32) << RV32_CELL_BITS);

        if d as u32 != RV32_REGISTER_AS {
            return Err(AotError::InvalidInstruction);
        }

        let a_reg = a / 4;

        if let Some(override_reg) = RISCV_TO_X86_OVERRIDE_MAP[a_reg as usize] {
            asm_str += &format!("   mov {override_reg}, {rd}\n");
        } else {
            asm_str += &format!("   mov {REG_A_W}, {rd}\n");
            asm_str += &gpr_to_xmm(REG_A_W, a_reg as u8);
        }

        Ok(asm_str)
    }

    fn is_aot_supported(&self, _inst: &Instruction<F>) -> bool {
        true
    }
}

impl<F, A> InterpreterMeteredExecutor<F> for Rv32AuipcExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<AuiPcPreCompute>>()
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
        let data: &mut E2PreCompute<AuiPcPreCompute> = data.borrow_mut();
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
        let data: &mut E2PreCompute<AuiPcPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;
        Ok(execute_e2_handler)
    }
}

#[cfg(feature = "aot")]
impl<F, A> AotMeteredExecutor<F> for Rv32AuipcExecutor<A>
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
        let mut asm_str = update_height_change_asm(chip_idx, 1)?;
        // read [a:4]_1
        asm_str += &update_adapter_heights_asm(config, RV32_REGISTER_AS)?;
        asm_str += &self.generate_x86_asm(inst, pc)?;
        Ok(asm_str)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &AuiPcPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pc = exec_state.pc();
    let rd = run_auipc(pc, pre_compute.imm);
    exec_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &rd);

    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &AuiPcPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<AuiPcPreCompute>()).borrow();
    execute_e12_impl(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<AuiPcPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<AuiPcPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl(&pre_compute.data, exec_state);
}
