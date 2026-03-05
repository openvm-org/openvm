use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::MulWOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

#[cfg(feature = "aot")]
use crate::common::*;
use super::MulWExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct MulWPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<A> MulWExecutor<A> {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut MulWPreCompute,
    ) -> Result<(), StaticProgramError> {
        assert_eq!(
            MulWOpcode::from_usize(inst.opcode.local_opcode_idx(self.offset)),
            MulWOpcode::MULW
        );
        if inst.d.as_canonical_u32() != RV64_REGISTER_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        *data = MulWPreCompute {
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
            c: inst.c.as_canonical_u32() as u8,
        };
        Ok(())
    }
}

impl<F, A> InterpreterExecutor<F> for MulWExecutor<A>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        size_of::<MulWPreCompute>()
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
        let pre_compute: &mut MulWPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
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
        let pre_compute: &mut MulWPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, pre_compute)?;
        Ok(execute_e1_handler)
    }
}

#[cfg(feature = "aot")]
impl<F, A> AotExecutor<F> for MulWExecutor<A>
where
    F: PrimeField32,
{
    fn is_aot_supported(&self, inst: &Instruction<F>) -> bool {
        inst.opcode == MulWOpcode::MULW.global_opcode()
    }

    fn generate_x86_asm(&self, inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        let to_i16 = |c: F| -> i16 {
            let c_u24 = (c.as_canonical_u64() & 0xFFFFFF) as u32;
            let c_i24 = ((c_u24 << 8) as i32) >> 8;
            c_i24 as i16
        };
        let a = to_i16(inst.a);
        let b = to_i16(inst.b);
        let c = to_i16(inst.c);

        if a % 4 != 0 || b % 4 != 0 || c % 4 != 0 {
            return Err(AotError::InvalidInstruction);
        }

        let mut asm_str = String::new();

        let str_reg_a = if RISCV_TO_X86_OVERRIDE_MAP[(a / 4) as usize].is_some() {
            RISCV_TO_X86_OVERRIDE_MAP[(a / 4) as usize].unwrap()
        } else {
            REG_A_W
        };

        if a == c {
            // a = b * c; commutative, so don't need to write to tmp, but should copy c to a first
            let (gpr_reg_c, delta_str_c) = xmm_to_gpr((c / 4) as u8, str_reg_a, true);
            asm_str += &delta_str_c;
            let (gpr_reg_b, delta_str_b) = xmm_to_gpr((b / 4) as u8, REG_C_W, false);
            asm_str += &delta_str_b;
            asm_str += &format!("   imul {gpr_reg_c}, {gpr_reg_b}\n");
            asm_str += &gpr_to_xmm(&gpr_reg_c, (a / 4) as u8);
        } else {
            let (gpr_reg_b, delta_str_b) = xmm_to_gpr((b / 4) as u8, str_reg_a, true);
            asm_str += &delta_str_b; // data is now in gpr_reg_b
            let (gpr_reg_c, delta_str_c) = xmm_to_gpr((c / 4) as u8, REG_C_W, false); // data is in gpr_reg_c now
            asm_str += &delta_str_c; // have to get a return value here, since it modifies further registers too
            asm_str += &format!("   imul {gpr_reg_b}, {gpr_reg_c}\n");
            asm_str += &gpr_to_xmm(&gpr_reg_b, (a / 4) as u8);
        }

        Ok(asm_str)
    }
}

impl<F, A> InterpreterMeteredExecutor<F> for MulWExecutor<A>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<MulWPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<MulWPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
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
        let pre_compute: &mut E2PreCompute<MulWPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        Ok(execute_e2_handler)
    }
}

#[cfg(feature = "aot")]
impl<F, A> AotMeteredExecutor<F> for MulWExecutor<A>
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
        let mut asm_str = self.generate_x86_asm(inst, pc)?;

        asm_str += &update_height_change_asm(chip_idx, 1)?;
        // read [b:4]_1
        asm_str += &update_adapter_heights_asm(config, RV64_REGISTER_AS)?;
        // read [c:4]_1
        asm_str += &update_adapter_heights_asm(config, RV64_REGISTER_AS)?;
        // write [a:4]_1
        asm_str += &update_adapter_heights_asm(config, RV64_REGISTER_AS)?;

        Ok(asm_str)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &MulWPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1: [u8; RV64_WORD_NUM_LIMBS] =
        exec_state.vm_read(RV64_REGISTER_AS, pre_compute.b as u32);
    let rs2: [u8; RV64_WORD_NUM_LIMBS] =
        exec_state.vm_read(RV64_REGISTER_AS, pre_compute.c as u32);
    let rs1 = u32::from_le_bytes(rs1);
    let rs2 = u32::from_le_bytes(rs2);
    let rd_word = rs1.wrapping_mul(rs2);
    let rd = (rd_word as i32 as i64 as u64).to_le_bytes();
    exec_state.vm_write::<u8, RV64_REGISTER_NUM_LIMBS>(RV64_REGISTER_AS, pre_compute.a as u32, &rd);

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &MulWPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<MulWPreCompute>()).borrow();
    execute_e12_impl(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<MulWPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<MulWPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl(&pre_compute.data, exec_state);
}
