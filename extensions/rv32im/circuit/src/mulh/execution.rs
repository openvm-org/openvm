use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::MulHOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

#[allow(unused_imports)]
use crate::common::*;
use crate::MulHExecutor;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct MulHPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

impl<A, const LIMB_BITS: usize> MulHExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        inst: &Instruction<F>,
        data: &mut MulHPreCompute,
    ) -> Result<MulHOpcode, StaticProgramError> {
        *data = MulHPreCompute {
            a: inst.a.as_canonical_u32() as u8,
            b: inst.b.as_canonical_u32() as u8,
            c: inst.c.as_canonical_u32() as u8,
        };
        Ok(MulHOpcode::from_usize(
            inst.opcode.local_opcode_idx(MulHOpcode::CLASS_OFFSET),
        ))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        match $local_opcode {
            MulHOpcode::MULH => Ok($execute_impl::<_, _, MulHOp>),
            MulHOpcode::MULHSU => Ok($execute_impl::<_, _, MulHSuOp>),
            MulHOpcode::MULHU => Ok($execute_impl::<_, _, MulHUOp>),
        }
    };
}

impl<F, A, const LIMB_BITS: usize> InterpreterExecutor<F>
    for MulHExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<MulHPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut MulHPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode)
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
        let pre_compute: &mut MulHPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(inst, pre_compute)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

#[cfg(feature = "aot")]
impl<F, A, const LIMB_BITS: usize> AotExecutor<F>
    for MulHExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn is_aot_supported(&self, inst: &Instruction<F>) -> bool {
        inst.opcode == MulHOpcode::MULH.global_opcode()
            || inst.opcode == MulHOpcode::MULHSU.global_opcode()
            || inst.opcode == MulHOpcode::MULHU.global_opcode()
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

        let opcode = MulHOpcode::from_usize(inst.opcode.local_opcode_idx(MulHOpcode::CLASS_OFFSET));

        let mut asm = String::new();

        /*
            for implicit multiplication, we need to load the multiplicand into `eax`
            result of hi bits are always stored in `edx`
            can't use REG_C_W, because it is edx, and it gets overridden
        */
        let (_, delta_str_b) = &xmm_to_gpr((b / 4) as u8, "eax", true);
        let (gpr_reg_c, delta_str_c) = &xmm_to_gpr((c / 4) as u8, REG_A_W, false);
        asm += delta_str_b;
        asm += delta_str_c;
        match opcode {
            MulHOpcode::MULH => {
                asm += &format!("   imul {gpr_reg_c}\n");
                asm += &gpr_to_xmm("edx", (a / 4) as u8);
            }
            MulHOpcode::MULHSU => {
                // free to modify edx:eax, since mul and imul operations modify anyways
                asm += &format!("   mov {REG_B_W}, eax\n");
                asm += &format!("   imul {gpr_reg_c}\n");
                asm += "   mov eax, edx\n";
                asm += &format!("   mov edx, {gpr_reg_c}\n");
                asm += "   sar edx, 31\n";
                asm += &format!("   and edx, {REG_B_W}\n");
                asm += "   add eax, edx\n";
                asm += &gpr_to_xmm("eax", (a / 4) as u8);
            }
            MulHOpcode::MULHU => {
                asm += &format!("   mul {gpr_reg_c}\n");
                asm += &gpr_to_xmm("edx", (a / 4) as u8);
            }
        }
        Ok(asm)
    }
}

impl<F, A, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for MulHExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<MulHPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<MulHPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode)
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
        let pre_compute: &mut E2PreCompute<MulHPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

#[cfg(feature = "aot")]
impl<F, A, const LIMB_BITS: usize> AotMeteredExecutor<F>
    for MulHExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
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
        asm_str += &update_adapter_heights_asm(config, RV32_REGISTER_AS)?;
        // read [c:4]_1
        asm_str += &update_adapter_heights_asm(config, RV32_REGISTER_AS)?;
        // write [a:4]_1
        asm_str += &update_adapter_heights_asm(config, RV32_REGISTER_AS)?;

        Ok(asm_str)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: MulHOperation>(
    pre_compute: &MulHPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1: [u8; RV32_REGISTER_NUM_LIMBS] =
        exec_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2: [u8; RV32_REGISTER_NUM_LIMBS] =
        exec_state.vm_read(RV32_REGISTER_AS, pre_compute.c as u32);
    let rd = <OP as MulHOperation>::compute(rs1, rs2);
    exec_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &rd);

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: MulHOperation>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &MulHPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<MulHPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, OP: MulHOperation>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<MulHPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<MulHPreCompute>>()).borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP>(&pre_compute.data, exec_state);
}

trait MulHOperation {
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4];
}
struct MulHOp;
struct MulHSuOp;
struct MulHUOp;
impl MulHOperation for MulHOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        let rs1 = i32::from_le_bytes(rs1) as i64;
        let rs2 = i32::from_le_bytes(rs2) as i64;
        ((rs1.wrapping_mul(rs2) >> 32) as u32).to_le_bytes()
    }
}
impl MulHOperation for MulHSuOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        let rs1 = i32::from_le_bytes(rs1) as i64;
        let rs2 = u32::from_le_bytes(rs2) as i64;
        ((rs1.wrapping_mul(rs2) >> 32) as u32).to_le_bytes()
    }
}
impl MulHOperation for MulHUOp {
    #[inline(always)]
    fn compute(rs1: [u8; 4], rs2: [u8; 4]) -> [u8; 4] {
        let rs1 = u32::from_le_bytes(rs1) as i64;
        let rs2 = u32::from_le_bytes(rs2) as i64;
        ((rs1.wrapping_mul(rs2) >> 32) as u32).to_le_bytes()
    }
}
