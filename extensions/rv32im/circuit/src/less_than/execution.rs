use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::LessThanOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::LessThanExecutor;
#[allow(unused_imports)]
use crate::{adapters::imm_to_bytes, common::*};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct LessThanPreCompute {
    c: u32,
    a: u8,
    b: u8,
}

impl<A, const LIMB_BITS: usize> LessThanExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut LessThanPreCompute,
    ) -> Result<(bool, bool), StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS
            || !(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let local_opcode = LessThanOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();

        *data = LessThanPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        Ok((is_imm, local_opcode == LessThanOpcode::SLTU))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $is_sltu:ident) => {
        match ($is_imm, $is_sltu) {
            (true, true) => Ok($execute_impl::<_, _, true, true>),
            (true, false) => Ok($execute_impl::<_, _, true, false>),
            (false, true) => Ok($execute_impl::<_, _, false, true>),
            (false, false) => Ok($execute_impl::<_, _, false, false>),
        }
    };
}

impl<F, A, const LIMB_BITS: usize> InterpreterExecutor<F>
    for LessThanExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<LessThanPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut LessThanPreCompute = data.borrow_mut();
        let (is_imm, is_sltu) = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, is_imm, is_sltu)
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
        let pre_compute: &mut LessThanPreCompute = data.borrow_mut();
        let (is_imm, is_sltu) = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, is_imm, is_sltu)
    }
}

#[cfg(feature = "aot")]
impl<F, A, const LIMB_BITS: usize> AotExecutor<F>
    for LessThanExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn is_aot_supported(&self, _inst: &Instruction<F>) -> bool {
        // true
        false
    }
    fn generate_x86_asm(&self, inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        let to_i16 = |c: F| -> i16 {
            let c_u24 = (c.as_canonical_u64() & 0xFFFFFF) as u32;
            let c_i24 = ((c_u24 << 8) as i32) >> 8;
            c_i24 as i16
        };
        let mut asm_str = String::new();
        let a: i16 = to_i16(inst.a);
        let b: i16 = to_i16(inst.b);
        let c: i16 = to_i16(inst.c);
        let e: i16 = to_i16(inst.e);
        assert!(a % 4 == 0, "instruction.a must be a multiple of 4");
        assert!(b % 4 == 0, "instruction.b must be a multiple of 4");

        asm_str += &rv32_register_to_gpr((b / 4) as u8, REG_A_W);

        let mut asm_opcode = String::new();
        if inst.opcode == LessThanOpcode::SLT.global_opcode() {
            asm_opcode += "setl";
        } else if inst.opcode == LessThanOpcode::SLTU.global_opcode() {
            asm_opcode += "setb";
        }

        if e == 0 {
            asm_str += &format!("   cmp {REG_A_W}, {c}\n");
        } else {
            // [a:4]_1 <- [a:4]_1 + [c:4]_1
            asm_str += &rv32_register_to_gpr((c / 4) as u8, REG_C_W);
            asm_str += &format!("   cmp {REG_A_W}, {REG_C_W}\n");
        }

        // Set REG_A to 1 if less than (signed), 0 otherwise
        asm_str += &format!("   {asm_opcode} cl\n"); // setl cl or setb cl
        asm_str += &format!("   movzx {REG_A_W}, cl\n"); // zero-extend to 32-bit

        asm_str += &gpr_to_rv32_register(REG_A_W, (a / 4) as u8);
        // let it fall to the next instruction
        Ok(asm_str)
    }
}

impl<F, A, const LIMB_BITS: usize> MeteredExecutor<F>
    for LessThanExecutor<A, { RV32_REGISTER_NUM_LIMBS }, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<LessThanPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<LessThanPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (is_imm, is_sltu) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, is_imm, is_sltu)
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
        let pre_compute: &mut E2PreCompute<LessThanPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (is_imm, is_sltu) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, is_imm, is_sltu)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const IS_U32: bool,
>(
    pre_compute: &LessThanPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2 = if E_IS_IMM {
        pre_compute.c.to_le_bytes()
    } else {
        exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.c)
    };
    let cmp_result = if IS_U32 {
        u32::from_le_bytes(rs1) < u32::from_le_bytes(rs2)
    } else {
        i32::from_le_bytes(rs1) < i32::from_le_bytes(rs2)
    };
    let mut rd = [0u8; RV32_REGISTER_NUM_LIMBS];
    rd[0] = cmp_result as u8;
    exec_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &rd);

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const IS_U32: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &LessThanPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<LessThanPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, E_IS_IMM, IS_U32>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const E_IS_IMM: bool,
    const IS_U32: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<LessThanPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<LessThanPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, E_IS_IMM, IS_U32>(&pre_compute.data, exec_state);
}
