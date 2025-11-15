use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::ShiftOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::ShiftExecutor;
#[allow(unused_imports)]
use crate::{adapters::imm_to_bytes, common::*};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct ShiftPreCompute {
    c: u32,
    a: u8,
    b: u8,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> ShiftExecutor<A, NUM_LIMBS, LIMB_BITS> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut ShiftPreCompute,
    ) -> Result<(bool, ShiftOpcode), StaticProgramError> {
        let Instruction {
            opcode, a, b, c, e, ..
        } = inst;
        let shift_opcode = ShiftOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let e_u32 = e.as_canonical_u32();
        if inst.d.as_canonical_u32() != RV32_REGISTER_AS
            || !(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = ShiftPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
        };
        // `d` is always expected to be RV32_REGISTER_AS.
        Ok((is_imm, shift_opcode))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $shift_opcode:ident) => {
        match ($is_imm, $shift_opcode) {
            (true, ShiftOpcode::SLL) => Ok($execute_impl::<_, _, true, SllOp>),
            (false, ShiftOpcode::SLL) => Ok($execute_impl::<_, _, false, SllOp>),
            (true, ShiftOpcode::SRL) => Ok($execute_impl::<_, _, true, SrlOp>),
            (false, ShiftOpcode::SRL) => Ok($execute_impl::<_, _, false, SrlOp>),
            (true, ShiftOpcode::SRA) => Ok($execute_impl::<_, _, true, SraOp>),
            (false, ShiftOpcode::SRA) => Ok($execute_impl::<_, _, false, SraOp>),
        }
    };
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterExecutor<F>
    for ShiftExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        size_of::<ShiftPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut ShiftPreCompute = data.borrow_mut();
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, data)?;
        // `d` is always expected to be RV32_REGISTER_AS.
        dispatch!(execute_e1_handler, is_imm, shift_opcode)
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
        let data: &mut ShiftPreCompute = data.borrow_mut();
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, data)?;
        // `d` is always expected to be RV32_REGISTER_AS.
        dispatch!(execute_e1_handler, is_imm, shift_opcode)
    }
}

#[cfg(feature = "aot")]
impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> AotExecutor<F>
    for ShiftExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn is_aot_supported(&self, _instruction: &Instruction<F>) -> bool {
        true
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

        // note: for shift we will use REG_B since
        // it is a hardware requirement that cl is used as the shift value
        // and we don't want to override the written [b:4]_1
        // [a:4]_1 <- [b:4]_1

        let str_reg_a = if RISCV_TO_X86_OVERRIDE_MAP[(a / 4) as usize].is_some() {
            RISCV_TO_X86_OVERRIDE_MAP[(a / 4) as usize].unwrap()
        } else {
            REG_A_W
        };

        if e == 0 {
            // [a:4]_1 <- [b:4]_1 (shift) c
            let mut asm_opcode = String::new();
            if inst.opcode == ShiftOpcode::SLL.global_opcode() {
                asm_opcode += "shl";
            } else if inst.opcode == ShiftOpcode::SRL.global_opcode() {
                asm_opcode += "shr";
            } else if inst.opcode == ShiftOpcode::SRA.global_opcode() {
                asm_opcode += "sar";
            }

            let (reg_b, delta_str_b) = &xmm_to_gpr((b / 4) as u8, str_reg_a, true);
            asm_str += delta_str_b;
            asm_str += &format!("   {asm_opcode} {reg_b}, {c}\n");
            asm_str += &gpr_to_xmm(reg_b, (a / 4) as u8);
        } else {
            // [b:4]_1 <- [b:4]_1 (shift) [c:4]_1
            let mut asm_opcode = String::new();
            if inst.opcode == ShiftOpcode::SLL.global_opcode() {
                asm_opcode += "shlx";
            } else if inst.opcode == ShiftOpcode::SRL.global_opcode() {
                asm_opcode += "shrx";
            } else if inst.opcode == ShiftOpcode::SRA.global_opcode() {
                asm_opcode += "sarx";
            }

            let (reg_b, delta_str_b) = &xmm_to_gpr((b / 4) as u8, REG_B_W, false);
            // after this force write, we set [a:4]_1 <- [b:4]_1
            asm_str += delta_str_b;

            let (reg_c, delta_str_c) = &xmm_to_gpr((c / 4) as u8, REG_C_W, false);
            asm_str += delta_str_c;

            asm_str += &format!("   {asm_opcode} {str_reg_a}, {reg_b}, {reg_c}\n");

            asm_str += &gpr_to_xmm(str_reg_a, (a / 4) as u8);
        }

        // let it fall to the next instruction
        Ok(asm_str)
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> InterpreterMeteredExecutor<F>
    for ShiftExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<ShiftPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, &mut data.data)?;
        // `d` is always expected to be RV32_REGISTER_AS.
        dispatch!(execute_e2_handler, is_imm, shift_opcode)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError> {
        let data: &mut E2PreCompute<ShiftPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, shift_opcode) = self.pre_compute_impl(pc, inst, &mut data.data)?;
        // `d` is always expected to be RV32_REGISTER_AS.
        dispatch!(execute_e2_handler, is_imm, shift_opcode)
    }
}

#[cfg(feature = "aot")]
impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> AotMeteredExecutor<F>
    for ShiftExecutor<A, NUM_LIMBS, LIMB_BITS>
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
        let is_imm = inst.e.as_canonical_u32() == RV32_IMM_AS;
        let mut asm_str = self.generate_x86_asm(inst, pc)?;
        asm_str += &update_height_change_asm(chip_idx, 1)?;
        // read [a:4]_1
        asm_str += &update_adapter_heights_asm(config, RV32_REGISTER_AS)?;
        // read [b:4]_1
        asm_str += &update_adapter_heights_asm(config, RV32_REGISTER_AS)?;
        if !is_imm {
            // read [c:4]_1
            asm_str += &update_adapter_heights_asm(config, RV32_REGISTER_AS)?;
        }
        Ok(asm_str)
    }
}
#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_IMM: bool,
    OP: ShiftOp,
>(
    pre_compute: &ShiftPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let rs1 = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.b as u32);
    let rs2 = if IS_IMM {
        pre_compute.c.to_le_bytes()
    } else {
        exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, pre_compute.c)
    };
    let rs2 = u32::from_le_bytes(rs2);

    // Execute the shift operation
    let rd = <OP as ShiftOp>::compute(rs1, rs2);
    // Write the result back to memory
    exec_state.vm_write(RV32_REGISTER_AS, pre_compute.a as u32, &rd);

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_IMM: bool,
    OP: ShiftOp,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &ShiftPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<ShiftPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, IS_IMM, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_IMM: bool,
    OP: ShiftOp,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<ShiftPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<ShiftPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, IS_IMM, OP>(&pre_compute.data, exec_state);
}

trait ShiftOp {
    fn compute(rs1: [u8; 4], rs2: u32) -> [u8; 4];
}
struct SllOp;
struct SrlOp;
struct SraOp;
impl ShiftOp for SllOp {
    fn compute(rs1: [u8; 4], rs2: u32) -> [u8; 4] {
        let rs1 = u32::from_le_bytes(rs1);
        // `rs2`'s  other bits are ignored.
        (rs1 << (rs2 & 0x1F)).to_le_bytes()
    }
}
impl ShiftOp for SrlOp {
    fn compute(rs1: [u8; 4], rs2: u32) -> [u8; 4] {
        let rs1 = u32::from_le_bytes(rs1);
        // `rs2`'s  other bits are ignored.
        (rs1 >> (rs2 & 0x1F)).to_le_bytes()
    }
}
impl ShiftOp for SraOp {
    fn compute(rs1: [u8; 4], rs2: u32) -> [u8; 4] {
        let rs1 = i32::from_le_bytes(rs1);
        // `rs2`'s  other bits are ignored.
        (rs1 >> (rs2 & 0x1F)).to_le_bytes()
    }
}
