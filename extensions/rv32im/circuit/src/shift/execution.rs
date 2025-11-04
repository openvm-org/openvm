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
    LocalOpcode, VmOpcode,
};
use openvm_rv32im_transpiler::ShiftOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use super::ShiftExecutor;
use crate::{
    adapters::imm_to_bytes,
    common::{gpr_to_rv32_register, rv32_register_to_gpr},
};

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

// Callee saved
const REG_EXEC_STATE_PTR: &str = "rbx";
const REG_INSNS_PTR: &str = "rbp";
const REG_PC: &str = "r13";
const REG_INSTRET: &str = "r14";
const REG_GUEST_MEM_PTR: &str = "r15";

// Caller saved
const REG_B: &str = "rax";
const REG_B_W: &str = "eax";

const REG_A: &str = "rcx";
const REG_A_W: &str = "ecx";

const REG_FOURTH_ARG: &str = "rcx";
const REG_THIRD_ARG: &str = "rdx";
const REG_SECOND_ARG: &str = "rsi";
const REG_FIRST_ARG: &str = "rdi";
const REG_RETURN_VAL: &str = "rax";

const REG_C: &str = "r10";
const REG_C_W: &str = "r10d";
const REG_C_B: &str = "r10b";
const REG_AUX: &str = "r11";

const DEFAULT_PC_OFFSET: i32 = 4;

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
    fn is_aot_supported(&self, instruction: &Instruction<F>) -> bool {
        true
    }
    fn generate_x86_asm(&self, inst: &Instruction<F>, pc: u32) -> Result<String, AotError> {
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
        asm_str += &rv32_register_to_gpr((b / 4) as u8, REG_B_W);

        let mut asm_opcode = String::new();
        if inst.opcode == ShiftOpcode::SLL.global_opcode() {
            asm_opcode += "shl";
        } else if inst.opcode == ShiftOpcode::SRL.global_opcode() {
            asm_opcode += "shr";
        } else if inst.opcode == ShiftOpcode::SRA.global_opcode() {
            asm_opcode += "sar";
        }

        if e == 0 {
            // [b:4]_1 <- [b:4]_1 (shift) c
            asm_str += &format!("   {} {}, {}\n", asm_opcode, REG_B_W, c);
        } else {
            // [b:4]_1 <- [b:4]_1 (shift) [c:4]_1

            asm_str += &rv32_register_to_gpr((c / 4) as u8, REG_C_W);
            asm_str += &format!("   mov cl, {}\n", REG_C_B);
            // reg_a = reg_a << c
            asm_str += &format!("   {} {}, cl\n", asm_opcode, REG_B_W);
        }
        // General Register -> XMM
        asm_str += &gpr_to_rv32_register(REG_B_W, (a / 4) as u8);
        asm_str += &format!("   add {}, {}\n", REG_PC, DEFAULT_PC_OFFSET);
        asm_str += &format!("   add {}, {}\n", REG_INSTRET, 1);
        // let it fall to the next instruction
        Ok(asm_str)
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> MeteredExecutor<F>
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
    pre_compute: &[u8],
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &ShiftPreCompute = pre_compute.borrow();
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
    pre_compute: &[u8],
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<ShiftPreCompute> = pre_compute.borrow();
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
