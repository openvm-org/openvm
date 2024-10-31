use std::marker::PhantomData;

use axvm_instructions::{
    instruction::Instruction,
    riscv::{RvIntrinsic, RV32_REGISTER_NUM_LIMBS},
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, DivRemOpcode, EccOpcode,
    LessThanOpcode, MulHOpcode, MulOpcode, PhantomInstruction, Rv32AuipcOpcode,
    Rv32HintStoreOpcode, Rv32JalLuiOpcode, Rv32JalrOpcode, Rv32LoadStoreOpcode,
    Rv32ModularArithmeticOpcode, ShiftOpcode, UsizeOpcode,
};
use axvm_platform::constants::{CUSTOM_0, CUSTOM_1};
use p3_field::PrimeField32;
use rrs_lib::{
    instruction_formats::{BType, IType, ITypeShamt, JType, RType, SType, UType},
    process_instruction, InstructionProcessor,
};
use strum::EnumCount;

use crate::util::*;

/// A transpiler that converts the 32-bit encoded instructions into instructions.
#[allow(dead_code)]
pub(crate) struct InstructionTranspiler<F>(PhantomData<F>);

impl<F: PrimeField32> InstructionProcessor for InstructionTranspiler<F> {
    type InstructionResult = Instruction<F>;

    fn process_add(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(BaseAluOpcode::ADD.with_default_offset(), 1, &dec_insn)
    }

    fn process_addi(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(BaseAluOpcode::ADD.with_default_offset(), &dec_insn)
    }

    fn process_sub(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(BaseAluOpcode::SUB.with_default_offset(), 1, &dec_insn)
    }

    fn process_xor(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(BaseAluOpcode::XOR.with_default_offset(), 1, &dec_insn)
    }

    fn process_xori(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(BaseAluOpcode::XOR.with_default_offset(), &dec_insn)
    }

    fn process_or(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(BaseAluOpcode::OR.with_default_offset(), 1, &dec_insn)
    }

    fn process_ori(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(BaseAluOpcode::OR.with_default_offset(), &dec_insn)
    }

    fn process_and(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(BaseAluOpcode::AND.with_default_offset(), 1, &dec_insn)
    }

    fn process_andi(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(BaseAluOpcode::AND.with_default_offset(), &dec_insn)
    }

    fn process_sll(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(ShiftOpcode::SLL.with_default_offset(), 1, &dec_insn)
    }

    fn process_slli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(ShiftOpcode::SLL.with_default_offset(), &dec_insn)
    }

    fn process_srl(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(ShiftOpcode::SRL.with_default_offset(), 1, &dec_insn)
    }

    fn process_srli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(ShiftOpcode::SRL.with_default_offset(), &dec_insn)
    }

    fn process_sra(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(ShiftOpcode::SRA.with_default_offset(), 1, &dec_insn)
    }

    fn process_srai(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(ShiftOpcode::SRA.with_default_offset(), &dec_insn)
    }

    fn process_slt(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(LessThanOpcode::SLT.with_default_offset(), 1, &dec_insn)
    }

    fn process_slti(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(LessThanOpcode::SLT.with_default_offset(), &dec_insn)
    }

    fn process_sltu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(LessThanOpcode::SLTU.with_default_offset(), 1, &dec_insn)
    }

    fn process_sltui(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(LessThanOpcode::SLTU.with_default_offset(), &dec_insn)
    }

    fn process_lb(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(Rv32LoadStoreOpcode::LOADB.with_default_offset(), &dec_insn)
    }

    fn process_lh(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(Rv32LoadStoreOpcode::LOADH.with_default_offset(), &dec_insn)
    }

    fn process_lw(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(Rv32LoadStoreOpcode::LOADW.with_default_offset(), &dec_insn)
    }

    fn process_lbu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(Rv32LoadStoreOpcode::LOADBU.with_default_offset(), &dec_insn)
    }

    fn process_lhu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(Rv32LoadStoreOpcode::LOADHU.with_default_offset(), &dec_insn)
    }

    fn process_sb(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(Rv32LoadStoreOpcode::STOREB.with_default_offset(), &dec_insn)
    }

    fn process_sh(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(Rv32LoadStoreOpcode::STOREH.with_default_offset(), &dec_insn)
    }

    fn process_sw(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(Rv32LoadStoreOpcode::STOREW.with_default_offset(), &dec_insn)
    }

    fn process_beq(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(BranchEqualOpcode::BEQ.with_default_offset(), &dec_insn)
    }

    fn process_bne(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(BranchEqualOpcode::BNE.with_default_offset(), &dec_insn)
    }

    fn process_blt(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(BranchLessThanOpcode::BLT.with_default_offset(), &dec_insn)
    }

    fn process_bge(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(BranchLessThanOpcode::BGE.with_default_offset(), &dec_insn)
    }

    fn process_bltu(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(BranchLessThanOpcode::BLTU.with_default_offset(), &dec_insn)
    }

    fn process_bgeu(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(BranchLessThanOpcode::BGEU.with_default_offset(), &dec_insn)
    }

    fn process_jal(&mut self, dec_insn: JType) -> Self::InstructionResult {
        from_j_type(Rv32JalLuiOpcode::JAL.with_default_offset(), &dec_insn)
    }

    fn process_jalr(&mut self, dec_insn: IType) -> Self::InstructionResult {
        Instruction::new(
            Rv32JalrOpcode::JALR.with_default_offset(),
            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
            F::from_canonical_u32((dec_insn.imm as u32) & 0xffff),
            F::one(),
            F::zero(),
            F::from_bool(dec_insn.rd != 0),
            F::zero(),
        )
    }

    fn process_lui(&mut self, dec_insn: UType) -> Self::InstructionResult {
        if dec_insn.rd == 0 {
            return nop();
        }
        // we need to set f to 1 because this is handled by the same chip as jal
        let mut result = from_u_type(Rv32JalLuiOpcode::LUI.with_default_offset(), &dec_insn);
        result.f = F::one();
        result
    }

    fn process_auipc(&mut self, dec_insn: UType) -> Self::InstructionResult {
        if dec_insn.rd == 0 {
            return nop();
        }
        Instruction::new(
            Rv32AuipcOpcode::AUIPC.with_default_offset(),
            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
            F::zero(),
            F::from_canonical_u32(((dec_insn.imm as u32) & 0xfffff000) >> 8),
            F::one(), // rd is a register
            F::zero(),
            F::zero(),
            F::zero(),
        )
    }

    fn process_mul(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(MulOpcode::MUL.with_default_offset(), 0, &dec_insn)
    }

    fn process_mulh(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(MulHOpcode::MULH.with_default_offset(), 0, &dec_insn)
    }

    fn process_mulhu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(MulHOpcode::MULHU.with_default_offset(), 0, &dec_insn)
    }

    fn process_mulhsu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(MulHOpcode::MULHSU.with_default_offset(), 0, &dec_insn)
    }

    fn process_div(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(DivRemOpcode::DIV.with_default_offset(), 0, &dec_insn)
    }

    fn process_divu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(DivRemOpcode::DIVU.with_default_offset(), 0, &dec_insn)
    }

    fn process_rem(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(DivRemOpcode::REM.with_default_offset(), 0, &dec_insn)
    }

    fn process_remu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(DivRemOpcode::REMU.with_default_offset(), 0, &dec_insn)
    }

    fn process_fence(&mut self, dec_insn: IType) -> Self::InstructionResult {
        eprintln!("trying to transpile fence ({:?})", dec_insn);
        nop()
    }
}

fn process_custom_instruction<F: PrimeField32>(instruction_u32: u32) -> Instruction<F> {
    let opcode = (instruction_u32 & 0x7f) as u8;
    let funct3 = ((instruction_u32 >> 12) & 0b111) as u8; // All our instructions are R- or I-type

    match opcode {
        CUSTOM_0 => match funct3 {
            0b000 => {
                let imm = (instruction_u32 >> 20) & 0xfff;
                Some(terminate(imm.try_into().expect("exit code must be byte")))
            }
            0b001 => {
                let rd = (instruction_u32 >> 7) & 0x1f;
                let imm = (instruction_u32 >> 20) & 0xfff;
                Some(Instruction::from_isize(
                    Rv32HintStoreOpcode::HINT_STOREW.with_default_offset(),
                    0,
                    (RV32_REGISTER_NUM_LIMBS * rd as usize) as isize,
                    imm as isize,
                    1,
                    2,
                ))
            }
            0b011 => Some(Instruction::phantom(
                PhantomInstruction::HintInputRv32,
                F::zero(),
                F::zero(),
                0,
            )),
            _ => unimplemented!(),
        },
        CUSTOM_1 => {
            match funct3 {
                Rv32ModularArithmeticOpcode::FUNCT3 => {
                    // mod operations
                    let funct7 = (instruction_u32 >> 25) & 0x7f;
                    let size = Rv32ModularArithmeticOpcode::COUNT as u32;
                    let prime_idx = funct7 / size;
                    let local_opcode_idx = funct7 % size;
                    let global_opcode_idx = (local_opcode_idx + prime_idx * size) as usize
                        + Rv32ModularArithmeticOpcode::default_offset();
                    Some(from_r_type(
                        global_opcode_idx,
                        2,
                        &RType::new(instruction_u32),
                    ))
                }
                EccOpcode::FUNCT3 => {
                    // short weierstrass ec
                    let funct7 = (instruction_u32 >> 25) & 0x7f;
                    let size = EccOpcode::COUNT as u32;
                    let prime_idx = funct7 / size;
                    let local_opcode_idx = funct7 % size;
                    let global_opcode_idx = (local_opcode_idx + prime_idx * size) as usize
                        + EccOpcode::default_offset();
                    Some(from_r_type(
                        global_opcode_idx,
                        2,
                        &RType::new(instruction_u32),
                    ))
                }
                _ => None,
            }
        }
        _ => None,
    }
    .unwrap_or_else(|| {
        panic!(
            "Failed to transpile custom instruction: {:b} (opcode = {:07b}, funct3 = {:03b})",
            instruction_u32, opcode, funct3
        )
    })
}

/// Transpile the [`Instruction`]s from the 32-bit encoded instructions.
///
/// # Panics
///
/// This function will return an error if the [`Instruction`] cannot be processed.
pub(crate) fn transpile<F: PrimeField32>(instructions_u32: &[u32]) -> Vec<Instruction<F>> {
    let mut instructions = Vec::new();
    let mut transpiler = InstructionTranspiler::<F>(PhantomData);
    for instruction_u32 in instructions_u32 {
        assert!(*instruction_u32 != 115, "ecall is not supported");
        let instruction = process_instruction(&mut transpiler, *instruction_u32)
            .unwrap_or_else(|| process_custom_instruction(*instruction_u32));
        instructions.push(instruction);
    }
    instructions
}
