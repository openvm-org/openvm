use std::marker::PhantomData;

use p3_field::PrimeField32;
use rrs_lib::{
    instruction_formats::{BType, IType, ITypeShamt, JType, RType, SType, UType},
    process_instruction, InstructionProcessor,
};
use stark_vm::{
    arch::instructions::{
        AluOpcode, CoreOpcode, DivRemOpcode, LessThanOpcode, MulHOpcode, MulOpcode,
        Rv32LoadStoreOpcode, ShiftOpcode, UnimplementedOpcode, UsizeOpcode,
    },
    program::Instruction,
};

use crate::util::*;

/// A transpiler that converts the 32-bit encoded instructions into instructions.
#[allow(dead_code)]
pub(crate) struct InstructionTranspiler<F>(PhantomData<F>);

impl<F: PrimeField32> InstructionProcessor for InstructionTranspiler<F> {
    type InstructionResult = Instruction<F>;

    fn process_add(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(AluOpcode::ADD.with_default_offset(), &dec_insn)
    }

    fn process_addi(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(AluOpcode::ADD.with_default_offset(), &dec_insn)
    }

    fn process_sub(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(AluOpcode::SUB.with_default_offset(), &dec_insn)
    }

    fn process_xor(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(AluOpcode::XOR.with_default_offset(), &dec_insn)
    }

    fn process_xori(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(AluOpcode::XOR.with_default_offset(), &dec_insn)
    }

    fn process_or(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(AluOpcode::OR.with_default_offset(), &dec_insn)
    }

    fn process_ori(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(AluOpcode::OR.with_default_offset(), &dec_insn)
    }

    fn process_and(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(AluOpcode::AND.with_default_offset(), &dec_insn)
    }

    fn process_andi(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(AluOpcode::AND.with_default_offset(), &dec_insn)
    }

    fn process_sll(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(ShiftOpcode::SLL.with_default_offset(), &dec_insn)
    }

    fn process_slli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(ShiftOpcode::SLL.with_default_offset(), &dec_insn)
    }

    fn process_srl(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(ShiftOpcode::SRL.with_default_offset(), &dec_insn)
    }

    fn process_srli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(ShiftOpcode::SRL.with_default_offset(), &dec_insn)
    }

    fn process_sra(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(ShiftOpcode::SRA.with_default_offset(), &dec_insn)
    }

    fn process_srai(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(ShiftOpcode::SRA.with_default_offset(), &dec_insn)
    }

    fn process_slt(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(LessThanOpcode::SLT.with_default_offset(), &dec_insn)
    }

    fn process_slti(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(LessThanOpcode::SLT.with_default_offset(), &dec_insn)
    }

    fn process_sltu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(LessThanOpcode::SLTU.with_default_offset(), &dec_insn)
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
        from_b_type(CoreOpcode::BEQ.with_default_offset(), &dec_insn)
    }

    fn process_bne(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(CoreOpcode::BNE.with_default_offset(), &dec_insn)
    }

    fn process_blt(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(
            UnimplementedOpcode::BLT_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_bge(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(
            UnimplementedOpcode::BGE_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_bltu(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(
            UnimplementedOpcode::BLTU_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_bgeu(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(
            UnimplementedOpcode::BGEU_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_jal(&mut self, dec_insn: JType) -> Self::InstructionResult {
        from_j_type(
            UnimplementedOpcode::JAL_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_jalr(&mut self, dec_insn: IType) -> Self::InstructionResult {
        let imm = dec_insn.imm / 2;
        Instruction::new(
            UnimplementedOpcode::JALR_RV32.with_default_offset(),
            F::from_canonical_usize(dec_insn.rd),
            F::from_canonical_usize(dec_insn.rs1),
            if imm < 0 {
                -F::from_canonical_u32((-imm) as u32)
            } else {
                F::from_canonical_u32(imm as u32)
            },
            F::one(),
            F::zero(),
            F::zero(),
            F::zero(),
            String::new(),
        )
    }

    fn process_lui(&mut self, dec_insn: UType) -> Self::InstructionResult {
        from_u_type(
            UnimplementedOpcode::LUI_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_auipc(&mut self, dec_insn: UType) -> Self::InstructionResult {
        Instruction::new(
            UnimplementedOpcode::AUIPC_RV32.with_default_offset(),
            F::from_canonical_usize(dec_insn.rd),
            F::zero(),
            F::from_canonical_u32(((dec_insn.imm as u32) & 0xfffff) << 4),
            F::one(), // rd is a register
            F::zero(),
            F::zero(),
            F::zero(),
            String::new(),
        )
    }

    fn process_mul(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(MulOpcode::MUL.with_default_offset(), &dec_insn)
    }

    fn process_mulh(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(MulHOpcode::MULH.with_default_offset(), &dec_insn)
    }

    fn process_mulhu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(MulHOpcode::MULHU.with_default_offset(), &dec_insn)
    }

    fn process_mulhsu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(MulHOpcode::MULHSU.with_default_offset(), &dec_insn)
    }

    fn process_div(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(DivRemOpcode::DIV.with_default_offset(), &dec_insn)
    }

    fn process_divu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(DivRemOpcode::DIVU.with_default_offset(), &dec_insn)
    }

    fn process_rem(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(DivRemOpcode::REM.with_default_offset(), &dec_insn)
    }

    fn process_remu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(DivRemOpcode::REMU.with_default_offset(), &dec_insn)
    }

    fn process_fence(&mut self, dec_insn: IType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        Instruction {
            debug: format!("fence({:?})", dec_insn),
            ..unimp()
        }
    }
}

/// Transpile the [`Instruction`]s from the 32-bit encoded instructions.
///
/// # Panics
///
/// This function will return an error if the [`Instruction`] cannot be processed.
#[allow(dead_code)]
pub(crate) fn transpile<F: PrimeField32>(instructions_u32: &[u32]) -> Vec<Instruction<F>> {
    let mut instructions = Vec::new();
    let mut transpiler = InstructionTranspiler::<F>(PhantomData);
    for instruction_u32 in instructions_u32 {
        // TODO: we probably want to forbid such instructions, but for now we just skip them
        if *instruction_u32 == 115 {
            instructions.push(unimp());
            continue;
        }
        let instruction = process_instruction(&mut transpiler, *instruction_u32).unwrap();
        instructions.push(instruction);
    }
    instructions
}
