use std::marker::PhantomData;

use p3_field::PrimeField32;
use rrs_lib::{
    instruction_formats::{BType, IType, ITypeShamt, JType, RType, SType, UType},
    process_instruction, InstructionProcessor,
};
use stark_vm::{
    arch::instructions::{CoreOpcode, U256Opcode, UsizeOpcode},
    program::Instruction,
};

use crate::util::*;

/// A transpiler that converts the 32-bit encoded instructions into instructions.
pub(crate) struct InstructionTranspiler<F>(PhantomData<F>);

fn unimp<F: PrimeField32>() -> Instruction<F> {
    Default::default()
}

impl<F: PrimeField32> InstructionProcessor for InstructionTranspiler<F> {
    type InstructionResult = Instruction<F>;

    fn process_add(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(U256Opcode::ADD.with_default_offset(), &dec_insn)
    }

    fn process_addi(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(U256Opcode::ADD.with_default_offset(), &dec_insn)
    }

    fn process_sub(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(U256Opcode::SUB.with_default_offset(), &dec_insn)
    }

    fn process_xor(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(U256Opcode::XOR.with_default_offset(), &dec_insn)
    }

    fn process_xori(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(U256Opcode::XOR.with_default_offset(), &dec_insn)
    }

    fn process_or(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(U256Opcode::OR.with_default_offset(), &dec_insn)
    }

    fn process_ori(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(U256Opcode::OR.with_default_offset(), &dec_insn)
    }

    fn process_and(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(U256Opcode::AND.with_default_offset(), &dec_insn)
    }

    fn process_andi(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(U256Opcode::AND.with_default_offset(), &dec_insn)
    }

    fn process_sll(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(U256Opcode::SLL.with_default_offset(), &dec_insn)
    }

    fn process_slli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(U256Opcode::SLL.with_default_offset(), &dec_insn)
    }

    fn process_srl(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(U256Opcode::SRL.with_default_offset(), &dec_insn)
    }

    fn process_srli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(U256Opcode::SRL.with_default_offset(), &dec_insn)
    }

    fn process_sra(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(U256Opcode::SRA.with_default_offset(), &dec_insn)
    }

    fn process_srai(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(U256Opcode::SRA.with_default_offset(), &dec_insn)
    }

    fn process_slt(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(U256Opcode::SLT.with_default_offset(), &dec_insn)
    }

    fn process_slti(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(U256Opcode::SLT.with_default_offset(), &dec_insn)
    }

    fn process_sltu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(U256Opcode::LT.with_default_offset(), &dec_insn)
    }

    fn process_sltui(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(U256Opcode::LT.with_default_offset(), &dec_insn)
    }

    fn process_lb(&mut self, dec_insn: IType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_lh(&mut self, dec_insn: IType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_lw(&mut self, dec_insn: IType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_lbu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_lhu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_sb(&mut self, dec_insn: SType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_sh(&mut self, dec_insn: SType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_sw(&mut self, dec_insn: SType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_beq(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(CoreOpcode::BEQ.with_default_offset(), &dec_insn)
    }

    fn process_bne(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(CoreOpcode::BNE.with_default_offset(), &dec_insn)
    }

    fn process_blt(&mut self, dec_insn: BType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_bge(&mut self, dec_insn: BType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_bltu(&mut self, dec_insn: BType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_bgeu(&mut self, dec_insn: BType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_jal(&mut self, dec_insn: JType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_jalr(&mut self, dec_insn: IType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_lui(&mut self, dec_insn: UType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_auipc(&mut self, dec_insn: UType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_mul(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(U256Opcode::MUL.with_default_offset(), &dec_insn)
    }

    fn process_mulh(&mut self, dec_insn: RType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_mulhu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_mulhsu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_div(&mut self, dec_insn: RType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_divu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_rem(&mut self, dec_insn: RType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_remu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        let _ = dec_insn;
        // unimplemented!()
        unimp()
    }

    fn process_fence(&mut self, _: IType) -> Self::InstructionResult {
        // unimplemented!()
        unimp()
    }
}

/// Transpile the [`Instruction`]s from the 32-bit encoded instructions.
///
/// # Panics
///
/// This function will return an error if the [`Instruction`] cannot be processed.
#[must_use]
pub(crate) fn transpile<F: PrimeField32>(instructions_u32: &[u32]) -> Vec<Instruction<F>> {
    let mut instructions = Vec::new();
    let mut transpiler = InstructionTranspiler::<F>(PhantomData);
    for instruction_u32 in instructions_u32 {
        // TODO: we probably want to forbid such instructions, but for now we just skip them
        if *instruction_u32 == 115 {
            continue;
        }
        let instruction = process_instruction(&mut transpiler, *instruction_u32).unwrap();
        instructions.push(instruction);
    }
    instructions
}
