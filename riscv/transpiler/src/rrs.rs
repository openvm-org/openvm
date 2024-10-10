use std::marker::PhantomData;

use p3_field::PrimeField32;
use rrs_lib::{
    instruction_formats::{BType, IType, ITypeShamt, JType, RType, SType, UType},
    process_instruction, InstructionProcessor,
};
use stark_vm::{
    arch::instructions::{CoreOpcode, UnimplementedOpcode, UsizeOpcode},
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
        from_r_type(
            UnimplementedOpcode::ADD_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_addi(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(
            UnimplementedOpcode::ADD_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_sub(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::SUB_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_xor(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::XOR_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_xori(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(
            UnimplementedOpcode::XOR_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_or(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::OR_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_ori(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(
            UnimplementedOpcode::OR_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_and(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::AND_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_andi(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(
            UnimplementedOpcode::AND_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_sll(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::SLL_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_slli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(
            UnimplementedOpcode::SLL_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_srl(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::SRL_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_srli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(
            UnimplementedOpcode::SRL_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_sra(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::SRA_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_srai(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(
            UnimplementedOpcode::SRA_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_slt(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::SLT_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_slti(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(
            UnimplementedOpcode::SLT_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_sltu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::SLTU_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_sltui(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(
            UnimplementedOpcode::SLTU_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_lb(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(
            UnimplementedOpcode::LOADB_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_lh(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(
            UnimplementedOpcode::LOADH_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_lw(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(
            UnimplementedOpcode::LOADW_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_lbu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(
            UnimplementedOpcode::LOADBU_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_lhu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(
            UnimplementedOpcode::LOADHU_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_sb(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(
            UnimplementedOpcode::STOREB_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_sh(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(
            UnimplementedOpcode::STOREH_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_sw(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(
            UnimplementedOpcode::STOREW_RV32.with_default_offset(),
            &dec_insn,
        )
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
        from_u_type(
            UnimplementedOpcode::AUIPC_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_mul(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::MUL_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_mulh(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::MULH_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_mulhu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::MULHU_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_mulhsu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::MULHSU_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_div(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::DIV_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_divu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::DIVU_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_rem(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::REM_RV32.with_default_offset(),
            &dec_insn,
        )
    }

    fn process_remu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            UnimplementedOpcode::REMU_RV32.with_default_offset(),
            &dec_insn,
        )
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
#[must_use]
pub(crate) fn transpile<F: PrimeField32>(instructions_u32: &[u32]) -> Vec<Instruction<F>> {
    let mut instructions = Vec::new();
    let mut transpiler = InstructionTranspiler::<F>(PhantomData);
    for instruction_u32 in instructions_u32 {
        // TODO: we probably want to forbid such instructions, but for now we just skip them
        if *instruction_u32 == 115 {
            instructions.push(unimp());
            continue;
        }
        println!("instruction_u32: {:032b}", instruction_u32);
        let instruction = process_instruction(&mut transpiler, *instruction_u32).unwrap();
        instructions.push(instruction);
    }
    instructions
}
