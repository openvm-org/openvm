use std::marker::PhantomData;

use axvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, *};
use axvm_transpiler::util::{
    from_b_type, from_i_type, from_i_type_shamt, from_j_type, from_load, from_r_type, from_s_type,
    from_u_type, nop,
};
use p3_field::PrimeField32;
use rrs_lib::{
    instruction_formats::{BType, IType, ITypeShamt, JType, RType, SType, UType},
    InstructionProcessor,
};

/// A transpiler that converts the 32-bit encoded instructions into instructions.
pub(crate) struct InstructionTranspiler<F>(pub PhantomData<F>);

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
            F::ONE,
            F::ZERO,
            F::from_bool(dec_insn.rd != 0),
            F::ZERO,
        )
    }

    fn process_lui(&mut self, dec_insn: UType) -> Self::InstructionResult {
        if dec_insn.rd == 0 {
            return nop();
        }
        // we need to set f to 1 because this is handled by the same chip as jal
        let mut result = from_u_type(Rv32JalLuiOpcode::LUI.with_default_offset(), &dec_insn);
        result.f = F::ONE;
        result
    }

    fn process_auipc(&mut self, dec_insn: UType) -> Self::InstructionResult {
        if dec_insn.rd == 0 {
            return nop();
        }
        Instruction::new(
            Rv32AuipcOpcode::AUIPC.with_default_offset(),
            F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
            F::ZERO,
            F::from_canonical_u32(((dec_insn.imm as u32) & 0xfffff000) >> 8),
            F::ONE, // rd is a register
            F::ZERO,
            F::ZERO,
            F::ZERO,
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
        tracing::debug!("Transpiling fence ({:?}) to nop", dec_insn);
        nop()
    }
}
