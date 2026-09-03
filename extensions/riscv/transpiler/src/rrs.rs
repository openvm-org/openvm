use openvm_decoder::{
    instruction_formats::{BType, IType, ITypeShamt, JType, RType, SType, UType},
    InstructionProcessor,
};
use openvm_instructions::{
    instruction::{Instruction, InstructionOperand},
    riscv::REGISTER_NUM_LIMBS,
    *,
};
use openvm_transpiler::util::{
    from_b_type, from_i_type, from_i_type_shamt, from_j_type, from_load, from_r_type, from_s_type,
    from_u_type, nop,
};

use crate::{
    AuipcOpcode, BaseAluImmOpcode, BaseAluOpcode, BaseAluWImmOpcode, BaseAluWOpcode,
    BranchEqualOpcode, BranchLessThanOpcode, DivRemOpcode, DivRemWOpcode, JalLuiOpcode, JalrOpcode,
    LessThanImmOpcode, LessThanOpcode, LoadStoreOpcode, MulHOpcode, MulOpcode, MulWOpcode,
    ShiftImmOpcode, ShiftOpcode, ShiftWImmOpcode, ShiftWOpcode,
};

/// A transpiler that converts the 32-bit encoded instructions into instructions.
pub(crate) struct InstructionTranspiler;

impl InstructionProcessor for InstructionTranspiler {
    type InstructionResult = Instruction;

    fn process_add(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            BaseAluOpcode::ADD.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_addi(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(BaseAluImmOpcode::ADDI.global_opcode().as_usize(), &dec_insn)
    }

    fn process_sub(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            BaseAluOpcode::SUB.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_xor(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            BaseAluOpcode::XOR.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_xori(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(BaseAluImmOpcode::XORI.global_opcode().as_usize(), &dec_insn)
    }

    fn process_or(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            BaseAluOpcode::OR.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_ori(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(BaseAluImmOpcode::ORI.global_opcode().as_usize(), &dec_insn)
    }

    fn process_and(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            BaseAluOpcode::AND.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_andi(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(BaseAluImmOpcode::ANDI.global_opcode().as_usize(), &dec_insn)
    }

    fn process_sll(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            ShiftOpcode::SLL.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_slli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(ShiftImmOpcode::SLLI.global_opcode().as_usize(), &dec_insn)
    }

    fn process_srl(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            ShiftOpcode::SRL.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_srli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(ShiftImmOpcode::SRLI.global_opcode().as_usize(), &dec_insn)
    }

    fn process_sra(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            ShiftOpcode::SRA.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_srai(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(ShiftImmOpcode::SRAI.global_opcode().as_usize(), &dec_insn)
    }

    fn process_slt(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            LessThanOpcode::SLT.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_slti(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(
            LessThanImmOpcode::SLTI.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_sltu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            LessThanOpcode::SLTU.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_sltui(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(
            LessThanImmOpcode::SLTIU.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_lb(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(LoadStoreOpcode::LOADB.global_opcode().as_usize(), &dec_insn)
    }

    fn process_lh(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(LoadStoreOpcode::LOADH.global_opcode().as_usize(), &dec_insn)
    }

    fn process_lw(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(LoadStoreOpcode::LOADW.global_opcode().as_usize(), &dec_insn)
    }

    fn process_lbu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(
            LoadStoreOpcode::LOADBU.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_lhu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(
            LoadStoreOpcode::LOADHU.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_lwu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(
            LoadStoreOpcode::LOADWU.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_ld(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(LoadStoreOpcode::LOADD.global_opcode().as_usize(), &dec_insn)
    }

    fn process_sb(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(
            LoadStoreOpcode::STOREB.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_sh(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(
            LoadStoreOpcode::STOREH.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_sw(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(
            LoadStoreOpcode::STOREW.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_sd(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(
            LoadStoreOpcode::STORED.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_beq(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(BranchEqualOpcode::BEQ.global_opcode().as_usize(), &dec_insn)
    }

    fn process_bne(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(BranchEqualOpcode::BNE.global_opcode().as_usize(), &dec_insn)
    }

    fn process_blt(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(
            BranchLessThanOpcode::BLT.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_bge(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(
            BranchLessThanOpcode::BGE.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_bltu(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(
            BranchLessThanOpcode::BLTU.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_bgeu(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type(
            BranchLessThanOpcode::BGEU.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_jal(&mut self, dec_insn: JType) -> Self::InstructionResult {
        from_j_type(JalLuiOpcode::JAL.global_opcode().as_usize(), &dec_insn)
    }

    fn process_jalr(&mut self, dec_insn: IType) -> Self::InstructionResult {
        Instruction::new(
            JalrOpcode::JALR.global_opcode(),
            InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rd),
            InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs1),
            InstructionOperand::from_u32((dec_insn.imm as u32) & 0xffff),
            InstructionOperand::ONE,
            InstructionOperand::ZERO,
            dec_insn.rd != 0,
            dec_insn.imm < 0,
        )
    }

    fn process_lui(&mut self, dec_insn: UType) -> Self::InstructionResult {
        if dec_insn.rd == 0 {
            return nop();
        }
        // we need to set f to 1 because this is handled by the same chip as jal
        let mut result = from_u_type(JalLuiOpcode::LUI.global_opcode().as_usize(), &dec_insn);
        result.f = InstructionOperand::ONE;
        result
    }

    fn process_auipc(&mut self, dec_insn: UType) -> Self::InstructionResult {
        if dec_insn.rd == 0 {
            return nop();
        }
        Instruction::new(
            AuipcOpcode::AUIPC.global_opcode(),
            InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rd),
            InstructionOperand::ZERO,
            InstructionOperand::from_u32(((dec_insn.imm as u32) & 0xfffff000) >> 8),
            InstructionOperand::ONE, // rd is a register
            InstructionOperand::ZERO,
            InstructionOperand::ZERO,
            InstructionOperand::ZERO,
        )
    }

    fn process_mul(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            MulOpcode::MUL.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_mulh(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            MulHOpcode::MULH.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_mulhu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            MulHOpcode::MULHU.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_mulhsu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            MulHOpcode::MULHSU.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_div(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            DivRemOpcode::DIV.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_divu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            DivRemOpcode::DIVU.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_rem(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            DivRemOpcode::REM.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_remu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            DivRemOpcode::REMU.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    // RV64-specific instructions

    fn process_addw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            BaseAluWOpcode::ADDW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_subw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            BaseAluWOpcode::SUBW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_addiw(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type(
            BaseAluWImmOpcode::ADDIW.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_sllw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            ShiftWOpcode::SLLW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_srlw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            ShiftWOpcode::SRLW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_sraw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            ShiftWOpcode::SRAW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_slliw(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(ShiftWImmOpcode::SLLIW.global_opcode().as_usize(), &dec_insn)
    }

    fn process_srliw(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(ShiftWImmOpcode::SRLIW.global_opcode().as_usize(), &dec_insn)
    }

    fn process_sraiw(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(ShiftWImmOpcode::SRAIW.global_opcode().as_usize(), &dec_insn)
    }

    fn process_mulw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            MulWOpcode::MULW.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_divw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            DivRemWOpcode::DIVW.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_divuw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            DivRemWOpcode::DIVUW.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_remw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            DivRemWOpcode::REMW.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_remuw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            DivRemWOpcode::REMUW.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_fence(&mut self, dec_insn: IType) -> Self::InstructionResult {
        tracing::debug!("Transpiling fence ({:?}) to nop", dec_insn);
        nop()
    }
}
