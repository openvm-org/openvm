use std::marker::PhantomData;

use openvm_instructions::{instruction::Instruction, riscv::RV64_REGISTER_NUM_LIMBS, *};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::{
    decoder::{BType, IType, ITypeShamt, InstructionProcessor, JType, RType, SType, UType},
    util::{
        from_b_type_rv64, from_i_type_rv64, from_i_type_shamt_rv64, from_j_type_rv64,
        from_load_rv64, from_r_type_rv64, from_s_type_rv64, from_u_type_rv64, nop,
    },
};

use crate::{
    Rv64AuipcOpcode, Rv64BaseAluOpcode, Rv64BaseAluWOpcode, Rv64BranchEqualOpcode,
    Rv64BranchLessThanOpcode, Rv64DivRemOpcode, Rv64DivRemWOpcode, Rv64JalLuiOpcode,
    Rv64JalrOpcode, Rv64LessThanOpcode, Rv64LoadStoreOpcode, Rv64MulHOpcode, Rv64MulOpcode,
    Rv64MulWOpcode, Rv64ShiftOpcode, Rv64ShiftWOpcode,
};

/// A transpiler that converts the 32-bit encoded instructions into OpenVM instructions (RV64).
pub(crate) struct InstructionTranspiler<F>(pub PhantomData<F>);

impl<F: PrimeField32> InstructionProcessor for InstructionTranspiler<F> {
    type InstructionResult = Instruction<F>;

    fn process_add(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64BaseAluOpcode::ADD.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_addi(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type_rv64(Rv64BaseAluOpcode::ADD.global_opcode().as_usize(), &dec_insn)
    }

    fn process_sub(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64BaseAluOpcode::SUB.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_xor(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64BaseAluOpcode::XOR.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_xori(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type_rv64(Rv64BaseAluOpcode::XOR.global_opcode().as_usize(), &dec_insn)
    }

    fn process_or(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64BaseAluOpcode::OR.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_ori(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type_rv64(Rv64BaseAluOpcode::OR.global_opcode().as_usize(), &dec_insn)
    }

    fn process_and(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64BaseAluOpcode::AND.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_andi(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type_rv64(Rv64BaseAluOpcode::AND.global_opcode().as_usize(), &dec_insn)
    }

    fn process_sll(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64ShiftOpcode::SLL.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_slli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt_rv64(Rv64ShiftOpcode::SLL.global_opcode().as_usize(), &dec_insn)
    }

    fn process_srl(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64ShiftOpcode::SRL.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_srli(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt_rv64(Rv64ShiftOpcode::SRL.global_opcode().as_usize(), &dec_insn)
    }

    fn process_sra(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64ShiftOpcode::SRA.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_srai(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt_rv64(Rv64ShiftOpcode::SRA.global_opcode().as_usize(), &dec_insn)
    }

    fn process_slt(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64LessThanOpcode::SLT.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_slti(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type_rv64(
            Rv64LessThanOpcode::SLT.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_sltu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64LessThanOpcode::SLTU.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_sltui(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type_rv64(
            Rv64LessThanOpcode::SLTU.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_lb(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load_rv64(
            Rv64LoadStoreOpcode::LOADB.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_lh(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load_rv64(
            Rv64LoadStoreOpcode::LOADH.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_lw(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load_rv64(
            Rv64LoadStoreOpcode::LOADW.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_lbu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load_rv64(
            Rv64LoadStoreOpcode::LOADBU.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_lhu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load_rv64(
            Rv64LoadStoreOpcode::LOADHU.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_lwu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load_rv64(
            Rv64LoadStoreOpcode::LOADWU.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_ld(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load_rv64(
            Rv64LoadStoreOpcode::LOADD.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_sb(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type_rv64(
            Rv64LoadStoreOpcode::STOREB.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_sh(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type_rv64(
            Rv64LoadStoreOpcode::STOREH.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_sw(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type_rv64(
            Rv64LoadStoreOpcode::STOREW.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_sd(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type_rv64(
            Rv64LoadStoreOpcode::STORED.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_beq(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type_rv64(
            Rv64BranchEqualOpcode::BEQ.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_bne(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type_rv64(
            Rv64BranchEqualOpcode::BNE.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_blt(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type_rv64(
            Rv64BranchLessThanOpcode::BLT.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_bge(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type_rv64(
            Rv64BranchLessThanOpcode::BGE.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_bltu(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type_rv64(
            Rv64BranchLessThanOpcode::BLTU.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_bgeu(&mut self, dec_insn: BType) -> Self::InstructionResult {
        from_b_type_rv64(
            Rv64BranchLessThanOpcode::BGEU.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_jal(&mut self, dec_insn: JType) -> Self::InstructionResult {
        from_j_type_rv64(Rv64JalLuiOpcode::JAL.global_opcode().as_usize(), &dec_insn)
    }

    fn process_jalr(&mut self, dec_insn: IType) -> Self::InstructionResult {
        Instruction::new(
            Rv64JalrOpcode::JALR.global_opcode(),
            F::from_canonical_usize(RV64_REGISTER_NUM_LIMBS * dec_insn.rd),
            F::from_canonical_usize(RV64_REGISTER_NUM_LIMBS * dec_insn.rs1),
            F::from_canonical_u32((dec_insn.imm as u32) & 0xffff),
            F::ONE,
            F::ZERO,
            F::from_bool(dec_insn.rd != 0),
            F::from_bool(dec_insn.imm < 0),
        )
    }

    fn process_lui(&mut self, dec_insn: UType) -> Self::InstructionResult {
        if dec_insn.rd == 0 {
            return nop();
        }
        let mut result =
            from_u_type_rv64(Rv64JalLuiOpcode::LUI.global_opcode().as_usize(), &dec_insn);
        result.f = F::ONE;
        result
    }

    fn process_auipc(&mut self, dec_insn: UType) -> Self::InstructionResult {
        if dec_insn.rd == 0 {
            return nop();
        }
        Instruction::new(
            Rv64AuipcOpcode::AUIPC.global_opcode(),
            F::from_canonical_usize(RV64_REGISTER_NUM_LIMBS * dec_insn.rd),
            F::ZERO,
            F::from_canonical_u32(((dec_insn.imm as u32) & 0xfffff000) >> 8),
            F::ONE,
            F::ZERO,
            F::ZERO,
            F::ZERO,
        )
    }

    fn process_mul(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64MulOpcode::MUL.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_mulh(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64MulHOpcode::MULH.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_mulhu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64MulHOpcode::MULHU.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_mulhsu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64MulHOpcode::MULHSU.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_div(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64DivRemOpcode::DIV.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_divu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64DivRemOpcode::DIVU.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_rem(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64DivRemOpcode::REM.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_remu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64DivRemOpcode::REMU.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_fence(&mut self, dec_insn: IType) -> Self::InstructionResult {
        tracing::debug!("Transpiling fence ({:?}) to nop", dec_insn);
        nop()
    }

    // ── RV64 OP-32 ───────────────────────────────────────────────────────

    fn process_addw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64BaseAluWOpcode::ADDW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_subw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64BaseAluWOpcode::SUBW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_sllw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64ShiftWOpcode::SLLW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_srlw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64ShiftWOpcode::SRLW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_sraw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64ShiftWOpcode::SRAW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    // ── RV64 OP-IMM-32 ──────────────────────────────────────────────────

    fn process_addiw(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_i_type_rv64(
            Rv64BaseAluWOpcode::ADDW.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_slliw(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt_rv64(Rv64ShiftWOpcode::SLLW.global_opcode().as_usize(), &dec_insn)
    }

    fn process_srliw(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt_rv64(Rv64ShiftWOpcode::SRLW.global_opcode().as_usize(), &dec_insn)
    }

    fn process_sraiw(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt_rv64(Rv64ShiftWOpcode::SRAW.global_opcode().as_usize(), &dec_insn)
    }

    // ── RV64M OP-32 ─────────────────────────────────────────────────────

    fn process_mulw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64MulWOpcode::MULW.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_divw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64DivRemWOpcode::DIVW.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_divuw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64DivRemWOpcode::DIVUW.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_remw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64DivRemWOpcode::REMW.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }

    fn process_remuw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type_rv64(
            Rv64DivRemWOpcode::REMUW.global_opcode().as_usize(),
            0,
            &dec_insn,
            false,
        )
    }
}
