use std::marker::PhantomData;

use openvm_decoder::{
    instruction_formats::{BType, IType, ITypeShamt, JType, RType, SType, UType},
    InstructionProcessor,
};
use openvm_instructions::{instruction::Instruction, riscv::RV64_REGISTER_NUM_LIMBS, *};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_transpiler::util::{
    from_b_type, from_i_type, from_i_type_shamt, from_j_type, from_load, from_r_type, from_s_type,
    from_u_type, nop,
};

use crate::{
    BaseAluImmOpcode, BaseAluOpcode, BaseAluWImmOpcode, BaseAluWOpcode, BitwiseInvOpcode,
    BranchEqualOpcode, BranchLessThanOpcode, ByteUnaryOpcode, CountZerosOpcode, CountZerosWOpcode,
    CpopOpcode, CpopWOpcode, DivRemOpcode, DivRemWOpcode, LessThanImmOpcode, LessThanOpcode,
    MinMaxOpcode, MulHOpcode, MulOpcode, MulWOpcode, RotateImmOpcode, RotateOpcode,
    RotateWImmOpcode, RotateWOpcode, Rv64AuipcOpcode, Rv64JalLuiOpcode, Rv64JalrOpcode,
    Rv64LoadStoreOpcode, ShAddOpcode, ShiftImmOpcode, ShiftOpcode, ShiftWImmOpcode, ShiftWOpcode,
    SingleBitImmOpcode, SingleBitOpcode, SlliUwOpcode,
};

/// A transpiler that converts the 32-bit encoded instructions into instructions.
pub(crate) struct InstructionTranspiler<F>(pub PhantomData<F>);

/// Unary bit-manipulation ops (`rd = f(rs1)`) reuse the ALU-immediate operand
/// layout with a zero immediate, so they can run on the one-register-read
/// immediate adapters. The raw `imm` field of these words carries the funct12
/// sub-operation selector, which must not leak into the operands.
fn from_unary<F: PrimeField32>(opcode: usize, rd: usize, rs1: usize) -> Instruction<F> {
    from_i_type(
        opcode,
        &IType {
            imm: 0,
            rs1,
            funct3: 0,
            rd,
        },
    )
}

impl<F: PrimeField32> InstructionProcessor for InstructionTranspiler<F> {
    type InstructionResult = Instruction<F>;

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
        from_load(
            Rv64LoadStoreOpcode::LOADB.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_lh(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(
            Rv64LoadStoreOpcode::LOADH.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_lw(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(
            Rv64LoadStoreOpcode::LOADW.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_lbu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(
            Rv64LoadStoreOpcode::LOADBU.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_lhu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(
            Rv64LoadStoreOpcode::LOADHU.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_lwu(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(
            Rv64LoadStoreOpcode::LOADWU.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_ld(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_load(
            Rv64LoadStoreOpcode::LOADD.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_sb(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(
            Rv64LoadStoreOpcode::STOREB.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_sh(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(
            Rv64LoadStoreOpcode::STOREH.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_sw(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(
            Rv64LoadStoreOpcode::STOREW.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_sd(&mut self, dec_insn: SType) -> Self::InstructionResult {
        from_s_type(
            Rv64LoadStoreOpcode::STORED.global_opcode().as_usize(),
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
        from_j_type(Rv64JalLuiOpcode::JAL.global_opcode().as_usize(), &dec_insn)
    }

    fn process_jalr(&mut self, dec_insn: IType) -> Self::InstructionResult {
        Instruction::new(
            Rv64JalrOpcode::JALR.global_opcode(),
            F::from_usize(RV64_REGISTER_NUM_LIMBS * dec_insn.rd),
            F::from_usize(RV64_REGISTER_NUM_LIMBS * dec_insn.rs1),
            F::from_u32((dec_insn.imm as u32) & 0xffff),
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
        // we need to set f to 1 because this is handled by the same chip as jal
        let mut result = from_u_type(Rv64JalLuiOpcode::LUI.global_opcode().as_usize(), &dec_insn);
        result.f = F::ONE;
        result
    }

    fn process_auipc(&mut self, dec_insn: UType) -> Self::InstructionResult {
        if dec_insn.rd == 0 {
            return nop();
        }
        Instruction::new(
            Rv64AuipcOpcode::AUIPC.global_opcode(),
            F::from_usize(RV64_REGISTER_NUM_LIMBS * dec_insn.rd),
            F::ZERO,
            F::from_u32(((dec_insn.imm as u32) & 0xfffff000) >> 8),
            F::ONE, // rd is a register
            F::ZERO,
            F::ZERO,
            F::ZERO,
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

    // ---- Zba ----

    fn process_add_uw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            ShAddOpcode::ADD_UW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_sh1add(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            ShAddOpcode::SH1ADD.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_sh2add(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            ShAddOpcode::SH2ADD.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_sh3add(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            ShAddOpcode::SH3ADD.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_sh1add_uw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            ShAddOpcode::SH1ADD_UW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_sh2add_uw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            ShAddOpcode::SH2ADD_UW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_sh3add_uw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            ShAddOpcode::SH3ADD_UW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_slli_uw(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(SlliUwOpcode::SLLI_UW.global_opcode().as_usize(), &dec_insn)
    }

    // ---- Zbb: logical with negate ----

    fn process_andn(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            BitwiseInvOpcode::ANDN.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_orn(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            BitwiseInvOpcode::ORN.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_xnor(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            BitwiseInvOpcode::XNOR.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    // ---- Zbb: counts (unary) ----

    fn process_clz(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_unary(
            CountZerosOpcode::CLZ.global_opcode().as_usize(),
            dec_insn.rd,
            dec_insn.rs1,
        )
    }

    fn process_ctz(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_unary(
            CountZerosOpcode::CTZ.global_opcode().as_usize(),
            dec_insn.rd,
            dec_insn.rs1,
        )
    }

    fn process_cpop(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_unary(
            CpopOpcode::CPOP.global_opcode().as_usize(),
            dec_insn.rd,
            dec_insn.rs1,
        )
    }

    fn process_clzw(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_unary(
            CountZerosWOpcode::CLZW.global_opcode().as_usize(),
            dec_insn.rd,
            dec_insn.rs1,
        )
    }

    fn process_ctzw(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_unary(
            CountZerosWOpcode::CTZW.global_opcode().as_usize(),
            dec_insn.rd,
            dec_insn.rs1,
        )
    }

    fn process_cpopw(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_unary(
            CpopWOpcode::CPOPW.global_opcode().as_usize(),
            dec_insn.rd,
            dec_insn.rs1,
        )
    }

    // ---- Zbb: min/max ----

    fn process_min(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            MinMaxOpcode::MIN.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_minu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            MinMaxOpcode::MINU.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_max(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            MinMaxOpcode::MAX.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_maxu(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            MinMaxOpcode::MAXU.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    // ---- Zbb: sign/zero extension (unary) ----

    fn process_sext_b(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_unary(
            ByteUnaryOpcode::SEXT_B.global_opcode().as_usize(),
            dec_insn.rd,
            dec_insn.rs1,
        )
    }

    fn process_sext_h(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_unary(
            ByteUnaryOpcode::SEXT_H.global_opcode().as_usize(),
            dec_insn.rd,
            dec_insn.rs1,
        )
    }

    fn process_zext_h(&mut self, dec_insn: RType) -> Self::InstructionResult {
        // R-type encoding with rs2 hardwired to zero; transpiles as unary.
        from_unary(
            ByteUnaryOpcode::ZEXT_H.global_opcode().as_usize(),
            dec_insn.rd,
            dec_insn.rs1,
        )
    }

    // ---- Zbb: rotates ----

    fn process_rol(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            RotateOpcode::ROL.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_ror(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            RotateOpcode::ROR.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_rori(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(RotateImmOpcode::RORI.global_opcode().as_usize(), &dec_insn)
    }

    fn process_rolw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            RotateWOpcode::ROLW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_rorw(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            RotateWOpcode::RORW.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_roriw(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(
            RotateWImmOpcode::RORIW.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    // ---- Zbb: byte ops (unary) ----

    fn process_orc_b(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_unary(
            ByteUnaryOpcode::ORC_B.global_opcode().as_usize(),
            dec_insn.rd,
            dec_insn.rs1,
        )
    }

    fn process_rev8(&mut self, dec_insn: IType) -> Self::InstructionResult {
        from_unary(
            ByteUnaryOpcode::REV8.global_opcode().as_usize(),
            dec_insn.rd,
            dec_insn.rs1,
        )
    }

    // ---- Zbs ----

    fn process_bclr(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            SingleBitOpcode::BCLR.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_bset(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            SingleBitOpcode::BSET.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_binv(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            SingleBitOpcode::BINV.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_bext(&mut self, dec_insn: RType) -> Self::InstructionResult {
        from_r_type(
            SingleBitOpcode::BEXT.global_opcode().as_usize(),
            1,
            &dec_insn,
            false,
        )
    }

    fn process_bclri(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(
            SingleBitImmOpcode::BCLRI.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_bseti(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(
            SingleBitImmOpcode::BSETI.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_binvi(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(
            SingleBitImmOpcode::BINVI.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_bexti(&mut self, dec_insn: ITypeShamt) -> Self::InstructionResult {
        from_i_type_shamt(
            SingleBitImmOpcode::BEXTI.global_opcode().as_usize(),
            &dec_insn,
        )
    }

    fn process_fence(&mut self, dec_insn: IType) -> Self::InstructionResult {
        tracing::debug!("Transpiling fence ({:?}) to nop", dec_insn);
        nop()
    }
}
