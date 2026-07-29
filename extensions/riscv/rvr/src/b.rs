//! RV64B instruction lifting and C code generation.

mod instruction;

use openvm_instructions::{
    riscv::{RV64_IMM_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
use openvm_riscv_transpiler::{
    BitwiseInvOpcode, ByteUnaryOpcode, CountZerosOpcode, CountZerosWOpcode, CpopOpcode,
    CpopWOpcode, MinMaxOpcode, RotateImmOpcode, RotateOpcode, RotateWImmOpcode, RotateWOpcode,
    ShAddOpcode, SingleBitImmOpcode, SingleBitOpcode, SlliUwOpcode,
};
use rvr_openvm_ir::{ExtInstr, InstrAt, LiftedInstr};
use rvr_openvm_lift::{RvrExtension, RvrInstruction};

use self::instruction::{BitManipOp, Rv64BInstr};
use crate::instruction::{decode_reg, NopInstr, ZERO};

/// RVR extension for RV64B bit-manipulation instructions (Zba + Zbb + Zbs).
pub struct Rv64BExtension;

impl Rv64BExtension {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for Rv64BExtension {
    fn default() -> Self {
        Self::new()
    }
}

impl RvrExtension for Rv64BExtension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        try_lift(insn, pc)
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![("rv64b.h", include_str!("../c/rv64b.h"))]
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        0
    }
}

pub(crate) fn try_lift(insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
    let opcode = insn.opcode.as_usize();

    if let Some(op) = reg_op(opcode) {
        return lift_reg(insn, pc, op);
    }
    if let Some(op) = imm_op(opcode) {
        return lift_imm(insn, pc, op);
    }

    None
}

fn lift_reg(insn: &RvrInstruction, pc: u64, op: BitManipOp) -> Option<LiftedInstr> {
    if insn.d != RV64_REGISTER_AS || insn.e != RV64_REGISTER_AS {
        return None;
    }

    let rd = decode_reg(insn.a);
    if rd == ZERO {
        return Some(body(pc, NopInstr));
    }
    Some(body(
        pc,
        Rv64BInstr::Reg {
            op,
            rd,
            lhs: decode_reg(insn.b),
            rhs: decode_reg(insn.c),
        },
    ))
}

fn lift_imm(insn: &RvrInstruction, pc: u64, op: BitManipOp) -> Option<LiftedInstr> {
    if insn.d != RV64_REGISTER_AS || insn.e != RV64_IMM_AS {
        return None;
    }
    if !valid_imm(op, insn.c) {
        return None;
    }

    let rd = decode_reg(insn.a);
    if rd == ZERO {
        return Some(body(pc, NopInstr));
    }
    Some(body(
        pc,
        Rv64BInstr::Imm {
            op,
            rd,
            lhs: decode_reg(insn.b),
            imm: insn.c,
        },
    ))
}

fn valid_imm(op: BitManipOp, imm: u32) -> bool {
    match op {
        BitManipOp::RorIw => imm < 32,
        BitManipOp::SlliUw
        | BitManipOp::Rori
        | BitManipOp::BclrI
        | BitManipOp::BsetI
        | BitManipOp::BinvI
        | BitManipOp::BextI => imm < 64,
        _ => true,
    }
}

fn reg_op(opcode: usize) -> Option<BitManipOp> {
    Some(match opcode {
        x if x == ShAddOpcode::SH1ADD.global_opcode_usize() => BitManipOp::Sh1Add,
        x if x == ShAddOpcode::SH2ADD.global_opcode_usize() => BitManipOp::Sh2Add,
        x if x == ShAddOpcode::SH3ADD.global_opcode_usize() => BitManipOp::Sh3Add,
        x if x == ShAddOpcode::ADD_UW.global_opcode_usize() => BitManipOp::AddUw,
        x if x == ShAddOpcode::SH1ADD_UW.global_opcode_usize() => BitManipOp::Sh1AddUw,
        x if x == ShAddOpcode::SH2ADD_UW.global_opcode_usize() => BitManipOp::Sh2AddUw,
        x if x == ShAddOpcode::SH3ADD_UW.global_opcode_usize() => BitManipOp::Sh3AddUw,
        x if x == BitwiseInvOpcode::ANDN.global_opcode_usize() => BitManipOp::AndN,
        x if x == BitwiseInvOpcode::ORN.global_opcode_usize() => BitManipOp::OrN,
        x if x == BitwiseInvOpcode::XNOR.global_opcode_usize() => BitManipOp::Xnor,
        x if x == RotateOpcode::ROL.global_opcode_usize() => BitManipOp::Rol,
        x if x == RotateOpcode::ROR.global_opcode_usize() => BitManipOp::Ror,
        x if x == RotateWOpcode::ROLW.global_opcode_usize() => BitManipOp::RolW,
        x if x == RotateWOpcode::RORW.global_opcode_usize() => BitManipOp::RorW,
        x if x == MinMaxOpcode::MIN.global_opcode_usize() => BitManipOp::Min,
        x if x == MinMaxOpcode::MINU.global_opcode_usize() => BitManipOp::MinU,
        x if x == MinMaxOpcode::MAX.global_opcode_usize() => BitManipOp::Max,
        x if x == MinMaxOpcode::MAXU.global_opcode_usize() => BitManipOp::MaxU,
        x if x == SingleBitOpcode::BCLR.global_opcode_usize() => BitManipOp::Bclr,
        x if x == SingleBitOpcode::BSET.global_opcode_usize() => BitManipOp::Bset,
        x if x == SingleBitOpcode::BINV.global_opcode_usize() => BitManipOp::Binv,
        x if x == SingleBitOpcode::BEXT.global_opcode_usize() => BitManipOp::Bext,
        _ => return None,
    })
}

fn imm_op(opcode: usize) -> Option<BitManipOp> {
    Some(match opcode {
        x if x == SlliUwOpcode::SLLI_UW.global_opcode_usize() => BitManipOp::SlliUw,
        x if x == RotateImmOpcode::RORI.global_opcode_usize() => BitManipOp::Rori,
        x if x == RotateWImmOpcode::RORIW.global_opcode_usize() => BitManipOp::RorIw,
        x if x == CountZerosOpcode::CLZ.global_opcode_usize() => BitManipOp::Clz,
        x if x == CountZerosOpcode::CTZ.global_opcode_usize() => BitManipOp::Ctz,
        x if x == CountZerosWOpcode::CLZW.global_opcode_usize() => BitManipOp::ClzW,
        x if x == CountZerosWOpcode::CTZW.global_opcode_usize() => BitManipOp::CtzW,
        x if x == CpopOpcode::CPOP.global_opcode_usize() => BitManipOp::Cpop,
        x if x == CpopWOpcode::CPOPW.global_opcode_usize() => BitManipOp::CpopW,
        x if x == ByteUnaryOpcode::SEXT_B.global_opcode_usize() => BitManipOp::SextB,
        x if x == ByteUnaryOpcode::SEXT_H.global_opcode_usize() => BitManipOp::SextH,
        x if x == ByteUnaryOpcode::ZEXT_H.global_opcode_usize() => BitManipOp::ZextH,
        x if x == ByteUnaryOpcode::ORC_B.global_opcode_usize() => BitManipOp::OrcB,
        x if x == ByteUnaryOpcode::REV8.global_opcode_usize() => BitManipOp::Rev8,
        x if x == SingleBitImmOpcode::BCLRI.global_opcode_usize() => BitManipOp::BclrI,
        x if x == SingleBitImmOpcode::BSETI.global_opcode_usize() => BitManipOp::BsetI,
        x if x == SingleBitImmOpcode::BINVI.global_opcode_usize() => BitManipOp::BinvI,
        x if x == SingleBitImmOpcode::BEXTI.global_opcode_usize() => BitManipOp::BextI,
        _ => return None,
    })
}

fn body(pc: u64, instr: impl ExtInstr + 'static) -> LiftedInstr {
    LiftedInstr::Body(InstrAt {
        pc,
        instr: Box::new(instr),
        source_loc: None,
    })
}

#[cfg(test)]
mod tests {
    use openvm_instructions::{instruction::Instruction, riscv::RV64_REGISTER_NUM_LIMBS, VmOpcode};
    use p3_baby_bear::BabyBear;
    use rvr_openvm_ir::InstrAt;

    use super::*;

    fn instruction(opcode: VmOpcode, operands: [usize; 7]) -> RvrInstruction {
        RvrInstruction::from_field(&Instruction::<BabyBear>::from_usize(opcode, operands))
    }

    fn reg_operands(rd: usize, d: u32, e: u32) -> [usize; 7] {
        [
            rd,
            2 * RV64_REGISTER_NUM_LIMBS,
            3 * RV64_REGISTER_NUM_LIMBS,
            d as usize,
            e as usize,
            1,
            0,
        ]
    }

    fn imm_operands(rd: usize, imm: usize, d: u32, e: u32) -> [usize; 7] {
        [
            rd,
            2 * RV64_REGISTER_NUM_LIMBS,
            imm,
            d as usize,
            e as usize,
            1,
            0,
        ]
    }

    fn lifted_name(insn: &RvrInstruction) -> Option<String> {
        Rv64BExtension
            .try_lift(insn, 0x100)
            .map(|lifted| match lifted {
                LiftedInstr::Body(InstrAt { instr, .. }) => instr.opname().to_string(),
                LiftedInstr::Term { .. } => {
                    unreachable!("RV64B instructions do not terminate blocks")
                }
            })
    }

    #[test]
    fn register_bitmanip_families_are_domain_separated() {
        let opcodes = [
            (ShAddOpcode::SH1ADD.global_opcode(), "sh1add"),
            (ShAddOpcode::SH2ADD.global_opcode(), "sh2add"),
            (ShAddOpcode::SH3ADD.global_opcode(), "sh3add"),
            (ShAddOpcode::ADD_UW.global_opcode(), "add.uw"),
            (ShAddOpcode::SH1ADD_UW.global_opcode(), "sh1add.uw"),
            (ShAddOpcode::SH2ADD_UW.global_opcode(), "sh2add.uw"),
            (ShAddOpcode::SH3ADD_UW.global_opcode(), "sh3add.uw"),
            (BitwiseInvOpcode::ANDN.global_opcode(), "andn"),
            (BitwiseInvOpcode::ORN.global_opcode(), "orn"),
            (BitwiseInvOpcode::XNOR.global_opcode(), "xnor"),
            (RotateOpcode::ROL.global_opcode(), "rol"),
            (RotateOpcode::ROR.global_opcode(), "ror"),
            (RotateWOpcode::ROLW.global_opcode(), "rolw"),
            (RotateWOpcode::RORW.global_opcode(), "rorw"),
            (MinMaxOpcode::MIN.global_opcode(), "min"),
            (MinMaxOpcode::MINU.global_opcode(), "minu"),
            (MinMaxOpcode::MAX.global_opcode(), "max"),
            (MinMaxOpcode::MAXU.global_opcode(), "maxu"),
            (SingleBitOpcode::BCLR.global_opcode(), "bclr"),
            (SingleBitOpcode::BSET.global_opcode(), "bset"),
            (SingleBitOpcode::BINV.global_opcode(), "binv"),
            (SingleBitOpcode::BEXT.global_opcode(), "bext"),
        ];
        for (opcode, name) in opcodes {
            let valid = instruction(
                opcode,
                reg_operands(RV64_REGISTER_NUM_LIMBS, RV64_REGISTER_AS, RV64_REGISTER_AS),
            );
            assert_eq!(lifted_name(&valid).as_deref(), Some(name));

            let wrong_domain = instruction(
                opcode,
                reg_operands(RV64_REGISTER_NUM_LIMBS, RV64_REGISTER_AS, RV64_IMM_AS),
            );
            assert!(Rv64BExtension.try_lift(&wrong_domain, 0x100).is_none());
        }
    }

    #[test]
    fn immediate_bitmanip_families_are_domain_separated() {
        let opcodes = [
            (SlliUwOpcode::SLLI_UW.global_opcode(), "slli.uw", 63),
            (RotateImmOpcode::RORI.global_opcode(), "rori", 63),
            (RotateWImmOpcode::RORIW.global_opcode(), "roriw", 31),
            (CountZerosOpcode::CLZ.global_opcode(), "clz", 0),
            (CountZerosOpcode::CTZ.global_opcode(), "ctz", 0),
            (CountZerosWOpcode::CLZW.global_opcode(), "clzw", 0),
            (CountZerosWOpcode::CTZW.global_opcode(), "ctzw", 0),
            (CpopOpcode::CPOP.global_opcode(), "cpop", 0),
            (CpopWOpcode::CPOPW.global_opcode(), "cpopw", 0),
            (ByteUnaryOpcode::SEXT_B.global_opcode(), "sext.b", 0),
            (ByteUnaryOpcode::SEXT_H.global_opcode(), "sext.h", 0),
            (ByteUnaryOpcode::ZEXT_H.global_opcode(), "zext.h", 0),
            (ByteUnaryOpcode::ORC_B.global_opcode(), "orc.b", 0),
            (ByteUnaryOpcode::REV8.global_opcode(), "rev8", 0),
            (SingleBitImmOpcode::BCLRI.global_opcode(), "bclri", 63),
            (SingleBitImmOpcode::BSETI.global_opcode(), "bseti", 63),
            (SingleBitImmOpcode::BINVI.global_opcode(), "binvi", 63),
            (SingleBitImmOpcode::BEXTI.global_opcode(), "bexti", 63),
        ];
        for (opcode, name, imm) in opcodes {
            let valid = instruction(
                opcode,
                imm_operands(RV64_REGISTER_NUM_LIMBS, imm, RV64_REGISTER_AS, RV64_IMM_AS),
            );
            assert_eq!(lifted_name(&valid).as_deref(), Some(name));

            let wrong_domain = instruction(
                opcode,
                imm_operands(
                    RV64_REGISTER_NUM_LIMBS,
                    imm,
                    RV64_REGISTER_AS,
                    RV64_REGISTER_AS,
                ),
            );
            assert!(Rv64BExtension.try_lift(&wrong_domain, 0x100).is_none());
        }
    }

    #[test]
    fn invalid_shift_immediates_do_not_lift() {
        for opcode in [
            SlliUwOpcode::SLLI_UW.global_opcode(),
            RotateImmOpcode::RORI.global_opcode(),
            SingleBitImmOpcode::BCLRI.global_opcode(),
            SingleBitImmOpcode::BSETI.global_opcode(),
            SingleBitImmOpcode::BINVI.global_opcode(),
            SingleBitImmOpcode::BEXTI.global_opcode(),
        ] {
            let insn = instruction(
                opcode,
                imm_operands(RV64_REGISTER_NUM_LIMBS, 64, RV64_REGISTER_AS, RV64_IMM_AS),
            );
            assert!(Rv64BExtension.try_lift(&insn, 0x100).is_none());
        }

        let roriw = instruction(
            RotateWImmOpcode::RORIW.global_opcode(),
            imm_operands(RV64_REGISTER_NUM_LIMBS, 32, RV64_REGISTER_AS, RV64_IMM_AS),
        );
        assert!(Rv64BExtension.try_lift(&roriw, 0x100).is_none());
    }

    #[test]
    fn writes_to_x0_remain_nops() {
        let insn = instruction(
            ShAddOpcode::SH1ADD.global_opcode(),
            reg_operands(0, RV64_REGISTER_AS, RV64_REGISTER_AS),
        );
        assert_eq!(lifted_name(&insn).as_deref(), Some("nop"));
    }
}
