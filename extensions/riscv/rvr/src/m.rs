//! RV64M instruction lifting and generated-C support.

mod instruction;

use openvm_instructions::{
    riscv::{RV64_IMM_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
use openvm_riscv_transpiler::{DivRemOpcode, DivRemWOpcode, MulHOpcode, MulOpcode, MulWOpcode};
use rvr_openvm_ir::{ExtInstr, InstrAt, LiftedInstr};
use rvr_openvm_lift::{RvrExtension, RvrInstruction};

use crate::{
    instruction::{decode_reg, NopInstr, ZERO},
    m::instruction::{MulDivOp, Rv64MInstr},
};

/// RVR lifter for RV64M instructions.
pub struct Rv64MExtension;

impl Rv64MExtension {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for Rv64MExtension {
    fn default() -> Self {
        Self::new()
    }
}

impl RvrExtension for Rv64MExtension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();
        let operations = [
            (MulOpcode::MUL.global_opcode_usize(), MulDivOp::Mul, false),
            (
                MulHOpcode::MULH.global_opcode_usize(),
                MulDivOp::MulHighSigned,
                false,
            ),
            (
                MulHOpcode::MULHSU.global_opcode_usize(),
                MulDivOp::MulHighSignedUnsigned,
                false,
            ),
            (
                MulHOpcode::MULHU.global_opcode_usize(),
                MulDivOp::MulHighUnsigned,
                false,
            ),
            (
                DivRemOpcode::DIV.global_opcode_usize(),
                MulDivOp::DivSigned,
                false,
            ),
            (
                DivRemOpcode::DIVU.global_opcode_usize(),
                MulDivOp::DivUnsigned,
                false,
            ),
            (
                DivRemOpcode::REM.global_opcode_usize(),
                MulDivOp::RemSigned,
                false,
            ),
            (
                DivRemOpcode::REMU.global_opcode_usize(),
                MulDivOp::RemUnsigned,
                false,
            ),
            (MulWOpcode::MULW.global_opcode_usize(), MulDivOp::Mul, true),
            (
                DivRemWOpcode::DIVW.global_opcode_usize(),
                MulDivOp::DivSigned,
                true,
            ),
            (
                DivRemWOpcode::DIVUW.global_opcode_usize(),
                MulDivOp::DivUnsigned,
                true,
            ),
            (
                DivRemWOpcode::REMW.global_opcode_usize(),
                MulDivOp::RemSigned,
                true,
            ),
            (
                DivRemWOpcode::REMUW.global_opcode_usize(),
                MulDivOp::RemUnsigned,
                true,
            ),
        ];
        let (_, op, word) = operations
            .into_iter()
            .find(|(candidate, _, _)| *candidate == opcode)?;
        if insn.d != RV64_REGISTER_AS || insn.e != RV64_IMM_AS {
            return None;
        }

        let rd = decode_reg(insn.a);
        let instruction: Box<dyn ExtInstr> = if rd == ZERO {
            Box::new(NopInstr)
        } else {
            Box::new(Rv64MInstr {
                op,
                word,
                rd,
                lhs: decode_reg(insn.b),
                rhs: decode_reg(insn.c),
            })
        };
        Some(LiftedInstr::Body(InstrAt {
            pc,
            instr: instruction,
            source_loc: None,
        }))
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![("rv64m.h", include_str!("../c/rv64m.h"))]
    }
}

#[cfg(test)]
mod tests {
    use openvm_instructions::{instruction::Instruction, riscv::RV64_REGISTER_NUM_LIMBS, VmOpcode};
    use p3_baby_bear::BabyBear;
    use rvr_openvm_ir::{InstrAt, LiftedInstr};

    use super::*;

    fn instruction(opcode: VmOpcode, d: u32, e: u32) -> RvrInstruction {
        RvrInstruction::from_field(&Instruction::<BabyBear>::from_usize(
            opcode,
            [
                RV64_REGISTER_NUM_LIMBS,
                2 * RV64_REGISTER_NUM_LIMBS,
                3 * RV64_REGISTER_NUM_LIMBS,
                d as usize,
                e as usize,
                1,
                0,
            ],
        ))
    }

    #[test]
    fn all_rv64m_families_lift_with_register_domains() {
        let extension = Rv64MExtension;
        for (opcode, name) in [
            (MulOpcode::MUL.global_opcode(), "mul"),
            (MulHOpcode::MULH.global_opcode(), "mulh"),
            (MulHOpcode::MULHSU.global_opcode(), "mulhsu"),
            (MulHOpcode::MULHU.global_opcode(), "mulhu"),
            (DivRemOpcode::DIV.global_opcode(), "div"),
            (DivRemOpcode::DIVU.global_opcode(), "divu"),
            (DivRemOpcode::REM.global_opcode(), "rem"),
            (DivRemOpcode::REMU.global_opcode(), "remu"),
            (MulWOpcode::MULW.global_opcode(), "mulw"),
            (DivRemWOpcode::DIVW.global_opcode(), "divw"),
            (DivRemWOpcode::DIVUW.global_opcode(), "divuw"),
            (DivRemWOpcode::REMW.global_opcode(), "remw"),
            (DivRemWOpcode::REMUW.global_opcode(), "remuw"),
        ] {
            let insn = instruction(opcode, RV64_REGISTER_AS, RV64_IMM_AS);
            let LiftedInstr::Body(InstrAt { instr, .. }) =
                extension.try_lift(&insn, 0x100).unwrap()
            else {
                panic!("expected body instruction");
            };
            assert_eq!(instr.opname(), name);

            let wrong_source = instruction(opcode, RV64_REGISTER_AS, RV64_REGISTER_AS);
            assert!(extension.try_lift(&wrong_source, 0x100).is_none());
        }
    }

    #[test]
    fn writes_to_x0_remain_nops() {
        let insn = RvrInstruction::from_field(&Instruction::<BabyBear>::from_usize(
            MulOpcode::MUL.global_opcode(),
            [
                0,
                2 * RV64_REGISTER_NUM_LIMBS,
                3 * RV64_REGISTER_NUM_LIMBS,
                RV64_REGISTER_AS as usize,
                RV64_IMM_AS as usize,
                1,
                0,
            ],
        ));
        let LiftedInstr::Body(InstrAt { instr, .. }) =
            Rv64MExtension.try_lift(&insn, 0x100).unwrap()
        else {
            panic!("expected body instruction");
        };
        assert_eq!(instr.opname(), "nop");
    }
}
