//! RV64I instruction lifting, CFG metadata, and C code generation.

mod instruction;

use std::collections::{BTreeMap, HashSet};

use openvm_instructions::{
    exe::SparseMemoryImage,
    program::DEFAULT_PC_STEP,
    riscv::{IMM_AS, MEMORY_AS, REGISTER_AS, REGISTER_BYTES},
    LocalOpcode,
};
use openvm_riscv_transpiler::{
    AuipcOpcode, BaseAluImmOpcode, BaseAluOpcode, BaseAluWImmOpcode, BaseAluWOpcode,
    BranchEqualOpcode, BranchLessThanOpcode, JalLuiOpcode, JalrOpcode, LessThanImmOpcode,
    LessThanOpcode, LoadStoreOpcode, ShiftImmOpcode, ShiftOpcode, ShiftWImmOpcode, ShiftWOpcode,
};
use rvr_openvm_ir::{
    CfgBranchCond, CfgOperand, ExtInstr, InstrAt, LiftedInstr, MemWidth, Terminator,
};
use rvr_openvm_lift::{max_main_memory_pages_for_contiguous_range, RvrExtension, RvrInstruction};

use self::instruction::{AluOp, RiscvIInstr};
use crate::instruction::{decode_imm_cg, decode_reg, reg_operand, ZERO};

const U24_MASK: u32 = (1 << 24) - 1;
const MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION: usize =
    max_main_memory_pages_for_contiguous_range(REGISTER_BYTES as usize);

/// RVR extension for RV64I base integer instructions.
pub struct RiscvIExtension;

impl RiscvIExtension {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for RiscvIExtension {
    fn default() -> Self {
        Self::new()
    }
}

impl RvrExtension for RiscvIExtension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        try_lift(insn, pc)
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        Vec::new()
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION
    }

    /// Find initialized values that match valid RV64 instruction PCs.
    ///
    /// Scans four-byte-aligned initialized main-memory locations as little-endian
    /// `u32`s. Matching values become additional indirect-jump target candidates.
    /// This covers absolute code pointers in switch tables, function-pointer
    /// arrays, and other initialized data, including read-only tables.
    fn extra_cfg_targets(
        &self,
        init_memory: &SparseMemoryImage,
        valid_pcs: &HashSet<u64>,
    ) -> Vec<u64> {
        let bytes = init_memory
            .iter()
            .filter_map(|(&(address_space, address), &byte)| {
                (address_space == MEMORY_AS).then_some((address, byte))
            })
            .collect::<BTreeMap<_, _>>();
        bytes
            .keys()
            .copied()
            .filter(|address| address.is_multiple_of(DEFAULT_PC_STEP))
            .filter_map(|address| {
                let address1 = address.checked_add(1)?;
                let address2 = address.checked_add(2)?;
                let address3 = address.checked_add(3)?;
                let value = u32::from_le_bytes([
                    *bytes.get(&address)?,
                    *bytes.get(&address1)?,
                    *bytes.get(&address2)?,
                    *bytes.get(&address3)?,
                ]);
                valid_pcs
                    .contains(&u64::from(value))
                    .then_some(u64::from(value))
            })
            .collect()
    }
}

pub(crate) fn try_lift(insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
    let opcode = insn.opcode.as_usize();

    let alu_reg = [
        (BaseAluOpcode::ADD.global_opcode_usize(), AluOp::Add),
        (BaseAluOpcode::SUB.global_opcode_usize(), AluOp::Sub),
        (BaseAluOpcode::XOR.global_opcode_usize(), AluOp::Xor),
        (BaseAluOpcode::OR.global_opcode_usize(), AluOp::Or),
        (BaseAluOpcode::AND.global_opcode_usize(), AluOp::And),
        (ShiftOpcode::SLL.global_opcode_usize(), AluOp::Sll),
        (ShiftOpcode::SRL.global_opcode_usize(), AluOp::Srl),
        (ShiftOpcode::SRA.global_opcode_usize(), AluOp::Sra),
        (LessThanOpcode::SLT.global_opcode_usize(), AluOp::Slt),
        (LessThanOpcode::SLTU.global_opcode_usize(), AluOp::Sltu),
    ];
    if let Some((_, op)) = alu_reg
        .into_iter()
        .find(|(candidate, _)| *candidate == opcode)
    {
        return lift_alu(insn, pc, op, false, None);
    }

    let alu_imm = [
        (BaseAluImmOpcode::ADDI.global_opcode_usize(), AluOp::Add),
        (BaseAluImmOpcode::XORI.global_opcode_usize(), AluOp::Xor),
        (BaseAluImmOpcode::ORI.global_opcode_usize(), AluOp::Or),
        (BaseAluImmOpcode::ANDI.global_opcode_usize(), AluOp::And),
        (LessThanImmOpcode::SLTI.global_opcode_usize(), AluOp::Slt),
        (LessThanImmOpcode::SLTIU.global_opcode_usize(), AluOp::Sltu),
    ];
    if let Some((_, op)) = alu_imm
        .into_iter()
        .find(|(candidate, _)| *candidate == opcode)
    {
        let raw = insn.c;
        let imm = sign_extend_12(raw);
        if raw != (imm as u32 & U24_MASK) {
            return None;
        }
        return lift_alu(
            insn,
            pc,
            op,
            false,
            Some(CfgOperand::Const(imm as i64 as u64)),
        );
    }

    let shift_imm = [
        (ShiftImmOpcode::SLLI.global_opcode_usize(), AluOp::Sll),
        (ShiftImmOpcode::SRLI.global_opcode_usize(), AluOp::Srl),
        (ShiftImmOpcode::SRAI.global_opcode_usize(), AluOp::Sra),
    ];
    if let Some((_, op)) = shift_imm
        .into_iter()
        .find(|(candidate, _)| *candidate == opcode)
    {
        if insn.c >= 64 {
            return None;
        }
        return lift_alu(
            insn,
            pc,
            op,
            false,
            Some(CfgOperand::Const(u64::from(insn.c))),
        );
    }

    let alu_w_reg = [
        (BaseAluWOpcode::ADDW.global_opcode_usize(), AluOp::Add),
        (BaseAluWOpcode::SUBW.global_opcode_usize(), AluOp::Sub),
        (ShiftWOpcode::SLLW.global_opcode_usize(), AluOp::Sll),
        (ShiftWOpcode::SRLW.global_opcode_usize(), AluOp::Srl),
        (ShiftWOpcode::SRAW.global_opcode_usize(), AluOp::Sra),
    ];
    if let Some((_, op)) = alu_w_reg
        .into_iter()
        .find(|(candidate, _)| *candidate == opcode)
    {
        return lift_alu(insn, pc, op, true, None);
    }

    if opcode == BaseAluWImmOpcode::ADDIW.global_opcode_usize() {
        let raw = insn.c;
        let imm = sign_extend_12(raw);
        if raw != (imm as u32 & U24_MASK) {
            return None;
        }
        return lift_alu(
            insn,
            pc,
            AluOp::Add,
            true,
            Some(CfgOperand::Const(imm as i64 as u64)),
        );
    }

    let shift_w_imm = [
        (ShiftWImmOpcode::SLLIW.global_opcode_usize(), AluOp::Sll),
        (ShiftWImmOpcode::SRLIW.global_opcode_usize(), AluOp::Srl),
        (ShiftWImmOpcode::SRAIW.global_opcode_usize(), AluOp::Sra),
    ];
    if let Some((_, op)) = shift_w_imm
        .into_iter()
        .find(|(candidate, _)| *candidate == opcode)
    {
        if insn.c >= 32 {
            return None;
        }
        return lift_alu(
            insn,
            pc,
            op,
            true,
            Some(CfgOperand::Const(u64::from(insn.c))),
        );
    }

    let loads = [
        (
            LoadStoreOpcode::LOADD.global_opcode_usize(),
            MemWidth::Double,
            false,
        ),
        (
            LoadStoreOpcode::LOADBU.global_opcode_usize(),
            MemWidth::Byte,
            false,
        ),
        (
            LoadStoreOpcode::LOADHU.global_opcode_usize(),
            MemWidth::Half,
            false,
        ),
        (
            LoadStoreOpcode::LOADWU.global_opcode_usize(),
            MemWidth::Word,
            false,
        ),
        (
            LoadStoreOpcode::LOADB.global_opcode_usize(),
            MemWidth::Byte,
            true,
        ),
        (
            LoadStoreOpcode::LOADH.global_opcode_usize(),
            MemWidth::Half,
            true,
        ),
        (
            LoadStoreOpcode::LOADW.global_opcode_usize(),
            MemWidth::Word,
            true,
        ),
    ];
    if let Some((_, width, signed)) = loads
        .into_iter()
        .find(|(candidate, _, _)| *candidate == opcode)
    {
        return lift_load(insn, pc, width, signed);
    }

    let stores = [
        (
            LoadStoreOpcode::STORED.global_opcode_usize(),
            MemWidth::Double,
        ),
        (
            LoadStoreOpcode::STOREW.global_opcode_usize(),
            MemWidth::Word,
        ),
        (
            LoadStoreOpcode::STOREH.global_opcode_usize(),
            MemWidth::Half,
        ),
        (
            LoadStoreOpcode::STOREB.global_opcode_usize(),
            MemWidth::Byte,
        ),
    ];
    if let Some((_, width)) = stores
        .into_iter()
        .find(|(candidate, _)| *candidate == opcode)
    {
        return lift_store(insn, pc, width);
    }

    let branches = [
        (
            BranchEqualOpcode::BEQ.global_opcode_usize(),
            CfgBranchCond::Eq,
        ),
        (
            BranchEqualOpcode::BNE.global_opcode_usize(),
            CfgBranchCond::Ne,
        ),
        (
            BranchLessThanOpcode::BLT.global_opcode_usize(),
            CfgBranchCond::LessThanSigned,
        ),
        (
            BranchLessThanOpcode::BLTU.global_opcode_usize(),
            CfgBranchCond::LessThanUnsigned,
        ),
        (
            BranchLessThanOpcode::BGE.global_opcode_usize(),
            CfgBranchCond::GreaterEqualSigned,
        ),
        (
            BranchLessThanOpcode::BGEU.global_opcode_usize(),
            CfgBranchCond::GreaterEqualUnsigned,
        ),
    ];
    if let Some((_, cond)) = branches
        .into_iter()
        .find(|(candidate, _)| *candidate == opcode)
    {
        return Some(lift_branch(insn, pc, cond));
    }

    if opcode == JalLuiOpcode::JAL.global_opcode_usize() {
        return Some(lift_jal(insn, pc));
    }
    if opcode == JalLuiOpcode::LUI.global_opcode_usize() {
        return Some(lift_lui(insn, pc));
    }
    if opcode == JalrOpcode::JALR.global_opcode_usize() {
        return Some(lift_jalr(insn, pc));
    }
    if opcode == AuipcOpcode::AUIPC.global_opcode_usize() {
        return Some(lift_auipc(insn, pc));
    }

    None
}

fn lift_alu(
    insn: &RvrInstruction,
    pc: u64,
    op: AluOp,
    word: bool,
    immediate: Option<CfgOperand>,
) -> Option<LiftedInstr> {
    let expected_e = if immediate.is_some() {
        IMM_AS
    } else {
        REGISTER_AS
    };
    if insn.d != REGISTER_AS || insn.e != expected_e {
        return None;
    }

    let rd = decode_reg(insn.a);
    let lhs = decode_reg(insn.b);
    let rhs_reg = immediate.is_none().then(|| decode_reg(insn.c));
    let rhs = immediate.unwrap_or_else(|| reg_operand(rhs_reg.unwrap()));
    Some(body(
        pc,
        RiscvIInstr::Alu {
            op,
            word,
            immediate: immediate.is_some(),
            rd,
            lhs,
            rhs,
            rhs_reg,
        },
    ))
}

/// Lift a load encoded as `rd=a/8`, `rs1=b/8`, immediate low bits in `c`,
/// and the immediate sign marker in `g`.
fn lift_load(insn: &RvrInstruction, pc: u64, width: MemWidth, signed: bool) -> Option<LiftedInstr> {
    if insn.d != REGISTER_AS || insn.e != MEMORY_AS {
        return None;
    }
    let rd = decode_reg(insn.a);
    Some(body(
        pc,
        RiscvIInstr::Load {
            width,
            signed,
            rd,
            base: decode_reg(insn.b),
            offset: decode_imm_cg(insn) as i16,
        },
    ))
}

/// Lift a store encoded as `rs2=a/8`, `rs1=b/8`, immediate low bits in `c`,
/// and the immediate sign marker in `g`.
fn lift_store(insn: &RvrInstruction, pc: u64, width: MemWidth) -> Option<LiftedInstr> {
    if insn.d != REGISTER_AS || insn.e != MEMORY_AS {
        return None;
    }
    Some(body(
        pc,
        RiscvIInstr::Store {
            width,
            base: decode_reg(insn.b),
            src: decode_reg(insn.a),
            offset: decode_imm_cg(insn) as i16,
        },
    ))
}

/// Lift a branch encoded as `rs1=a/8`, `rs2=b/8`, and an offset in `c`
/// interpreted as a signed source-field value.
fn lift_branch(insn: &RvrInstruction, pc: u64, cond: CfgBranchCond) -> LiftedInstr {
    term(
        pc,
        RiscvIInstr::Branch {
            cond,
            lhs: decode_reg(insn.a),
            rhs: decode_reg(insn.b),
            target: pc.wrapping_add_signed(i64::from(insn.signed_c())),
        },
    )
}

/// Lift JAL encoded as `rd=a/8` and an offset in `c` interpreted as a signed
/// source-field value.
fn lift_jal(insn: &RvrInstruction, pc: u64) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    term(
        pc,
        RiscvIInstr::Jump {
            link_dst: (rd != ZERO).then_some(rd),
            target: pc.wrapping_add_signed(i64::from(insn.signed_c())),
        },
    )
}

/// Lift JALR encoded as `rd=a/8`, `rs1=b/8`, immediate low bits in `c`,
/// and the immediate sign marker in `g`.
fn lift_jalr(insn: &RvrInstruction, pc: u64) -> LiftedInstr {
    let rd = decode_reg(insn.a);
    term(
        pc,
        RiscvIInstr::JumpIndirect {
            link_dst: (rd != ZERO).then_some(rd),
            base: decode_reg(insn.b),
            offset: decode_imm_cg(insn) as i32,
        },
    )
}

/// Lift LUI encoded as `rd=a/8` and the upper immediate in `c << 12`.
fn lift_lui(insn: &RvrInstruction, pc: u64) -> LiftedInstr {
    body(
        pc,
        RiscvIInstr::Const {
            name: "lui",
            rd: decode_reg(insn.a),
            value: sign_extend_32(insn.c << 12),
        },
    )
}

/// Lift AUIPC encoded as `rd=a/8` and `c << 8`.
///
/// The transpiler stores `(imm & 0xfffff000) >> 8` in `c`, so shifting by
/// eight reconstructs the upper 20 bits with twelve low zero bits.
fn lift_auipc(insn: &RvrInstruction, pc: u64) -> LiftedInstr {
    let upper = insn.c << 8;
    body(
        pc,
        RiscvIInstr::Const {
            name: "auipc",
            rd: decode_reg(insn.a),
            value: pc.wrapping_add(sign_extend_32(upper)),
        },
    )
}

fn body(pc: u64, instr: impl ExtInstr + 'static) -> LiftedInstr {
    LiftedInstr::Body(InstrAt {
        pc,
        instr: Box::new(instr),
        source_loc: None,
    })
}

fn term(pc: u64, instr: RiscvIInstr) -> LiftedInstr {
    LiftedInstr::Term {
        pc,
        terminator: Terminator::instruction(instr),
        source_loc: None,
    }
}

/// Sign-extend a 12-bit immediate stored in the low 24 bits.
fn sign_extend_12(value: u32) -> i32 {
    let value = value & 0xfff;
    if value & 0x800 != 0 {
        (value | 0xffff_f000) as i32
    } else {
        value as i32
    }
}

/// Sign-extend a 32-bit value into an RV64 register value.
fn sign_extend_32(value: u32) -> u64 {
    value as i32 as i64 as u64
}

#[cfg(test)]
mod tests {
    use openvm_instructions::{
        instruction::Instruction, riscv::REGISTER_NUM_LIMBS, VmOpcode, DEFERRAL_AS,
    };
    use p3_baby_bear::BabyBear;

    use super::*;
    use crate::instruction::Reg;

    fn instruction(opcode: impl LocalOpcode, operands: [usize; 7]) -> RvrInstruction {
        instruction_for_opcode(opcode.global_opcode(), operands)
    }

    fn instruction_for_opcode(opcode: VmOpcode, operands: [usize; 7]) -> RvrInstruction {
        RvrInstruction::from_field(&Instruction::<BabyBear>::from_usize(opcode, operands))
    }

    fn alu_operands(c: usize, d: u32, e: u32) -> [usize; 7] {
        [
            REGISTER_NUM_LIMBS,
            2 * REGISTER_NUM_LIMBS,
            c,
            d as usize,
            e as usize,
            1,
            0,
        ]
    }

    fn lifted_name(insn: &RvrInstruction) -> Option<String> {
        try_lift(insn, 0x100).map(|lifted| match lifted {
            LiftedInstr::Body(InstrAt { instr, .. }) => instr.opname().to_string(),
            LiftedInstr::Term { terminator, .. } => terminator.opname().to_string(),
        })
    }

    #[test]
    fn initialized_u32_code_pointer_at_four_byte_alignment_is_a_cfg_target() {
        let target = 0x100u32;
        let init_memory = SparseMemoryImage::from_iter(
            target
                .to_le_bytes()
                .into_iter()
                .enumerate()
                .map(|(offset, byte)| ((MEMORY_AS, 4 + offset as u32), byte)),
        );

        assert_eq!(
            RiscvIExtension.extra_cfg_targets(&init_memory, &HashSet::from([u64::from(target)])),
            vec![u64::from(target)]
        );
    }

    #[test]
    fn register_and_immediate_alu_families_are_domain_separated() {
        let register_opcodes = [
            (BaseAluOpcode::ADD.global_opcode(), "add"),
            (BaseAluOpcode::SUB.global_opcode(), "sub"),
            (BaseAluOpcode::XOR.global_opcode(), "xor"),
            (BaseAluOpcode::OR.global_opcode(), "or"),
            (BaseAluOpcode::AND.global_opcode(), "and"),
            (ShiftOpcode::SLL.global_opcode(), "sll"),
            (ShiftOpcode::SRL.global_opcode(), "srl"),
            (ShiftOpcode::SRA.global_opcode(), "sra"),
            (LessThanOpcode::SLT.global_opcode(), "slt"),
            (LessThanOpcode::SLTU.global_opcode(), "sltu"),
            (BaseAluWOpcode::ADDW.global_opcode(), "addw"),
            (BaseAluWOpcode::SUBW.global_opcode(), "subw"),
            (ShiftWOpcode::SLLW.global_opcode(), "sllw"),
            (ShiftWOpcode::SRLW.global_opcode(), "srlw"),
            (ShiftWOpcode::SRAW.global_opcode(), "sraw"),
        ];
        for (opcode, name) in register_opcodes {
            let valid = instruction_for_opcode(
                opcode,
                alu_operands(3 * REGISTER_NUM_LIMBS, REGISTER_AS, REGISTER_AS),
            );
            assert_eq!(lifted_name(&valid).as_deref(), Some(name));

            let wrong_domain = instruction_for_opcode(opcode, alu_operands(0, REGISTER_AS, IMM_AS));
            assert!(try_lift(&wrong_domain, 0x100).is_none());
        }
    }

    #[test]
    fn immediate_alu_families_require_canonical_i12_values() {
        let opcodes = [
            (BaseAluImmOpcode::ADDI.global_opcode(), "addi"),
            (BaseAluImmOpcode::XORI.global_opcode(), "xori"),
            (BaseAluImmOpcode::ORI.global_opcode(), "ori"),
            (BaseAluImmOpcode::ANDI.global_opcode(), "andi"),
            (LessThanImmOpcode::SLTI.global_opcode(), "slti"),
            (LessThanImmOpcode::SLTIU.global_opcode(), "sltiu"),
            (BaseAluWImmOpcode::ADDIW.global_opcode(), "addiw"),
        ];
        for (opcode, name) in opcodes {
            for immediate in [0, 0x7ff, 0xff_f800, 0xff_ffff] {
                let valid =
                    instruction_for_opcode(opcode, alu_operands(immediate, REGISTER_AS, IMM_AS));
                assert_eq!(lifted_name(&valid).as_deref(), Some(name));
            }
            for invalid in [0x800, 0xffff] {
                let invalid =
                    instruction_for_opcode(opcode, alu_operands(invalid, REGISTER_AS, IMM_AS));
                assert!(try_lift(&invalid, 0x100).is_none());
            }
            let wrong_domain =
                instruction_for_opcode(opcode, alu_operands(0, REGISTER_AS, REGISTER_AS));
            assert!(try_lift(&wrong_domain, 0x100).is_none());
        }
    }

    #[test]
    fn shift_immediate_widths_are_enforced() {
        for (opcode, name) in [
            (ShiftImmOpcode::SLLI.global_opcode(), "slli"),
            (ShiftImmOpcode::SRLI.global_opcode(), "srli"),
            (ShiftImmOpcode::SRAI.global_opcode(), "srai"),
        ] {
            for shamt in [0, 63] {
                let valid =
                    instruction_for_opcode(opcode, alu_operands(shamt, REGISTER_AS, IMM_AS));
                assert_eq!(lifted_name(&valid).as_deref(), Some(name));
            }
            let invalid = instruction_for_opcode(opcode, alu_operands(64, REGISTER_AS, IMM_AS));
            assert!(try_lift(&invalid, 0x100).is_none());
        }

        for (opcode, name) in [
            (ShiftWImmOpcode::SLLIW.global_opcode(), "slliw"),
            (ShiftWImmOpcode::SRLIW.global_opcode(), "srliw"),
            (ShiftWImmOpcode::SRAIW.global_opcode(), "sraiw"),
        ] {
            for shamt in [0, 31] {
                let valid =
                    instruction_for_opcode(opcode, alu_operands(shamt, REGISTER_AS, IMM_AS));
                assert_eq!(lifted_name(&valid).as_deref(), Some(name));
            }
            let invalid = instruction_for_opcode(opcode, alu_operands(32, REGISTER_AS, IMM_AS));
            assert!(try_lift(&invalid, 0x100).is_none());
        }
    }

    #[test]
    fn load_store_families_require_main_memory_domain() {
        let opcodes = [
            (LoadStoreOpcode::LOADD.global_opcode(), "ld"),
            (LoadStoreOpcode::LOADBU.global_opcode(), "lbu"),
            (LoadStoreOpcode::LOADHU.global_opcode(), "lhu"),
            (LoadStoreOpcode::LOADWU.global_opcode(), "lwu"),
            (LoadStoreOpcode::LOADB.global_opcode(), "lb"),
            (LoadStoreOpcode::LOADH.global_opcode(), "lh"),
            (LoadStoreOpcode::LOADW.global_opcode(), "lw"),
            (LoadStoreOpcode::STORED.global_opcode(), "sd"),
            (LoadStoreOpcode::STOREW.global_opcode(), "sw"),
            (LoadStoreOpcode::STOREH.global_opcode(), "sh"),
            (LoadStoreOpcode::STOREB.global_opcode(), "sb"),
        ];
        for (opcode, name) in opcodes {
            let valid = instruction_for_opcode(opcode, alu_operands(0, REGISTER_AS, MEMORY_AS));
            assert_eq!(lifted_name(&valid).as_deref(), Some(name));

            let wrong_memory =
                instruction_for_opcode(opcode, alu_operands(0, REGISTER_AS, DEFERRAL_AS));
            assert!(try_lift(&wrong_memory, 0x100).is_none());
            let wrong_destination =
                instruction_for_opcode(opcode, alu_operands(0, MEMORY_AS, MEMORY_AS));
            assert!(try_lift(&wrong_destination, 0x100).is_none());
        }
    }

    #[test]
    fn control_flow_preserves_negative_field_offsets() {
        let pc = 0x1000;
        for opcode in [
            BranchEqualOpcode::BEQ.global_opcode(),
            JalLuiOpcode::JAL.global_opcode(),
        ] {
            let insn = RvrInstruction::from_field(&Instruction::<BabyBear>::from_isize(
                opcode, 8, 16, -12, 0, 0,
            ));
            let LiftedInstr::Term { terminator, .. } = try_lift(&insn, pc).unwrap() else {
                panic!("expected terminator");
            };
            match terminator.cfg_term(pc, pc + 4) {
                rvr_openvm_ir::CfgTerm::Branch { target, .. }
                | rvr_openvm_ir::CfgTerm::Jump { target, .. } => assert_eq!(target, pc - 12),
                other => panic!("unexpected terminator: {other:?}"),
            }
        }
    }

    #[test]
    fn load_to_x0_keeps_the_memory_access() {
        let insn = instruction(
            LoadStoreOpcode::LOADD,
            [
                0,
                REGISTER_NUM_LIMBS,
                0,
                REGISTER_AS as usize,
                MEMORY_AS as usize,
                1,
                0,
            ],
        );
        let LiftedInstr::Body(InstrAt { instr, .. }) = try_lift(&insn, 0).unwrap() else {
            panic!("expected body instruction");
        };
        assert_eq!(instr.opname(), "ld");
        assert!(instr.accesses_memory());
        assert_eq!(instr.cfg_effect(), rvr_openvm_ir::CfgEffect::None);
    }

    #[test]
    fn preserves_high_auipc_values() {
        let pc = 0x1_0000_0000;
        let insn = instruction(
            AuipcOpcode::AUIPC,
            [REGISTER_NUM_LIMBS, 0, 1, REGISTER_AS as usize, 0, 0, 0],
        );
        let LiftedInstr::Body(InstrAt { instr, .. }) = try_lift(&insn, pc).unwrap() else {
            panic!("expected body instruction");
        };
        assert_eq!(
            instr.cfg_effect(),
            rvr_openvm_ir::CfgEffect::WriteConst {
                dst: Reg::new(1),
                value: pc + 0x100,
            }
        );
    }
}
