//! Int256 (256-bit integer) extension for rvr-openvm.
//!
//! Provides IR nodes for all Int256 opcodes (ALU, shift, comparison, multiplication,
//! and branch instructions) and the `Int256Extension` for lifting and executing them
//! via double FFI.

use std::path::{Path, PathBuf};

use openvm_bigint_transpiler::{
    Rv32BaseAlu256Opcode, Rv32BranchEqual256Opcode, Rv32BranchLessThan256Opcode,
    Rv32LessThan256Opcode, Rv32Mul256Opcode, Rv32Shift256Opcode,
};
use openvm_instructions::instruction::Instruction;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_instructions::LocalOpcode;
use openvm_rv32im_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, LessThanOpcode, MulOpcode, ShiftOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg, Terminator};
use rvr_openvm_lift::{RvrExtension, RvrExtensionCtx};
use strum::EnumCount;

// ── ALU / branch opcode enums ───────────────────────────────────────────────
//
// Used only at codegen time to select the specialized FFI function name. There
// is no runtime `op` parameter on the FFI — see crates/extensions/bigint/c/
// rvr_ext_bigint.h.

#[derive(Debug, Clone, Copy)]
pub enum Int256AluOp {
    Add,
    Sub,
    Xor,
    Or,
    And,
    Sll,
    Srl,
    Sra,
    Slt,
    Sltu,
    Mul,
}

impl Int256AluOp {
    fn ffi_name(self) -> &'static str {
        match self {
            Self::Add => "rvr_ext_int256_add",
            Self::Sub => "rvr_ext_int256_sub",
            Self::Xor => "rvr_ext_int256_xor",
            Self::Or => "rvr_ext_int256_or",
            Self::And => "rvr_ext_int256_and",
            Self::Sll => "rvr_ext_int256_sll",
            Self::Srl => "rvr_ext_int256_srl",
            Self::Sra => "rvr_ext_int256_sra",
            Self::Slt => "rvr_ext_int256_slt",
            Self::Sltu => "rvr_ext_int256_sltu",
            Self::Mul => "rvr_ext_int256_mul",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Int256BranchLtOp {
    Blt,
    Bltu,
    Bge,
    Bgeu,
}

impl Int256BranchLtOp {
    fn ffi_name(self) -> &'static str {
        match self {
            Self::Blt => "rvr_ext_int256_blt",
            Self::Bltu => "rvr_ext_int256_bltu",
            Self::Bge => "rvr_ext_int256_bge",
            Self::Bgeu => "rvr_ext_int256_bgeu",
        }
    }
}

// ── IR instruction nodes ────────────────────────────────────────────────────

/// IR node for a 256-bit ALU body instruction.
///
/// Covers ADD, SUB, XOR, OR, AND, SLL, SRL, SRA, SLT, SLTU, MUL.
/// All read two 256-bit operands via register pointers and write a 256-bit result.
#[derive(Debug, Clone)]
pub struct Int256AluInstr {
    /// Register index holding pointer to destination (rd).
    pub rd_reg: Reg,
    /// Register index holding pointer to first operand (rs1).
    pub rs1_reg: Reg,
    /// Register index holding pointer to second operand (rs2).
    pub rs2_reg: Reg,
    /// The ALU operation to perform (selects the FFI function at codegen time).
    pub op: Int256AluOp,
    /// Chip index for metering. Currently not consumed by the FFI (each
    /// instruction is one row already counted by the per-block chip update
    /// emitted at block entry), kept on the IR for parity with other
    /// extensions and future trace-chip use.
    pub chip_idx: u32,
}

impl ExtInstr for Int256AluInstr {
    fn opname(&self) -> &str {
        "int256_alu"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_reg(self.rd_reg);
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        ctx.write_line(&format!(
            "{}(state, {rd}, {rs1}, {rs2});",
            self.op.ffi_name()
        ));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// IR node for a 256-bit branch-equal terminator instruction (BEQ256 / BNE256).
#[derive(Debug, Clone)]
pub struct Int256BranchEqInstr {
    /// Register index holding pointer to first operand.
    pub rs1_reg: Reg,
    /// Register index holding pointer to second operand.
    pub rs2_reg: Reg,
    /// PC to jump to if condition is true.
    pub target_pc: u32,
    /// PC to fall through to if condition is false.
    pub fall_pc: u32,
    /// If true, branch on *not* equal (BNE); otherwise branch on equal (BEQ).
    pub is_ne: bool,
    /// Chip index for metering. See [`Int256AluInstr::chip_idx`].
    pub chip_idx: u32,
}

impl ExtInstr for Int256BranchEqInstr {
    fn opname(&self) -> &str {
        "int256_beq"
    }

    fn emit_c(&self, _ctx: &mut dyn ExtEmitCtx) {
        // Terminators use emit_c_term instead.
    }

    fn emit_c_term(&self, ctx: &mut dyn ExtEmitCtx, branch_to: &dyn Fn(u32) -> String) {
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        let fn_name = if self.is_ne {
            "rvr_ext_int256_bne"
        } else {
            "rvr_ext_int256_beq"
        };
        ctx.write_line(&format!("if ({fn_name}(state, {rs1}, {rs2})) {{"));
        ctx.write_line(&format!("  {}", branch_to(self.target_pc)));
        ctx.write_line("} else {");
        ctx.write_line(&format!("  {}", branch_to(self.fall_pc)));
        ctx.write_line("}");
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn successors(&self, _fall_pc: u32) -> Vec<u32> {
        vec![self.target_pc, self.fall_pc]
    }

    fn is_block_end(&self) -> bool {
        true
    }
}

/// IR node for a 256-bit branch-less-than terminator instruction (BLT/BLTU/BGE/BGEU).
#[derive(Debug, Clone)]
pub struct Int256BranchLtInstr {
    /// Register index holding pointer to first operand.
    pub rs1_reg: Reg,
    /// Register index holding pointer to second operand.
    pub rs2_reg: Reg,
    /// PC to jump to if condition is true.
    pub target_pc: u32,
    /// PC to fall through to if condition is false.
    pub fall_pc: u32,
    /// The branch-less-than variant (selects the FFI function at codegen time).
    pub op: Int256BranchLtOp,
    /// Chip index for metering. See [`Int256AluInstr::chip_idx`].
    pub chip_idx: u32,
}

impl ExtInstr for Int256BranchLtInstr {
    fn opname(&self) -> &str {
        "int256_blt"
    }

    fn emit_c(&self, _ctx: &mut dyn ExtEmitCtx) {
        // Terminators use emit_c_term instead.
    }

    fn emit_c_term(&self, ctx: &mut dyn ExtEmitCtx, branch_to: &dyn Fn(u32) -> String) {
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        ctx.write_line(&format!(
            "if ({}(state, {rs1}, {rs2})) {{",
            self.op.ffi_name()
        ));
        ctx.write_line(&format!("  {}", branch_to(self.target_pc)));
        ctx.write_line("} else {");
        ctx.write_line(&format!("  {}", branch_to(self.fall_pc)));
        ctx.write_line("}");
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn successors(&self, _fall_pc: u32) -> Vec<u32> {
        vec![self.target_pc, self.fall_pc]
    }

    fn is_block_end(&self) -> bool {
        true
    }
}

// ── Extension struct ────────────────────────────────────────────────────────

/// The Int256 extension. Register this with the `ExtensionRegistry`.
pub struct Int256Extension {
    base_alu_chip_idx: u32,
    shift_chip_idx: u32,
    less_than_chip_idx: u32,
    mul_chip_idx: u32,
    branch_eq_chip_idx: u32,
    branch_lt_chip_idx: u32,
    staticlib_path: PathBuf,
}

impl Int256Extension {
    /// Create a `Int256Extension` for pure execution where chip indices
    /// don't matter (trace_chip is a no-op in pure mode).
    pub fn new_pure(staticlib_path: PathBuf) -> Self {
        Self {
            base_alu_chip_idx: u32::MAX,
            shift_chip_idx: u32::MAX,
            less_than_chip_idx: u32::MAX,
            mul_chip_idx: u32::MAX,
            branch_eq_chip_idx: u32::MAX,
            branch_lt_chip_idx: u32::MAX,
            staticlib_path,
        }
    }

    /// Create a new `Int256Extension`, resolving chip indices from the VM config.
    pub fn new(ctx: &RvrExtensionCtx, staticlib_path: PathBuf) -> Self {
        let base_alu_chip_idx =
            ctx.require_opcode_air_idx(Rv32BaseAlu256Opcode(BaseAluOpcode::ADD).global_opcode());
        let shift_chip_idx =
            ctx.require_opcode_air_idx(Rv32Shift256Opcode(ShiftOpcode::SLL).global_opcode());
        let less_than_chip_idx =
            ctx.require_opcode_air_idx(Rv32LessThan256Opcode(LessThanOpcode::SLT).global_opcode());
        let mul_chip_idx =
            ctx.require_opcode_air_idx(Rv32Mul256Opcode(MulOpcode::MUL).global_opcode());
        let branch_eq_chip_idx = ctx
            .require_opcode_air_idx(Rv32BranchEqual256Opcode(BranchEqualOpcode::BEQ).global_opcode());
        let branch_lt_chip_idx = ctx.require_opcode_air_idx(
            Rv32BranchLessThan256Opcode(BranchLessThanOpcode::BLT).global_opcode(),
        );

        Self {
            base_alu_chip_idx,
            shift_chip_idx,
            less_than_chip_idx,
            mul_chip_idx,
            branch_eq_chip_idx,
            branch_lt_chip_idx,
            staticlib_path,
        }
    }

    /// Map a global opcode to the chip index for that operation.
    fn chip_idx_for_opcode(&self, opcode: usize) -> u32 {
        let base_alu_start = Rv32BaseAlu256Opcode::CLASS_OFFSET;
        let shift_start = Rv32Shift256Opcode::CLASS_OFFSET;
        let lt_start = Rv32LessThan256Opcode::CLASS_OFFSET;
        let beq_start = Rv32BranchEqual256Opcode::CLASS_OFFSET;
        let blt_start = Rv32BranchLessThan256Opcode::CLASS_OFFSET;
        let mul_start = Rv32Mul256Opcode::CLASS_OFFSET;

        if opcode >= base_alu_start && opcode < base_alu_start + BaseAluOpcode::COUNT {
            self.base_alu_chip_idx
        } else if opcode >= shift_start && opcode < shift_start + ShiftOpcode::COUNT {
            self.shift_chip_idx
        } else if opcode >= lt_start && opcode < lt_start + LessThanOpcode::COUNT {
            self.less_than_chip_idx
        } else if opcode >= beq_start && opcode < beq_start + BranchEqualOpcode::COUNT {
            self.branch_eq_chip_idx
        } else if opcode >= blt_start && opcode < blt_start + BranchLessThanOpcode::COUNT {
            self.branch_lt_chip_idx
        } else if opcode >= mul_start && opcode < mul_start + MulOpcode::COUNT {
            self.mul_chip_idx
        } else {
            panic!("unknown Int256 opcode: {opcode:#x}");
        }
    }
}

/// Decode register index from OpenVM operand (divided by RV32_REGISTER_NUM_LIMBS).
fn decode_reg<F: PrimeField32>(f: F) -> u8 {
    (f.as_canonical_u32() / RV32_REGISTER_NUM_LIMBS as u32) as u8
}

/// Decode a field element as a signed immediate (for branch offsets).
fn decode_imm<F: PrimeField32>(f: F) -> i32 {
    let v = f.as_canonical_u32();
    let p = F::ORDER_U32;
    if v > p / 2 {
        v.wrapping_sub(p) as i32
    } else {
        v as i32
    }
}

const DEFAULT_PC_STEP: u32 = 4;

impl<F: PrimeField32> RvrExtension<F> for Int256Extension {
    fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        // ── ALU body instructions ───────────────────────────────────────

        // BaseAlu256: ADD(0), SUB(1), XOR(2), OR(3), AND(4)
        let base_alu_start = Rv32BaseAlu256Opcode::CLASS_OFFSET;
        if opcode >= base_alu_start && opcode < base_alu_start + BaseAluOpcode::COUNT {
            let op = match opcode - base_alu_start {
                0 => Int256AluOp::Add,
                1 => Int256AluOp::Sub,
                2 => Int256AluOp::Xor,
                3 => Int256AluOp::Or,
                4 => Int256AluOp::And,
                _ => unreachable!(),
            };
            return Some(self.lift_alu(insn, pc, op));
        }

        // Shift256: SLL(0), SRL(1), SRA(2)
        let shift_start = Rv32Shift256Opcode::CLASS_OFFSET;
        if opcode >= shift_start && opcode < shift_start + ShiftOpcode::COUNT {
            let op = match opcode - shift_start {
                0 => Int256AluOp::Sll,
                1 => Int256AluOp::Srl,
                2 => Int256AluOp::Sra,
                _ => unreachable!(),
            };
            return Some(self.lift_alu(insn, pc, op));
        }

        // LessThan256: SLT(0), SLTU(1)
        let lt_start = Rv32LessThan256Opcode::CLASS_OFFSET;
        if opcode >= lt_start && opcode < lt_start + LessThanOpcode::COUNT {
            let op = match opcode - lt_start {
                0 => Int256AluOp::Slt,
                1 => Int256AluOp::Sltu,
                _ => unreachable!(),
            };
            return Some(self.lift_alu(insn, pc, op));
        }

        // Mul256: MUL(0)
        let mul_start = Rv32Mul256Opcode::CLASS_OFFSET;
        if opcode >= mul_start && opcode < mul_start + MulOpcode::COUNT {
            return Some(self.lift_alu(insn, pc, Int256AluOp::Mul));
        }

        // ── Branch terminator instructions ──────────────────────────────

        // BranchEqual256: BEQ(0), BNE(1)
        let beq_start = Rv32BranchEqual256Opcode::CLASS_OFFSET;
        if opcode >= beq_start && opcode < beq_start + BranchEqualOpcode::COUNT {
            let is_ne = opcode - beq_start == 1;
            let rs1_reg = decode_reg(insn.a);
            let rs2_reg = decode_reg(insn.b);
            let imm = decode_imm(insn.c);
            let target_pc = (pc as i32 + imm) as u32;
            let fall_pc = pc + DEFAULT_PC_STEP;
            let chip_idx = self.chip_idx_for_opcode(opcode);

            return Some(LiftedInstr::Term {
                pc,
                terminator: Terminator::Extension(Box::new(Int256BranchEqInstr {
                    rs1_reg,
                    rs2_reg,
                    target_pc,
                    fall_pc,
                    is_ne,
                    chip_idx,
                })),
                source_loc: None,
            });
        }

        // BranchLessThan256: BLT(0), BLTU(1), BGE(2), BGEU(3)
        let blt_start = Rv32BranchLessThan256Opcode::CLASS_OFFSET;
        if opcode >= blt_start && opcode < blt_start + BranchLessThanOpcode::COUNT {
            let op = match opcode - blt_start {
                0 => Int256BranchLtOp::Blt,
                1 => Int256BranchLtOp::Bltu,
                2 => Int256BranchLtOp::Bge,
                3 => Int256BranchLtOp::Bgeu,
                _ => unreachable!(),
            };
            let rs1_reg = decode_reg(insn.a);
            let rs2_reg = decode_reg(insn.b);
            let imm = decode_imm(insn.c);
            let target_pc = (pc as i32 + imm) as u32;
            let fall_pc = pc + DEFAULT_PC_STEP;
            let chip_idx = self.chip_idx_for_opcode(opcode);

            return Some(LiftedInstr::Term {
                pc,
                terminator: Terminator::Extension(Box::new(Int256BranchLtInstr {
                    rs1_reg,
                    rs2_reg,
                    target_pc,
                    fall_pc,
                    op,
                    chip_idx,
                })),
                source_loc: None,
            });
        }

        None
    }

    fn c_headers(&self) -> Vec<(&str, &str)> {
        vec![("rvr_ext_bigint.h", include_str!("../c/rvr_ext_bigint.h"))]
    }

    fn staticlib_path(&self) -> &Path {
        &self.staticlib_path
    }
}

impl Int256Extension {
    /// Lift an R-type ALU instruction: a=rd, b=rs1, c=rs2.
    fn lift_alu<F: PrimeField32>(
        &self,
        insn: &Instruction<F>,
        pc: u32,
        op: Int256AluOp,
    ) -> LiftedInstr {
        let rd_reg = decode_reg(insn.a);
        let rs1_reg = decode_reg(insn.b);
        let rs2_reg = decode_reg(insn.c);
        let chip_idx = self.chip_idx_for_opcode(insn.opcode.as_usize());

        LiftedInstr::Body(InstrAt {
            pc,
            instr: Instr::Ext(Box::new(Int256AluInstr {
                rd_reg,
                rs1_reg,
                rs2_reg,
                op,
                chip_idx,
            })),
            source_loc: None,
        })
    }
}
