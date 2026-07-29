//! Int256 (256-bit integer) extension for rvr-openvm.
//!
//! Provides IR nodes for all Int256 opcodes (ALU, shift, comparison, multiplication,
//! and branch instructions) and the `Int256Extension` for lifting and executing them
//! via double FFI.

use openvm_bigint_transpiler::{
    Rv64BaseAlu256Opcode, Rv64BranchEqual256Opcode, Rv64BranchLessThan256Opcode,
    Rv64LessThan256Opcode, Rv64Mul256Opcode, Rv64Shift256Opcode,
};
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV64_NUM_REGISTERS, RV64_REGISTER_BYTES},
    LocalOpcode,
};
use openvm_riscv_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, LessThanOpcode, MulOpcode, ShiftOpcode,
};
use rvr_openvm_ir::{
    CfgEffect, CfgTerm, ExtEmitCtx, ExtInstr, InstrAt, LiftedInstr, Terminator, Variable,
};
use rvr_openvm_lift::{
    decode_variable, max_main_memory_pages_for_contiguous_range, RvrExtension, RvrInstruction,
};
use strum::EnumCount;

// An Int256 operation can read two independent 32-byte values and write one.
const INT256_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION: usize =
    3 * max_main_memory_pages_for_contiguous_range(32);

fn decode_reg(value: u32) -> Variable {
    decode_variable(value, RV64_REGISTER_BYTES as u32, RV64_NUM_REGISTERS as u32)
}

fn emit_pointer_alignment_guard(ctx: &mut dyn ExtEmitCtx, pointers: &[&str]) {
    let pointers = pointers.join(" | ");
    ctx.write_line(&format!(
        "if (unlikely((({pointers}) & {}ull) != 0ull)) {{",
        RV64_REGISTER_BYTES - 1
    ));
    ctx.emit_trap();
    ctx.write_line("}");
}

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
    pub rd_reg: Variable,
    /// Register index holding pointer to first operand (rs1).
    pub rs1_reg: Variable,
    /// Register index holding pointer to second operand (rs2).
    pub rs2_reg: Variable,
    /// The ALU operation to perform (selects the FFI function at codegen time).
    pub op: Int256AluOp,
}

impl ExtInstr for Int256AluInstr {
    fn opname(&self) -> &str {
        "int256_alu"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rs1 = ctx.read_var(self.rs1_reg);
        let rs2 = ctx.read_var(self.rs2_reg);
        let rd = ctx.read_var(self.rd_reg);
        emit_pointer_alignment_guard(ctx, &[&rd, &rs1, &rs2]);
        // The FFI performs eight aligned heap reads followed by four aligned
        // heap writes. Checkpoint replay reconstructs those events from the
        // postimage; pure and metered modes emit neither reservation nor peek.
        ctx.reserve_preflight_timestamp_slots("12u");
        let checkpoint = ctx.is_checkpoint_preflight();
        if checkpoint {
            ctx.reserve_replay_values("4u");
        } else if ctx.counts_checkpoint_residuals() {
            ctx.count_fixed_replay_values(4);
        }
        ctx.emit_call(self.op.ffi_name(), &["state", &rd, &rs1, &rs2]);
        if checkpoint {
            ctx.append_replay_memory_u64_range(&rd, "4u");
        }
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }

    fn supports_preflight(&self) -> bool {
        true
    }
}

/// IR node for a 256-bit branch-equal terminator instruction (BEQ256 / BNE256).
#[derive(Debug, Clone)]
pub struct Int256BranchEqInstr {
    /// Register index holding pointer to first operand.
    pub rs1_reg: Variable,
    /// Register index holding pointer to second operand.
    pub rs2_reg: Variable,
    /// PC to jump to if condition is true.
    pub target_pc: u64,
    /// PC to fall through to if condition is false.
    pub fall_pc: u64,
    /// If true, branch on *not* equal (BNE); otherwise branch on equal (BEQ).
    pub is_ne: bool,
}

impl ExtInstr for Int256BranchEqInstr {
    fn opname(&self) -> &str {
        "int256_beq"
    }

    fn emit_c(&self, _ctx: &mut dyn ExtEmitCtx) {
        // Terminators use emit_c_term instead.
    }

    fn emit_c_term(&self, ctx: &mut dyn ExtEmitCtx, branch_to: &dyn Fn(u64) -> String) {
        let rs1 = ctx.read_var(self.rs1_reg);
        let rs2 = ctx.read_var(self.rs2_reg);
        let fn_name = if self.is_ne {
            "rvr_ext_int256_bne"
        } else {
            "rvr_ext_int256_beq"
        };
        emit_pointer_alignment_guard(ctx, &[&rs1, &rs2]);
        let cond = ctx.emit_call_expr("bool", fn_name, &["state", &rs1, &rs2]);
        // The predicate call performs eight aligned heap reads. Its one-bit
        // result is the minimum information needed by independent GPU chunks
        // to recover the dynamic successor before memory chronology exists.
        ctx.advance_checkpoint_timestamp(8);
        ctx.append_replay_value(&cond);
        ctx.flush_before_control_transfer();
        ctx.write_line(&format!("if ({cond}) {{"));
        ctx.write_line(&format!("  {}", branch_to(self.target_pc)));
        ctx.write_line("} else {");
        ctx.write_line(&format!("  {}", branch_to(self.fall_pc)));
        ctx.write_line("}");
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }

    fn cfg_term(&self, _pc: u64, _fall_pc: u64) -> Option<CfgTerm> {
        Some(CfgTerm::Opaque {
            successors: vec![self.target_pc, self.fall_pc],
        })
    }

    fn supports_preflight(&self) -> bool {
        true
    }
}

/// IR node for a 256-bit branch-less-than terminator instruction (BLT/BLTU/BGE/BGEU).
#[derive(Debug, Clone)]
pub struct Int256BranchLtInstr {
    /// Register index holding pointer to first operand.
    pub rs1_reg: Variable,
    /// Register index holding pointer to second operand.
    pub rs2_reg: Variable,
    /// PC to jump to if condition is true.
    pub target_pc: u64,
    /// PC to fall through to if condition is false.
    pub fall_pc: u64,
    /// The branch-less-than variant (selects the FFI function at codegen time).
    pub op: Int256BranchLtOp,
}

impl ExtInstr for Int256BranchLtInstr {
    fn opname(&self) -> &str {
        "int256_blt"
    }

    fn emit_c(&self, _ctx: &mut dyn ExtEmitCtx) {
        // Terminators use emit_c_term instead.
    }

    fn emit_c_term(&self, ctx: &mut dyn ExtEmitCtx, branch_to: &dyn Fn(u64) -> String) {
        let rs1 = ctx.read_var(self.rs1_reg);
        let rs2 = ctx.read_var(self.rs2_reg);
        emit_pointer_alignment_guard(ctx, &[&rs1, &rs2]);
        let cond = ctx.emit_call_expr("bool", self.op.ffi_name(), &["state", &rs1, &rs2]);
        ctx.advance_checkpoint_timestamp(8);
        ctx.append_replay_value(&cond);
        ctx.flush_before_control_transfer();
        ctx.write_line(&format!("if ({cond}) {{"));
        ctx.write_line(&format!("  {}", branch_to(self.target_pc)));
        ctx.write_line("} else {");
        ctx.write_line(&format!("  {}", branch_to(self.fall_pc)));
        ctx.write_line("}");
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }

    fn cfg_term(&self, _pc: u64, _fall_pc: u64) -> Option<CfgTerm> {
        Some(CfgTerm::Opaque {
            successors: vec![self.target_pc, self.fall_pc],
        })
    }

    fn supports_preflight(&self) -> bool {
        true
    }
}

// ── Extension struct ────────────────────────────────────────────────────────

/// The Int256 extension. Register this with the `ExtensionRegistry`.
pub struct Int256Extension;

impl Int256Extension {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for Int256Extension {
    fn default() -> Self {
        Self::new()
    }
}

impl RvrExtension for Int256Extension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        // ── ALU body instructions ───────────────────────────────────────

        // BaseAlu256: ADD(0), SUB(1), XOR(2), OR(3), AND(4)
        let base_alu_start = Rv64BaseAlu256Opcode::CLASS_OFFSET;
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
        let shift_start = Rv64Shift256Opcode::CLASS_OFFSET;
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
        let lt_start = Rv64LessThan256Opcode::CLASS_OFFSET;
        if opcode >= lt_start && opcode < lt_start + LessThanOpcode::COUNT {
            let op = match opcode - lt_start {
                0 => Int256AluOp::Slt,
                1 => Int256AluOp::Sltu,
                _ => unreachable!(),
            };
            return Some(self.lift_alu(insn, pc, op));
        }

        // Mul256: MUL(0)
        let mul_start = Rv64Mul256Opcode::CLASS_OFFSET;
        if opcode >= mul_start && opcode < mul_start + MulOpcode::COUNT {
            return Some(self.lift_alu(insn, pc, Int256AluOp::Mul));
        }

        // ── Branch terminator instructions ──────────────────────────────

        // BranchEqual256: BEQ(0), BNE(1)
        let beq_start = Rv64BranchEqual256Opcode::CLASS_OFFSET;
        if opcode >= beq_start && opcode < beq_start + BranchEqualOpcode::COUNT {
            let is_ne = opcode - beq_start == 1;
            let rs1_reg = decode_reg(insn.a);
            let rs2_reg = decode_reg(insn.b);
            let imm = insn.signed_c();
            let target_pc = (pc as i64 + imm as i64) as u64;
            let fall_pc = pc + DEFAULT_PC_STEP as u64;
            return Some(LiftedInstr::Term {
                pc,
                terminator: Terminator::instruction(Int256BranchEqInstr {
                    rs1_reg,
                    rs2_reg,
                    target_pc,
                    fall_pc,
                    is_ne,
                }),
                source_loc: None,
            });
        }

        // BranchLessThan256: BLT(0), BLTU(1), BGE(2), BGEU(3)
        let blt_start = Rv64BranchLessThan256Opcode::CLASS_OFFSET;
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
            let imm = insn.signed_c();
            let target_pc = (pc as i64 + imm as i64) as u64;
            let fall_pc = pc + DEFAULT_PC_STEP as u64;
            return Some(LiftedInstr::Term {
                pc,
                terminator: Terminator::instruction(Int256BranchLtInstr {
                    rs1_reg,
                    rs2_reg,
                    target_pc,
                    fall_pc,
                    op,
                }),
                source_loc: None,
            });
        }

        None
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![("rvr_ext_bigint.h", include_str!("../c/rvr_ext_bigint.h"))]
    }

    fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        vec![(
            "librvr_openvm_ext_bigint_ffi.a",
            include_bytes!(env!("RVR_BIGINT_FFI_STATICLIB")),
        )]
    }

    fn uses_memory_wrappers(&self) -> bool {
        true
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        INT256_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION
    }
}

impl Int256Extension {
    /// Lift an R-type ALU instruction: a=rd, b=rs1, c=rs2.
    fn lift_alu(&self, insn: &RvrInstruction, pc: u64, op: Int256AluOp) -> LiftedInstr {
        let rd_reg = decode_reg(insn.a);
        let rs1_reg = decode_reg(insn.b);
        let rs2_reg = decode_reg(insn.c);
        LiftedInstr::Body(InstrAt {
            pc,
            instr: Box::new(Int256AluInstr {
                rd_reg,
                rs1_reg,
                rs2_reg,
                op,
            }),
            source_loc: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use rvr_openvm_ir::{MemWidth, PageAddressSpace};

    use super::*;

    struct TestEmitCtx {
        lines: Vec<String>,
        next_tmp: usize,
        mode: TestEmitMode,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum TestEmitMode {
        Checkpoint,
        Direct,
        Metered,
    }

    impl Default for TestEmitCtx {
        fn default() -> Self {
            Self {
                lines: Vec::new(),
                next_tmp: 0,
                mode: TestEmitMode::Checkpoint,
            }
        }
    }

    impl TestEmitCtx {
        fn execution(mode: TestEmitMode) -> Self {
            Self {
                mode,
                ..Self::default()
            }
        }

        fn records_checkpoint(&self) -> bool {
            self.mode == TestEmitMode::Checkpoint
        }
    }

    impl ExtEmitCtx for TestEmitCtx {
        fn is_checkpoint_preflight(&self) -> bool {
            self.records_checkpoint()
        }

        fn read_var(&mut self, var: Variable) -> String {
            let value = format!("r{}", var.index());
            self.lines.push(format!("read({value})"));
            value
        }

        fn peek_var(&mut self, var: Variable) -> String {
            format!("r{}", var.index())
        }

        fn advance_timestamp(&mut self, slots: u32) {
            if self.records_checkpoint() {
                self.lines.push(format!("advance({slots})"));
            }
        }

        fn advance_checkpoint_timestamp(&mut self, slots: u32) {
            if self.records_checkpoint() {
                self.lines.push(format!("advance_checkpoint({slots})"));
            }
        }

        fn write_var(&mut self, _var: Variable, _val: &str) {
            unreachable!()
        }

        fn write_line(&mut self, line: &str) {
            self.lines.push(line.to_string());
        }

        fn emit_trap(&mut self) {
            self.lines.push("trap".to_string());
        }

        fn read_mem(&mut self, _base: &str, _offset: i16, _width: u8, _signed: bool) -> String {
            unreachable!()
        }

        fn write_mem(&mut self, _base: &str, _offset: i16, _val: &str, _width: u8) {
            unreachable!()
        }

        fn write_aligned_mem_block(&mut self, _addr: &str, _val: &str) {
            unreachable!()
        }

        fn reserve_preflight_timestamp_slots(&mut self, slots: &str) {
            if self.records_checkpoint() {
                self.lines.push(format!("reserve({slots})"));
            }
        }

        fn reserve_replay_values(&mut self, count: &str) {
            if self.records_checkpoint() {
                self.lines.push(format!("reserve_replay({count})"));
            }
        }

        fn append_replay_value(&mut self, value: &str) {
            if self.records_checkpoint() {
                self.lines.push(format!("append({value})"));
            }
        }

        fn append_replay_memory_u64_range(&mut self, base: &str, count: &str) {
            if self.records_checkpoint() {
                self.lines.push(format!("append_range({base}, {count})"));
            }
        }

        fn flush_before_control_transfer(&mut self) {
            if self.records_checkpoint() {
                self.lines.push("flush".to_string());
            }
        }

        fn emit_call(&mut self, name: &str, args: &[&str]) {
            self.lines.push(format!("{name}({})", args.join(", ")));
        }

        fn emit_call_without_page_flush(&mut self, name: &str, args: &[&str]) {
            self.emit_call(name, args);
        }

        fn emit_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String {
            let tmp = format!("tmp{}", self.next_tmp);
            self.next_tmp += 1;
            self.lines
                .push(format!("{ret_ty} {tmp} = {name}({})", args.join(", ")));
            tmp
        }

        fn emit_call_with_trace_result(
            &mut self,
            _ret_ty: &str,
            _name: &str,
            _args: &[&str],
        ) -> Option<String> {
            unreachable!()
        }

        fn trace_chip(&mut self, _chip_idx: u32, _count_expr: &str) {
            unreachable!()
        }

        fn trace_chip_if_nonzero(&mut self, _chip_idx: u32, _count_expr: &str) {
            unreachable!()
        }

        fn trace_page_access(
            &mut self,
            _addr: &str,
            _width: MemWidth,
            _addr_space: PageAddressSpace,
        ) {
            unreachable!()
        }

        fn trace_page_access_u64_range(
            &mut self,
            _base_addr: &str,
            _num_dwords: &str,
            _addr_space: PageAddressSpace,
        ) {
            unreachable!()
        }
    }

    fn instruction(opcode: impl LocalOpcode, c: u32) -> RvrInstruction {
        RvrInstruction::from_canonical(opcode.global_opcode(), [8, 16, c, 0, 0, 0, 0], 101)
    }

    #[test]
    fn bigint_branches_preserve_negative_field_encoded_offsets() {
        let pc = 0x1000;
        let ext = Int256Extension::new();

        for insn in [
            instruction(Rv64BranchEqual256Opcode(BranchEqualOpcode::BEQ), 101 - 12),
            instruction(
                Rv64BranchLessThan256Opcode(BranchLessThanOpcode::BLT),
                101 - 12,
            ),
        ] {
            let lifted = ext.try_lift(&insn, pc).unwrap();
            let LiftedInstr::Term { terminator, .. } = lifted else {
                panic!("expected bigint branch terminator");
            };
            assert_eq!(
                terminator.successors(pc, pc + DEFAULT_PC_STEP as u64),
                [pc - 12, pc + 4]
            );
        }
    }

    #[test]
    fn int256_alu_checkpoint_emits_exact_schedule_and_postimage() {
        let instruction = Int256AluInstr {
            rd_reg: Variable::new(1),
            rs1_reg: Variable::new(2),
            rs2_reg: Variable::new(3),
            op: Int256AluOp::Add,
        };
        assert!(instruction.supports_preflight());

        let mut checkpoint = TestEmitCtx::default();
        instruction.emit_c(&mut checkpoint);
        assert_eq!(
            checkpoint.lines,
            [
                "read(r2)",
                "read(r3)",
                "read(r1)",
                "if (unlikely(((r1 | r2 | r3) & 7ull) != 0ull)) {",
                "trap",
                "}",
                "reserve(12u)",
                "reserve_replay(4u)",
                "rvr_ext_int256_add(state, r1, r2, r3)",
                "append_range(r1, 4u)",
            ]
        );

        for mode in [TestEmitMode::Direct, TestEmitMode::Metered] {
            let mut execution = TestEmitCtx::execution(mode);
            instruction.emit_c(&mut execution);
            assert_eq!(
                execution.lines,
                [
                    "read(r2)",
                    "read(r3)",
                    "read(r1)",
                    "if (unlikely(((r1 | r2 | r3) & 7ull) != 0ull)) {",
                    "trap",
                    "}",
                    "rvr_ext_int256_add(state, r1, r2, r3)",
                ]
            );
        }
    }

    #[test]
    fn int256_branches_checkpoint_emit_only_the_decision_residual() {
        let instruction = Int256BranchEqInstr {
            rs1_reg: Variable::new(2),
            rs2_reg: Variable::new(3),
            target_pc: 40,
            fall_pc: 44,
            is_ne: false,
        };
        assert!(instruction.supports_preflight());

        let mut checkpoint = TestEmitCtx::default();
        instruction.emit_c_term(&mut checkpoint, &|pc| format!("goto_{pc}"));
        assert_eq!(
            checkpoint.lines,
            [
                "read(r2)",
                "read(r3)",
                "if (unlikely(((r2 | r3) & 7ull) != 0ull)) {",
                "trap",
                "}",
                "bool tmp0 = rvr_ext_int256_beq(state, r2, r3)",
                "advance_checkpoint(8)",
                "append(tmp0)",
                "flush",
                "if (tmp0) {",
                "  goto_40",
                "} else {",
                "  goto_44",
                "}",
            ]
        );

        let mut pure = TestEmitCtx::execution(TestEmitMode::Direct);
        instruction.emit_c_term(&mut pure, &|pc| format!("goto_{pc}"));
        assert_eq!(
            pure.lines,
            [
                "read(r2)",
                "read(r3)",
                "if (unlikely(((r2 | r3) & 7ull) != 0ull)) {",
                "trap",
                "}",
                "bool tmp0 = rvr_ext_int256_beq(state, r2, r3)",
                "if (tmp0) {",
                "  goto_40",
                "} else {",
                "  goto_44",
                "}",
            ]
        );

        let instruction = Int256BranchLtInstr {
            rs1_reg: Variable::new(2),
            rs2_reg: Variable::new(3),
            target_pc: 40,
            fall_pc: 44,
            op: Int256BranchLtOp::Bltu,
        };
        let mut checkpoint = TestEmitCtx::default();
        instruction.emit_c_term(&mut checkpoint, &|pc| format!("goto_{pc}"));
        assert_eq!(checkpoint.lines[6], "advance_checkpoint(8)");
        assert_eq!(checkpoint.lines[8], "flush");
        assert_eq!(
            checkpoint.lines[5],
            "bool tmp0 = rvr_ext_int256_bltu(state, r2, r3)"
        );
    }
}
