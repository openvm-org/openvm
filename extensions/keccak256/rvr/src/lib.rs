//! Keccak-256 extension: IR nodes for KECCAKF + XORIN and the
//! `KeccakExtension` that lifts and emits them.
//!
//! The keccak-f permutation runs in C against the keccak-ffi staticlib; the
//! `.c` shim is emitted alongside generated code so clang can inline the
//! tracer helpers across the call boundary.

use openvm_instructions::{
    riscv::{RV64_NUM_REGISTERS, RV64_REGISTER_BYTES},
    LocalOpcode,
};
use openvm_keccak256_transpiler::{KeccakfOpcode, XorinOpcode};
use rvr_openvm_ir::{
    CfgEffect, ExtEmitCtx, ExtInstr, FixedTraceRows, InstrAt, LiftedInstr, Variable,
};
use rvr_openvm_lift::{
    decode_variable, fixed_trace_rows_for_chip, max_main_memory_pages_for_contiguous_range,
    opcode_air_idx, AirIndex, ExtensionError, RvrExtension, RvrExtensionCtx, RvrInstruction,
};

fn decode_reg(value: u32) -> Variable {
    decode_variable(value, RV64_REGISTER_BYTES as u32, RV64_NUM_REGISTERS as u32)
}

const KECCAK_NUM_ROUNDS: u32 = p3_keccak_air::NUM_ROUNDS as u32;
const _: () = assert!(KECCAK_NUM_ROUNDS as usize == p3_keccak_air::NUM_ROUNDS);
// XORIN reads one 136-byte rate buffer, writes it back, and separately reads its input.
const XORIN_MAX_PAGES: usize = 3 * max_main_memory_pages_for_contiguous_range(136);
// KECCAKF reads and writes the 200-byte state in place.
const KECCAKF_MAX_PAGES: usize = 2 * max_main_memory_pages_for_contiguous_range(200);
const KECCAK_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION: usize = if XORIN_MAX_PAGES > KECCAKF_MAX_PAGES {
    XORIN_MAX_PAGES
} else {
    KECCAKF_MAX_PAGES
};

/// keccak-f\[1600\]: read 200 bytes via `buffer_ptr_reg`, permute in place.
#[derive(Debug, Clone)]
pub struct KeccakfInstr {
    pub buffer_ptr_reg: Variable,
    /// KeccakfPerm chip (24 rows per instruction).
    pub perm_chip_idx: Option<AirIndex>,
}

impl ExtInstr for KeccakfInstr {
    fn opname(&self) -> &str {
        "keccakf"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let buf = ctx.read_var(self.buffer_ptr_reg);
        ctx.reserve_preflight_writes("25u", "25u");
        ctx.reserve_replay_values("25u");
        ctx.emit_call("rvr_ext_keccakf", &["state", &buf]);
        ctx.append_replay_memory_u64_range(&buf, "25u");
    }

    fn fixed_trace_rows(&self) -> Vec<FixedTraceRows> {
        fixed_trace_rows_for_chip(self.perm_chip_idx, KECCAK_NUM_ROUNDS)
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }

    fn supports_checkpoint_preflight(&self) -> bool {
        true
    }
}

/// XORIN: XOR `len_reg` bytes from `input_ptr_reg` into `buffer_ptr_reg` in place.
#[derive(Debug, Clone)]
pub struct XorinInstr {
    pub buffer_ptr_reg: Variable,
    pub input_ptr_reg: Variable,
    pub len_reg: Variable,
}

impl ExtInstr for XorinInstr {
    fn opname(&self) -> &str {
        "xorin"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let buf_ptr = ctx.read_var(self.buffer_ptr_reg);
        let input = ctx.read_var(self.input_ptr_reg);
        let len = ctx.read_var(self.len_reg);
        let words = format!("((uint32_t)(({len} + 7ull) / 8ull))");
        ctx.reserve_preflight_writes(&words, &format!("{words} * 3u"));
        ctx.reserve_replay_values(&words);
        ctx.emit_checked_call("rvr_ext_xorin", &["state", &buf_ptr, &input, &len]);
        ctx.append_replay_memory_u64_range(&buf_ptr, &words);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }

    fn supports_checkpoint_preflight(&self) -> bool {
        true
    }
}

/// Keccak-256 extension. Register with the `ExtensionRegistry`.
pub struct KeccakExtension {
    keccakf_perm_chip_idx: Option<AirIndex>,
}

impl KeccakExtension {
    pub fn new(ctx: Option<&RvrExtensionCtx>) -> Result<Self, ExtensionError> {
        opcode_air_idx(ctx, XorinOpcode::XORIN)?;
        let keccakf_op_chip_idx = opcode_air_idx(ctx, KeccakfOpcode::KECCAKF)?;
        // KeccakfPerm is registered adjacent to KeccakfOp and assigned the next
        // AIR index (keccakf_op_chip_idx + 1) due to reverse registration order.
        let keccakf_perm_chip_idx = keccakf_op_chip_idx.map(AirIndex::next);

        Ok(Self {
            keccakf_perm_chip_idx,
        })
    }
}

impl RvrExtension for KeccakExtension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode == KeccakfOpcode::KECCAKF.global_opcode_usize() {
            let buffer_ptr_reg = decode_reg(insn.a);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Box::new(KeccakfInstr {
                    buffer_ptr_reg,
                    perm_chip_idx: self.keccakf_perm_chip_idx,
                }),
                source_loc: None,
            }));
        }

        if opcode == XorinOpcode::XORIN.global_opcode_usize() {
            let buffer_ptr_reg = decode_reg(insn.a);
            let input_ptr_reg = decode_reg(insn.b);
            let len_reg = decode_reg(insn.c);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Box::new(XorinInstr {
                    buffer_ptr_reg,
                    input_ptr_reg,
                    len_reg,
                }),
                source_loc: None,
            }));
        }

        None
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![("rvr_ext_keccak.h", include_str!("../c/rvr_ext_keccak.h"))]
    }

    fn c_sources(&self) -> Vec<(&'static str, &'static str)> {
        vec![("rvr_ext_keccak.c", include_str!("../c/rvr_ext_keccak.c"))]
    }

    fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        vec![(
            "librvr_openvm_ext_keccak_ffi.a",
            include_bytes!(env!("RVR_KECCAK_FFI_STATICLIB")),
        )]
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        KECCAK_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION
    }
}

#[cfg(test)]
mod tests {
    use rvr_openvm_ir::{MemWidth, PageAddressSpace};

    use super::*;

    struct TestEmitCtx {
        lines: Vec<String>,
        next_tmp: usize,
        record_checkpoint: bool,
    }

    impl Default for TestEmitCtx {
        fn default() -> Self {
            Self {
                lines: Vec::new(),
                next_tmp: 0,
                record_checkpoint: true,
            }
        }
    }

    impl TestEmitCtx {
        fn pure() -> Self {
            Self {
                record_checkpoint: false,
                ..Self::default()
            }
        }
    }

    impl ExtEmitCtx for TestEmitCtx {
        fn read_var(&mut self, var: Variable) -> String {
            let value = format!("r{}", var.index());
            self.lines.push(format!("read({value})"));
            value
        }

        fn peek_var(&mut self, var: Variable) -> String {
            format!("r{}", var.index())
        }

        fn advance_timestamp(&mut self, slots: u32) {
            self.lines.push(format!("advance({slots})"));
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

        fn reserve_preflight_writes(&mut self, writes: &str, slots: &str) {
            if self.record_checkpoint {
                self.lines.push(format!("reserve({writes}, {slots})"));
            }
        }

        fn reserve_replay_values(&mut self, count: &str) {
            if self.record_checkpoint {
                self.lines.push(format!("reserve_replay({count})"));
            }
        }

        fn append_replay_memory_u64_range(&mut self, base: &str, count: &str) {
            if self.record_checkpoint {
                self.lines.push(format!("append_range({base}, {count})"));
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

    #[test]
    fn keccakf_reserves_exact_schedule_and_postimage() {
        let instruction = KeccakfInstr {
            buffer_ptr_reg: Variable::new(5),
            perm_chip_idx: None,
        };
        assert!(!instruction.supports_preflight());
        assert!(instruction.supports_checkpoint_preflight());

        let mut ctx = TestEmitCtx::default();
        instruction.emit_c(&mut ctx);
        assert_eq!(
            ctx.lines,
            [
                "read(r5)",
                "reserve(25u, 25u)",
                "reserve_replay(25u)",
                "rvr_ext_keccakf(state, r5)",
                "append_range(r5, 25u)",
            ]
        );
    }

    #[test]
    fn xorin_reserves_three_slots_and_one_postimage_per_word() {
        let instruction = XorinInstr {
            buffer_ptr_reg: Variable::new(5),
            input_ptr_reg: Variable::new(6),
            len_reg: Variable::new(7),
        };
        assert!(!instruction.supports_preflight());
        assert!(instruction.supports_checkpoint_preflight());

        let mut ctx = TestEmitCtx::default();
        instruction.emit_c(&mut ctx);
        let words = "((uint32_t)((r7 + 7ull) / 8ull))";
        assert_eq!(ctx.lines[0..3], ["read(r5)", "read(r6)", "read(r7)"]);
        assert_eq!(ctx.lines[3], format!("reserve({words}, {words} * 3u)"));
        assert_eq!(ctx.lines[4], format!("reserve_replay({words})"));
        assert_eq!(ctx.lines[5], "bool tmp0 = rvr_ext_xorin(state, r5, r6, r7)");
        assert_eq!(ctx.lines[6], "if (unlikely(!tmp0)) {");
        assert_eq!(ctx.lines[7], "trap");
        assert_eq!(ctx.lines[8], "}");
        assert_eq!(ctx.lines[9], format!("append_range(r5, {words})"));
    }

    #[test]
    fn pure_emit_keeps_the_original_keccak_calls_byte_for_byte() {
        let keccakf = KeccakfInstr {
            buffer_ptr_reg: Variable::new(5),
            perm_chip_idx: None,
        };
        let mut ctx = TestEmitCtx::pure();
        keccakf.emit_c(&mut ctx);
        assert_eq!(ctx.lines, ["read(r5)", "rvr_ext_keccakf(state, r5)"]);

        let xorin = XorinInstr {
            buffer_ptr_reg: Variable::new(5),
            input_ptr_reg: Variable::new(6),
            len_reg: Variable::new(7),
        };
        let mut ctx = TestEmitCtx::pure();
        xorin.emit_c(&mut ctx);
        assert_eq!(
            ctx.lines,
            [
                "read(r5)",
                "read(r6)",
                "read(r7)",
                "bool tmp0 = rvr_ext_xorin(state, r5, r6, r7)",
                "if (unlikely(!tmp0)) {",
                "trap",
                "}",
            ]
        );
    }
}
