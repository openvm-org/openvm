//! Pairing extension for rvr-openvm.
//!
//! Provides an IR node for the `HintFinalExp` phantom instruction and the
//! `PairingExtension` for lifting and executing it via FFI.

use openvm_instructions::{
    riscv::{RV64_NUM_REGISTERS, RV64_REGISTER_BYTES},
    LocalOpcode, SystemOpcode,
};
use openvm_pairing_transpiler::PairingPhantom;
use rvr_openvm_ir::{CfgEffect, ExtEmitCtx, ExtInstr, InstrAt, LiftedInstr, Variable};
use rvr_openvm_lift::{decode_variable, RvrExtension, RvrInstruction};

fn decode_reg(value: u32) -> Variable {
    decode_variable(value, RV64_REGISTER_BYTES as u32, RV64_NUM_REGISTERS as u32)
}

#[derive(Debug, Clone, Copy)]
enum KnownPairingCurve {
    Bn254,
    Bls12_381,
}

impl KnownPairingCurve {
    fn from_idx(curve_idx: u16) -> Option<Self> {
        match curve_idx {
            0 => Some(Self::Bn254),
            1 => Some(Self::Bls12_381),
            _ => None,
        }
    }

    fn ffi_symbol(self) -> &'static str {
        match self {
            Self::Bn254 => "rvr_ext_pairing_hint_final_exp_bn254",
            Self::Bls12_381 => "rvr_ext_pairing_hint_final_exp_bls12_381",
        }
    }
}

/// IR node for the HintFinalExp phantom instruction.
///
/// At runtime, reads P (G1 points) and Q (G2 points) from memory via
/// register-indirect slice pointers, computes the multi-Miller loop and
/// final exponentiation hint, and sets the hint stream to the result.
#[derive(Debug, Clone)]
pub struct HintFinalExpInstr {
    /// Register holding pointer to P slice header (data_ptr, len).
    pub rs1_reg: Variable,
    /// Register holding pointer to Q slice header (data_ptr, len).
    pub rs2_reg: Variable,
    /// Pairing curve, resolved at lift time.
    curve: KnownPairingCurve,
}

impl ExtInstr for HintFinalExpInstr {
    fn opname(&self) -> &str {
        "hint_finalexp"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rs1 = ctx.peek_var(self.rs1_reg);
        let rs2 = ctx.peek_var(self.rs2_reg);
        ctx.emit_checked_call(self.curve.ffi_symbol(), &["state", &rs1, &rs2]);
        ctx.advance_timestamp(1);
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

/// The Pairing extension (HintFinalExp phantom instruction).
/// Register this with the `ExtensionRegistry`.
pub struct PairingExtension;

impl PairingExtension {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PairingExtension {
    fn default() -> Self {
        Self::new()
    }
}

impl RvrExtension for PairingExtension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode != SystemOpcode::PHANTOM.global_opcode_usize() {
            return None;
        }

        let c_val = insn.c;
        let discriminant = (c_val & 0xffff) as u16;
        let curve_idx = (c_val >> 16) as u16;

        if discriminant != PairingPhantom::HintFinalExp as u16 {
            return None;
        }

        let rs1_reg = decode_reg(insn.a);
        let rs2_reg = decode_reg(insn.b);
        let curve = KnownPairingCurve::from_idx(curve_idx)?;

        Some(LiftedInstr::Body(InstrAt {
            pc,
            instr: Box::new(HintFinalExpInstr {
                rs1_reg,
                rs2_reg,
                curve,
            }),
            source_loc: None,
        }))
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![("rvr_ext_pairing.h", include_str!("../c/rvr_ext_pairing.h"))]
    }

    fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        vec![(
            "librvr_openvm_ext_pairing_ffi.a",
            include_bytes!(env!("RVR_PAIRING_FFI_STATICLIB")),
        )]
    }

    fn requires_cxx_linker(&self) -> bool {
        true
    }

    fn uses_memory_wrappers(&self) -> bool {
        true
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        // Pairing's guest-memory reads record zero main-memory page entries.
        0
    }
}

#[cfg(test)]
mod tests {
    use rvr_openvm_ir::{MemWidth, PageAddressSpace};

    use super::*;

    #[derive(Default)]
    struct TestEmitCtx {
        operations: Vec<String>,
    }

    impl ExtEmitCtx for TestEmitCtx {
        fn read_var(&mut self, var: Variable) -> String {
            self.operations.push(format!("read(r{});", var.index()));
            format!("r{}", var.index())
        }

        fn peek_var(&mut self, var: Variable) -> String {
            format!("r{}", var.index())
        }

        fn advance_timestamp(&mut self, slots: u32) {
            self.operations.push(format!("advance_timestamp({slots});"));
        }

        fn write_var(&mut self, _var: Variable, _val: &str) {
            unreachable!()
        }

        fn write_line(&mut self, line: &str) {
            self.operations.push(line.to_string());
        }

        fn emit_trap(&mut self) {
            self.operations.push("trap;".to_string());
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

        fn reserve_preflight_writes(&mut self, _writes: &str, _slots: &str) {
            unreachable!()
        }

        fn emit_call(&mut self, name: &str, args: &[&str]) {
            self.operations
                .push(format!("{name}({});", args.join(", ")));
        }

        fn emit_call_without_page_flush(&mut self, _name: &str, _args: &[&str]) {
            unreachable!()
        }

        fn emit_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String {
            self.operations
                .push(format!("{ret_ty} result = {name}({});", args.join(", ")));
            "result".to_string()
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
    fn hint_final_exp_peeks_operands_and_advances_one_timestamp() {
        let instr = HintFinalExpInstr {
            rs1_reg: Variable::new(10),
            rs2_reg: Variable::new(11),
            curve: KnownPairingCurve::Bn254,
        };
        assert!(instr.supports_preflight());

        let mut ctx = TestEmitCtx::default();
        instr.emit_c(&mut ctx);

        assert!(!ctx.operations.iter().any(|op| op.starts_with("read(")));
        assert!(ctx.operations.iter().any(|op| {
            op == "bool result = rvr_ext_pairing_hint_final_exp_bn254(state, r10, r11);"
        }));
        assert!(ctx.operations.iter().any(|op| op == "trap;"));
        assert_eq!(
            ctx.operations.last().map(String::as_str),
            Some("advance_timestamp(1);")
        );
    }
}
