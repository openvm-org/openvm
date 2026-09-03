//! Modular arithmetic IR nodes, phantom hints, and the
//! [`ModularRvrExtension`] lifter.

use num_bigint::BigUint;
use openvm_algebra_transpiler::{ModularArithmeticOpcode, ModularPhantom};
use openvm_algebra_utils::{find_non_qr, NQR_RNG_SEED};
#[cfg(test)]
use openvm_instructions::MEMORY_BLOCK_BYTES;
use openvm_instructions::{
    instruction::Instruction,
    riscv::{MEMORY_AS, REGISTER_AS},
    LocalOpcode, SystemOpcode,
};
use rand::{rngs::StdRng, SeedableRng};
use rvr_openvm_ir::{CfgEffect, ExtEmitCtx, ExtInstr, InstrAt, LiftedInstr, Variable};
use rvr_openvm_lift::{max_main_memory_pages_for_contiguous_range, RvrExtension};
use strum::EnumCount;

#[cfg(test)]
use crate::BINARY_INPUTS_AND_OUTPUT;
use crate::{
    decode_reg, emit_word_alignment_guard, format_c_byte_array, pad_modulus, ArithKind,
    FieldArithInstr, FieldIsEqInstr, FieldKind, FieldSetupInstr, IsEqKind, KnownField, ModOp,
    SetupKind, BINARY_INPUTS, MEMORY_BLOCK_BYTES_U32,
};

include!(concat!(env!("OUT_DIR"), "/secp256k1_files.rs"));

// A modular operation can read two independent 48-byte values and write one.
const MODULAR_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION: usize =
    3 * max_main_memory_pages_for_contiguous_range(48);

/// Per-modulus info for the modular extension. Includes a precomputed non-QR
/// for the `HintNonQr` / `HintSqrt` phantoms.
struct ModulusInfo {
    modulus_bytes: Vec<u8>,
    non_qr_bytes: Vec<u8>,
    num_limbs: u32,
}

fn make_moduli(moduli: Vec<BigUint>) -> Vec<ModulusInfo> {
    // Use the same deterministic seed as the circuit-side `NonQrHintSubEx::new`
    // (single rng across the full modulus list), so rvr-emitted NQRs match
    // what the circuit would compute.
    let mut rng = StdRng::from_seed(NQR_RNG_SEED);
    moduli
        .into_iter()
        .map(|m| make_modulus_info(&m, &mut rng))
        .collect()
}

fn make_modulus_info(modulus: &BigUint, rng: &mut StdRng) -> ModulusInfo {
    let (modulus_bytes, num_limbs) = pad_modulus(modulus);
    let non_qr = find_non_qr(modulus, rng);
    let mut non_qr_bytes = non_qr.to_bytes_le();
    non_qr_bytes.resize(num_limbs as usize, 0);
    ModulusInfo {
        modulus_bytes,
        non_qr_bytes,
        num_limbs,
    }
}

// ── Modular arithmetic IR ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct ModArithKind;

impl FieldKind for ModArithKind {
    const SETUP_OUTPUT_IS_STATIC_ZERO: bool = true;

    fn c_prefix() -> &'static str {
        "mod"
    }
    fn known_suffix(field: KnownField) -> Option<&'static str> {
        Some(field.c_suffix())
    }
}

impl ArithKind for ModArithKind {
    fn opname() -> &'static str {
        "mod_arith"
    }
}

impl SetupKind for ModArithKind {
    fn opname() -> &'static str {
        "mod_setup"
    }
}

impl IsEqKind for ModArithKind {
    fn opname() -> &'static str {
        "mod_iseq"
    }
}

/// IR node for modular arithmetic (ADD, SUB, MUL, DIV).
pub(crate) type ModArithInstr = FieldArithInstr<ModArithKind>;

/// IR node for modular IS_EQ.
pub(crate) type ModIsEqInstr = FieldIsEqInstr<ModArithKind>;

/// IR node for modular SETUP (SETUP_ADDSUB and SETUP_MULDIV).
pub(crate) type ModSetupInstr = FieldSetupInstr<ModArithKind>;

#[derive(Debug, Clone)]
struct ModSetupIsEqInstr {
    rd_reg: Variable,
    rs1_reg: Variable,
    rs2_reg: Variable,
    num_limbs: u32,
    modulus: Vec<u8>,
}

impl ExtInstr for ModSetupIsEqInstr {
    fn opname(&self) -> &str {
        "mod_setup_iseq"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rs1 = ctx.read_var(self.rs1_reg);
        let rs2 = ctx.read_var(self.rs2_reg);
        let timed_blocks = self.num_limbs * BINARY_INPUTS / MEMORY_BLOCK_BYTES_U32;
        ctx.advance_timestamp(timed_blocks);
        emit_word_alignment_guard(ctx, &[&rs1, &rs2]);
        let mod_literal = format_c_byte_array(&self.modulus);
        let num_limbs = format!("{}u", self.num_limbs);
        ctx.write_line("{");
        ctx.write_line(&format!("static constexpr uint8_t mod_[] = {mod_literal};"));
        let result = ctx.emit_call_expr(
            "uint8_t",
            "rvr_ext_mod_setup_iseq",
            &["state", &rs1, &rs2, &num_limbs, "mod_"],
        );
        ctx.write_line(&format!("if (unlikely({result} > 1u)) {{"));
        ctx.emit_trap();
        ctx.write_line("}");
        ctx.write_var(self.rd_reg, &result);
        ctx.write_line("}");
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::WriteUnknown { dst: self.rd_reg }
    }

    fn supports_preflight(&self) -> bool {
        true
    }
}

// ── Phantom instructions (HintNonQr, HintSqrt) ──────────────────────────────

/// IR node for HintNonQr phantom instruction.
#[derive(Debug, Clone)]
pub struct HintNonQrInstr {
    pub non_qr_bytes: Vec<u8>,
}

impl ExtInstr for HintNonQrInstr {
    fn opname(&self) -> &str {
        "hint_nonqr"
    }

    fn accesses_memory(&self) -> bool {
        false
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let literal = format_c_byte_array(&self.non_qr_bytes);
        ctx.write_line("{");
        ctx.write_line(&format!("static constexpr uint8_t nqr[] = {literal};"));
        let len = format!("{}u", self.non_qr_bytes.len());
        ctx.emit_call_without_page_flush("ext_hint_stream_set", &["nqr", &len]);
        ctx.write_line("}");
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

/// IR node for HintSqrt phantom instruction.
#[derive(Debug, Clone)]
pub struct HintSqrtInstr {
    pub rs1_reg: Variable,
    pub num_limbs: u32,
    pub modulus: Vec<u8>,
    pub non_qr_bytes: Vec<u8>,
}

impl ExtInstr for HintSqrtInstr {
    fn opname(&self) -> &str {
        "hint_sqrt"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rs1 = ctx.peek_var(self.rs1_reg);
        let mod_literal = format_c_byte_array(&self.modulus);
        let nqr_literal = format_c_byte_array(&self.non_qr_bytes);
        ctx.write_line("{");
        ctx.write_line(&format!("static constexpr uint8_t mod_[] = {mod_literal};"));
        ctx.write_line(&format!("static constexpr uint8_t nqr[] = {nqr_literal};"));
        let num_limbs = format!("{}u", self.num_limbs);
        ctx.emit_call(
            "rvr_ext_algebra_hint_sqrt",
            &["state", &rs1, &num_limbs, "mod_", "nqr"],
        );
        ctx.write_line("}");
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

// ── Modular extension ────────────────────────────────────────────────────────

/// Modular arithmetic and phantom hints.
pub struct ModularRvrExtension {
    moduli: Vec<ModulusInfo>,
}

impl ModularRvrExtension {
    pub fn new(moduli: Vec<BigUint>) -> Self {
        Self {
            moduli: make_moduli(moduli),
        }
    }
}

impl RvrExtension for ModularRvrExtension {
    fn try_lift(&self, insn: &Instruction, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if let Some(lifted) = self.try_lift_modular(insn, pc, opcode) {
            return Some(lifted);
        }

        if opcode == SystemOpcode::PHANTOM.global_opcode_usize() {
            if let Some(lifted) = self.try_lift_phantom(insn, pc) {
                return Some(lifted);
            }
        }

        None
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "rvr_ext_bls12_381.h",
                include_str!("../c/rvr_ext_bls12_381.h"),
            ),
            ("rvr_ext_mod.h", include_str!("../c/rvr_ext_mod.h")),
        ]
    }

    fn c_sources(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "rvr_ext_secp256k1.c",
                include_str!("../ffi/modular/c/rvr_ext_secp256k1.c"),
            ),
            (
                "rvr_ext_bls12_381.c",
                include_str!("../ffi/modular/c/rvr_ext_bls12_381.c"),
            ),
        ]
    }

    fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        vec![
            (
                "librvr_openvm_ext_algebra_modular_ffi.a",
                include_bytes!(env!("RVR_ALGEBRA_MODULAR_FFI_STATICLIB")),
            ),
            (
                "libblst.a",
                include_bytes!(env!("RVR_ALGEBRA_BLST_STATICLIB")),
            ),
        ]
    }

    fn uses_memory_wrappers(&self) -> bool {
        true
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        MODULAR_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION
    }

    fn vendored_c_sources(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "precomputed_ecmult.c",
                include_str!("../ffi/modular/secp256k1/src/precomputed_ecmult.c"),
            ),
            (
                "precomputed_ecmult_gen.c",
                include_str!("../ffi/modular/secp256k1/src/precomputed_ecmult_gen.c"),
            ),
        ]
    }

    fn extra_c_include_files(&self) -> Vec<(&'static str, &'static str)> {
        let mut files = SECP256K1_C_FILES.to_vec();
        files.extend([
            (
                "blst/blst.h",
                include_str!("../ffi/modular/blst/bindings/blst.h"),
            ),
            (
                "blst/blst_aux.h",
                include_str!("../ffi/modular/blst/bindings/blst_aux.h"),
            ),
        ]);
        files
    }

    fn extra_cflags(&self) -> Vec<String> {
        vec![
            "-isystem".to_string(),
            "secp256k1/src".to_string(),
            "-isystem".to_string(),
            "secp256k1".to_string(),
            "-isystem".to_string(),
            "blst".to_string(),
            // ENABLE_MODULE_RECOVERY keeps the ECC modules compiled in so the
            // k256 EC ops in rvr_ext_secp256k1.c can call into libsecp256k1.
            // (-DSECP256K1_BUILD is not set here — secp256k1.c defines it
            // internally.)
            "-DENABLE_MODULE_RECOVERY".to_string(),
        ]
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use rvr_openvm_ir::{MemWidth, PageAddressSpace};

    use super::*;

    #[derive(Default)]
    struct TestEmitCtx {
        operations: Vec<String>,
        preflight: bool,
        next_tmp: usize,
    }

    impl TestEmitCtx {
        fn preflight() -> Self {
            Self {
                preflight: true,
                ..Self::default()
            }
        }
    }

    impl ExtEmitCtx for TestEmitCtx {
        fn is_preflight(&self) -> bool {
            self.preflight
        }

        fn read_var(&mut self, var: Variable) -> String {
            self.operations.push(format!("read(r{});", var.index()));
            format!("r{}", var.index())
        }

        fn peek_var(&mut self, var: Variable) -> String {
            format!("r{}", var.index())
        }

        fn advance_timestamp(&mut self, slots: u32) {
            if self.preflight {
                self.operations.push(format!("timestamp_slots({slots});"));
            }
        }

        fn write_var(&mut self, var: Variable, val: &str) {
            self.operations
                .push(format!("write(r{}, {val});", var.index()));
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

        fn reserve_preflight_timestamp_slots(&mut self, _slots: &str) {
            unreachable!()
        }

        fn append_replay_value(&mut self, value: &str) {
            if self.preflight {
                self.operations.push(format!("replay_value({value});"));
            }
        }

        fn emit_call(&mut self, name: &str, args: &[&str]) {
            self.operations
                .push(format!("{name}({});", args.join(", ")));
        }

        fn emit_call_without_page_flush(&mut self, name: &str, args: &[&str]) {
            self.emit_call(name, args)
        }

        fn emit_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String {
            let value = format!("tmp{}", self.next_tmp);
            self.next_tmp += 1;
            self.operations
                .push(format!("{ret_ty} {value} = {name}({});", args.join(", ")));
            value
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
    fn hint_non_qr_advances_one_timestamp() {
        let instr = HintNonQrInstr {
            non_qr_bytes: vec![1; 32],
        };
        assert!(instr.supports_preflight());

        let mut ctx = TestEmitCtx::preflight();
        instr.emit_c(&mut ctx);

        assert!(!ctx.operations.iter().any(|op| op.starts_with("read(")));
        assert!(ctx
            .operations
            .iter()
            .any(|op| op.starts_with("ext_hint_stream_set(nqr,")));
        assert_eq!(
            ctx.operations
                .iter()
                .filter(|op| op.as_str() == "timestamp_slots(1);")
                .count(),
            1
        );
        assert_eq!(
            ctx.operations.last().map(String::as_str),
            Some("timestamp_slots(1);")
        );
    }

    #[test]
    fn hint_sqrt_peeks_pointer_and_advances_one_timestamp() {
        let instr = HintSqrtInstr {
            rs1_reg: Variable::new(10),
            num_limbs: 32,
            modulus: vec![0; 32],
            non_qr_bytes: vec![0; 32],
        };
        assert!(instr.supports_preflight());

        let mut ctx = TestEmitCtx::preflight();
        instr.emit_c(&mut ctx);

        assert!(!ctx.operations.iter().any(|op| op.starts_with("read(")));
        assert!(ctx
            .operations
            .iter()
            .any(|op| op.starts_with("rvr_ext_algebra_hint_sqrt(state, r10,")));
        assert_eq!(
            ctx.operations
                .iter()
                .filter(|op| op.as_str() == "timestamp_slots(1);")
                .count(),
            1
        );
        assert_eq!(
            ctx.operations.last().map(String::as_str),
            Some("timestamp_slots(1);")
        );
    }

    #[test]
    fn modular_arithmetic_preflight_emits_air_order_and_minimal_postimage() {
        for num_limbs in [32, 48] {
            for op in [ModOp::Add, ModOp::Sub, ModOp::Mul, ModOp::Div] {
                let instr = ModArithInstr::new(
                    op,
                    Variable::new(1),
                    Variable::new(2),
                    Variable::new(3),
                    num_limbs,
                    vec![7; num_limbs as usize],
                );
                assert!(instr.supports_preflight());

                let mut preflight = TestEmitCtx::preflight();
                instr.emit_c(&mut preflight);
                assert_eq!(
                    &preflight.operations[..4],
                    [
                        "read(r2);",
                        "read(r3);",
                        "read(r1);",
                        &format!(
                            "timestamp_slots({});",
                            num_limbs * BINARY_INPUTS_AND_OUTPUT / MEMORY_BLOCK_BYTES_U32
                        ),
                    ]
                );
                assert!(preflight
                    .operations
                    .iter()
                    .any(|operation| operation.contains("& 7ull")));
                let replay_values: Vec<_> = preflight
                    .operations
                    .iter()
                    .filter(|operation| operation.starts_with("replay_value("))
                    .cloned()
                    .collect();
                assert_eq!(replay_values.len(), num_limbs as usize / MEMORY_BLOCK_BYTES);
                for (word, replay_value) in replay_values.iter().enumerate() {
                    assert_eq!(
                        replay_value,
                        &format!(
                            "replay_value(peek_mem_u64(state, r1 + {}ull));",
                            word * MEMORY_BLOCK_BYTES
                        )
                    );
                }

                let mut legacy = TestEmitCtx::default();
                instr.emit_c(&mut legacy);
                assert_eq!(
                    &legacy.operations[..3],
                    ["read(r2);", "read(r3);", "read(r1);"]
                );
                assert!(!legacy
                    .operations
                    .iter()
                    .any(|operation| operation.starts_with("timestamp_slots(")
                        || operation.starts_with("replay_value(")));
            }
        }
    }

    #[test]
    fn modular_setup_preflight_has_no_replay_value() {
        for num_limbs in [32, 48] {
            // SETUP_ADDSUB and SETUP_MULDIV both lift to this node.
            let instr = ModSetupInstr::new(
                Variable::new(1),
                Variable::new(2),
                Variable::new(3),
                num_limbs,
                vec![11; num_limbs as usize],
            );
            assert!(instr.supports_preflight());

            let mut preflight = TestEmitCtx::preflight();
            instr.emit_c(&mut preflight);
            assert_eq!(
                &preflight.operations[..4],
                [
                    "read(r2);",
                    "read(r3);",
                    "read(r1);",
                    &format!(
                        "timestamp_slots({});",
                        num_limbs * BINARY_INPUTS_AND_OUTPUT / MEMORY_BLOCK_BYTES_U32
                    ),
                ]
            );
            assert!(preflight
                .operations
                .iter()
                .any(|operation| operation.contains("& 7ull")));
            assert!(!preflight
                .operations
                .iter()
                .any(|operation| operation.starts_with("replay_value(")));
        }
    }

    #[test]
    fn modular_iseq_preflight_emits_only_result_replay_value() {
        for num_limbs in [32, 48] {
            let instr = ModIsEqInstr::new(
                Variable::new(1),
                Variable::new(2),
                Variable::new(3),
                num_limbs,
                vec![13; num_limbs as usize],
            );
            assert!(instr.supports_preflight());

            let mut preflight = TestEmitCtx::preflight();
            instr.emit_c(&mut preflight);
            assert_eq!(
                &preflight.operations[..3],
                [
                    "read(r2);",
                    "read(r3);",
                    &format!(
                        "timestamp_slots({});",
                        num_limbs * BINARY_INPUTS / MEMORY_BLOCK_BYTES_U32
                    ),
                ]
            );
            assert!(preflight
                .operations
                .iter()
                .any(|operation| operation.contains("& 7ull")));
            assert_eq!(
                preflight
                    .operations
                    .iter()
                    .filter(|operation| operation.starts_with("replay_value("))
                    .count(),
                1
            );
            assert!(preflight
                .operations
                .iter()
                .any(|operation| operation == "replay_value(tmp0);"));
            assert!(preflight
                .operations
                .iter()
                .any(|operation| operation == "write(r1, tmp0);"));
        }
    }

    #[test]
    fn modular_setup_iseq_preflight_result_is_derivable() {
        for num_limbs in [32, 48] {
            let instr = ModSetupIsEqInstr {
                rd_reg: Variable::new(1),
                rs1_reg: Variable::new(2),
                rs2_reg: Variable::new(3),
                num_limbs,
                modulus: vec![17; num_limbs as usize],
            };
            assert!(instr.supports_preflight());

            let mut preflight = TestEmitCtx::preflight();
            instr.emit_c(&mut preflight);
            assert_eq!(
                &preflight.operations[..3],
                [
                    "read(r2);",
                    "read(r3);",
                    &format!(
                        "timestamp_slots({});",
                        num_limbs * BINARY_INPUTS / MEMORY_BLOCK_BYTES_U32
                    ),
                ]
            );
            assert!(preflight
                .operations
                .iter()
                .any(|operation| operation.contains("& 7ull")));
            assert!(!preflight
                .operations
                .iter()
                .any(|operation| operation.starts_with("replay_value(")));
        }
    }

    #[test]
    fn modular_lifter_requires_register_and_memory_address_spaces() {
        let extension = ModularRvrExtension::new(vec![BigUint::from(17u8)]);

        for opcode in [
            ModularArithmeticOpcode::ADD,
            ModularArithmeticOpcode::SUB,
            ModularArithmeticOpcode::SETUP_ADDSUB,
            ModularArithmeticOpcode::MUL,
            ModularArithmeticOpcode::DIV,
            ModularArithmeticOpcode::SETUP_MULDIV,
            ModularArithmeticOpcode::IS_EQ,
            ModularArithmeticOpcode::SETUP_ISEQ,
        ] {
            let instruction = |a, d, e| {
                Instruction::from_usize(opcode.global_opcode(), [a, 16, 24, d as usize, e as usize])
            };

            assert!(extension
                .try_lift(&instruction(8, REGISTER_AS, MEMORY_AS), 0)
                .is_some());
            assert!(extension
                .try_lift(&instruction(8, MEMORY_AS, MEMORY_AS), 0)
                .is_none());
            assert!(extension
                .try_lift(&instruction(8, REGISTER_AS, REGISTER_AS), 0)
                .is_none());

            if matches!(
                opcode,
                ModularArithmeticOpcode::IS_EQ | ModularArithmeticOpcode::SETUP_ISEQ
            ) {
                assert!(extension
                    .try_lift(&instruction(0, REGISTER_AS, MEMORY_AS), 0)
                    .is_none());
            }
        }
    }
}

impl ModularRvrExtension {
    fn try_lift_modular(&self, insn: &Instruction, pc: u64, opcode: usize) -> Option<LiftedInstr> {
        let base_offset = ModularArithmeticOpcode::CLASS_OFFSET;
        let count = ModularArithmeticOpcode::COUNT;

        if opcode < base_offset {
            return None;
        }
        let relative = opcode - base_offset;
        let mod_idx = relative / count;
        let local = relative % count;

        if mod_idx >= self.moduli.len() {
            return None;
        }
        if insn.d.as_u32() != REGISTER_AS || insn.e.as_u32() != MEMORY_AS {
            return None;
        }
        if insn.a.is_zero()
            && matches!(
                ModularArithmeticOpcode::from_repr(local),
                Some(ModularArithmeticOpcode::IS_EQ | ModularArithmeticOpcode::SETUP_ISEQ)
            )
        {
            return None;
        }

        let info = &self.moduli[mod_idx];
        let rd_reg = decode_reg(insn.a.as_u32());
        let rs1_reg = decode_reg(insn.b.as_u32());
        let rs2_reg = decode_reg(insn.c.as_u32());

        let instr: Box<dyn ExtInstr> = match local {
            x if x == ModularArithmeticOpcode::ADD as usize => Box::new(ModArithInstr::new(
                ModOp::Add,
                rd_reg,
                rs1_reg,
                rs2_reg,
                info.num_limbs,
                info.modulus_bytes.clone(),
            )),
            x if x == ModularArithmeticOpcode::SUB as usize => Box::new(ModArithInstr::new(
                ModOp::Sub,
                rd_reg,
                rs1_reg,
                rs2_reg,
                info.num_limbs,
                info.modulus_bytes.clone(),
            )),
            x if x == ModularArithmeticOpcode::SETUP_ADDSUB as usize => {
                Box::new(ModSetupInstr::new(
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    info.num_limbs,
                    info.modulus_bytes.clone(),
                ))
            }
            x if x == ModularArithmeticOpcode::MUL as usize => Box::new(ModArithInstr::new(
                ModOp::Mul,
                rd_reg,
                rs1_reg,
                rs2_reg,
                info.num_limbs,
                info.modulus_bytes.clone(),
            )),
            x if x == ModularArithmeticOpcode::DIV as usize => Box::new(ModArithInstr::new(
                ModOp::Div,
                rd_reg,
                rs1_reg,
                rs2_reg,
                info.num_limbs,
                info.modulus_bytes.clone(),
            )),
            x if x == ModularArithmeticOpcode::SETUP_MULDIV as usize => {
                Box::new(ModSetupInstr::new(
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    info.num_limbs,
                    info.modulus_bytes.clone(),
                ))
            }
            x if x == ModularArithmeticOpcode::IS_EQ as usize => Box::new(ModIsEqInstr::new(
                rd_reg,
                rs1_reg,
                rs2_reg,
                info.num_limbs,
                info.modulus_bytes.clone(),
            )),
            x if x == ModularArithmeticOpcode::SETUP_ISEQ as usize => Box::new(ModSetupIsEqInstr {
                rd_reg,
                rs1_reg,
                rs2_reg,
                num_limbs: info.num_limbs,
                modulus: info.modulus_bytes.clone(),
            }),
            _ => return None,
        };

        Some(LiftedInstr::Body(InstrAt {
            pc,
            instr,
            source_loc: None,
        }))
    }

    fn try_lift_phantom(&self, insn: &Instruction, pc: u64) -> Option<LiftedInstr> {
        let discriminant = u16::try_from(insn.c.as_u32()).ok()?;
        let mod_idx = usize::try_from(insn.d.as_u32()).ok()?;

        match ModularPhantom::from_repr(discriminant) {
            Some(ModularPhantom::HintNonQr) => {
                let info = self.moduli.get(mod_idx)?;
                Some(LiftedInstr::Body(InstrAt {
                    pc,
                    instr: Box::new(HintNonQrInstr {
                        non_qr_bytes: info.non_qr_bytes.clone(),
                    }),
                    source_loc: None,
                }))
            }
            Some(ModularPhantom::HintSqrt) => {
                let info = self.moduli.get(mod_idx)?;
                let rs1_reg = decode_reg(insn.a.as_u32());
                Some(LiftedInstr::Body(InstrAt {
                    pc,
                    instr: Box::new(HintSqrtInstr {
                        rs1_reg,
                        num_limbs: info.num_limbs,
                        modulus: info.modulus_bytes.clone(),
                        non_qr_bytes: info.non_qr_bytes.clone(),
                    }),
                    source_loc: None,
                }))
            }
            None => None,
        }
    }
}
