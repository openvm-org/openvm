//! Fp2 (complex extension field) IR nodes and the [`Fp2RvrExtension`] lifter.

use num_bigint::BigUint;
use openvm_algebra_transpiler::Fp2Opcode;
use openvm_instructions::LocalOpcode;
#[cfg(test)]
use openvm_instructions::MEMORY_BLOCK_BYTES;
use rvr_openvm_ir::{ExtInstr, InstrAt, LiftedInstr};
use rvr_openvm_lift::{max_main_memory_pages_for_contiguous_range, RvrExtension, RvrInstruction};
use strum::EnumCount;

use crate::{
    decode_reg, pad_modulus, ArithKind, FieldArithInstr, FieldKind, FieldSetupInstr, KnownField,
    ModOp, SetupKind,
};
#[cfg(test)]
use crate::{BINARY_INPUTS_AND_OUTPUT, MEMORY_BLOCK_BYTES_U32};

// An Fp2 operation can read two independent 96-byte values and write one.
const FP2_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION: usize =
    3 * max_main_memory_pages_for_contiguous_range(96);

/// Per-modulus info for the Fp2 extension. Fp2 lifting never consults a
/// non-QR, so we only carry the padded modulus and limb count.
struct ModulusInfo {
    modulus_bytes: Vec<u8>,
    num_limbs: u32,
}

fn make_moduli(moduli: Vec<BigUint>) -> Vec<ModulusInfo> {
    moduli.into_iter().map(make_modulus_info).collect()
}

fn make_modulus_info(modulus: BigUint) -> ModulusInfo {
    let (modulus_bytes, num_limbs) = pad_modulus(&modulus);
    ModulusInfo {
        modulus_bytes,
        num_limbs,
    }
}

// ── Fp2 arithmetic IR ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct Fp2Kind;

impl FieldKind for Fp2Kind {
    const STORAGE_FACTOR: u32 = 2;
    const SETUP_OUTPUT_IS_STATIC_ZERO: bool = false;

    fn c_prefix() -> &'static str {
        "fp2"
    }
    fn known_suffix(field: KnownField) -> Option<&'static str> {
        field.fp2_c_suffix()
    }
}

impl ArithKind for Fp2Kind {
    fn opname() -> &'static str {
        "fp2_arith"
    }
}

impl SetupKind for Fp2Kind {
    fn opname() -> &'static str {
        "fp2_setup"
    }
}

/// IR node for Fp2 arithmetic (ADD, SUB, MUL, DIV).
pub(crate) type Fp2ArithInstr = FieldArithInstr<Fp2Kind>;

/// IR node for Fp2 SETUP (SETUP_ADDSUB, SETUP_MULDIV).
pub(crate) type Fp2SetupInstr = FieldSetupInstr<Fp2Kind>;

// ── Fp2 extension ────────────────────────────────────────────────────────────

/// Fp2 arithmetic for the configured base fields.
pub struct Fp2RvrExtension {
    fp2_moduli: Vec<ModulusInfo>,
}

impl Fp2RvrExtension {
    pub fn new(fp2_moduli: Vec<BigUint>) -> Self {
        Self {
            fp2_moduli: make_moduli(fp2_moduli),
        }
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use rvr_openvm_ir::{ExtEmitCtx, MemWidth, PageAddressSpace, Variable};

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
            self.emit_call(name, args);
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
    fn fp2_arithmetic_preflight_matches_vec_heap_schedule() {
        for num_limbs in [32, 48] {
            for op in [ModOp::Add, ModOp::Sub, ModOp::Mul, ModOp::Div] {
                let instr = Fp2ArithInstr::new(
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
                            num_limbs * Fp2Kind::STORAGE_FACTOR * BINARY_INPUTS_AND_OUTPUT
                                / MEMORY_BLOCK_BYTES_U32
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
                    .collect();
                assert_eq!(
                    replay_values.len(),
                    num_limbs as usize * Fp2Kind::STORAGE_FACTOR as usize
                        / MEMORY_BLOCK_BYTES as usize
                );
                for (word, replay_value) in replay_values.iter().enumerate() {
                    assert_eq!(
                        replay_value.as_str(),
                        format!(
                            "replay_value(peek_mem_u64(state, r1 + {}ull));",
                            word * MEMORY_BLOCK_BYTES as usize
                        )
                    );
                }

                let mut legacy = TestEmitCtx::default();
                instr.emit_c(&mut legacy);
                assert_eq!(
                    &legacy.operations[..3],
                    ["read(r2);", "read(r3);", "read(r1);"]
                );
                assert!(!legacy.operations.iter().any(|operation| {
                    operation.starts_with("timestamp_slots(")
                        || operation.starts_with("replay_value(")
                }));
            }
        }
    }

    #[test]
    fn fp2_setup_preflight_appends_full_destination_postimage() {
        for num_limbs in [32, 48] {
            let instr = Fp2SetupInstr::new(
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
                        num_limbs * Fp2Kind::STORAGE_FACTOR * BINARY_INPUTS_AND_OUTPUT
                            / MEMORY_BLOCK_BYTES_U32
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
                .collect();
            assert_eq!(
                replay_values.len(),
                num_limbs as usize * Fp2Kind::STORAGE_FACTOR as usize / MEMORY_BLOCK_BYTES as usize
            );
            for (word, replay_value) in replay_values.iter().enumerate() {
                assert_eq!(
                    replay_value.as_str(),
                    format!(
                        "replay_value(peek_mem_u64(state, r1 + {}ull));",
                        word * MEMORY_BLOCK_BYTES as usize
                    )
                );
            }
            assert!(preflight
                .operations
                .iter()
                .any(|operation| operation.starts_with("bool tmp0 = rvr_ext_fp2_setup(")));
            assert!(preflight
                .operations
                .iter()
                .any(|operation| operation == "if (unlikely(!tmp0)) {"));
            assert!(preflight
                .operations
                .iter()
                .any(|operation| operation == "trap;"));
            assert!(!preflight
                .operations
                .iter()
                .any(|operation| operation.starts_with("write(r")));

            let mut legacy = TestEmitCtx::default();
            instr.emit_c(&mut legacy);
            assert_eq!(
                &legacy.operations[..3],
                ["read(r2);", "read(r3);", "read(r1);"]
            );
            assert!(!legacy.operations.iter().any(|operation| {
                operation.starts_with("timestamp_slots(") || operation.starts_with("replay_value(")
            }));
        }
    }
}

impl RvrExtension for Fp2RvrExtension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();
        self.try_lift_fp2(insn, pc, opcode)
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![("rvr_ext_fp2.h", include_str!("../c/rvr_ext_fp2.h"))]
    }

    fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        vec![(
            "librvr_openvm_ext_algebra_fp2_ffi.a",
            include_bytes!(env!("RVR_ALGEBRA_FP2_FFI_STATICLIB")),
        )]
    }

    fn uses_memory_wrappers(&self) -> bool {
        true
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        FP2_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION
    }
}

impl Fp2RvrExtension {
    fn try_lift_fp2(&self, insn: &RvrInstruction, pc: u64, opcode: usize) -> Option<LiftedInstr> {
        let base_offset = Fp2Opcode::CLASS_OFFSET;
        let count = Fp2Opcode::COUNT;

        if opcode < base_offset {
            return None;
        }
        let relative = opcode - base_offset;
        let fp2_idx = relative / count;
        let local = relative % count;

        if fp2_idx >= self.fp2_moduli.len() {
            return None;
        }

        let info = &self.fp2_moduli[fp2_idx];
        let rd_reg = decode_reg(insn.a);
        let rs1_reg = decode_reg(insn.b);
        let rs2_reg = decode_reg(insn.c);

        let instr: Box<dyn ExtInstr> = match local {
            x if x == Fp2Opcode::ADD as usize => Box::new(Fp2ArithInstr::new(
                ModOp::Add,
                rd_reg,
                rs1_reg,
                rs2_reg,
                info.num_limbs,
                info.modulus_bytes.clone(),
            )),
            x if x == Fp2Opcode::SUB as usize => Box::new(Fp2ArithInstr::new(
                ModOp::Sub,
                rd_reg,
                rs1_reg,
                rs2_reg,
                info.num_limbs,
                info.modulus_bytes.clone(),
            )),
            x if x == Fp2Opcode::SETUP_ADDSUB as usize => Box::new(Fp2SetupInstr::new(
                rd_reg,
                rs1_reg,
                rs2_reg,
                info.num_limbs,
                info.modulus_bytes.clone(),
            )),
            x if x == Fp2Opcode::MUL as usize => Box::new(Fp2ArithInstr::new(
                ModOp::Mul,
                rd_reg,
                rs1_reg,
                rs2_reg,
                info.num_limbs,
                info.modulus_bytes.clone(),
            )),
            x if x == Fp2Opcode::DIV as usize => Box::new(Fp2ArithInstr::new(
                ModOp::Div,
                rd_reg,
                rs1_reg,
                rs2_reg,
                info.num_limbs,
                info.modulus_bytes.clone(),
            )),
            x if x == Fp2Opcode::SETUP_MULDIV as usize => Box::new(Fp2SetupInstr::new(
                rd_reg,
                rs1_reg,
                rs2_reg,
                info.num_limbs,
                info.modulus_bytes.clone(),
            )),
            _ => return None,
        };

        Some(LiftedInstr::Body(InstrAt {
            pc,
            instr,
            source_loc: None,
        }))
    }
}
