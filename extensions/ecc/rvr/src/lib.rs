//! ECC extension for rvr-openvm.
//!
//! Provides IR nodes and extension trait implementation for the short Weierstrass
//! elliptic curve opcodes (EC_ADD_NE, EC_DOUBLE + setups).
//!
//! Modular arithmetic opcodes are handled separately by the algebra extension.

use openvm_ecc_transpiler::Rv64WeierstrassOpcode::{
    self, EC_ADD_NE, EC_DOUBLE, SETUP_EC_ADD_NE, SETUP_EC_DOUBLE,
};
use openvm_instructions::{
    riscv::{RV64_NUM_REGISTERS, RV64_REGISTER_BYTES},
    LocalOpcode,
};
use rvr_openvm_ir::{CfgEffect, ExtEmitCtx, ExtInstr, InstrAt, LiftedInstr, Variable};
use rvr_openvm_lift::{
    decode_variable, max_main_memory_pages_for_contiguous_range, RvrExtension, RvrInstruction,
};
use strum::EnumCount;

// An ECC addition can read two independent 96-byte points and write one.
const ECC_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION: usize =
    3 * max_main_memory_pages_for_contiguous_range(96);

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

#[derive(Debug, Clone, Copy)]
enum KnownCurve {
    K256,
    P256,
    Bn254,
    Bls12381,
}

impl KnownCurve {
    fn from_id(curve_id: u32) -> Option<Self> {
        match curve_id {
            0 => Some(Self::K256),
            1 => Some(Self::P256),
            2 => Some(Self::Bn254),
            3 => Some(Self::Bls12381),
            _ => None,
        }
    }

    fn c_suffix(self) -> &'static str {
        match self {
            Self::K256 => "k256",
            Self::P256 => "p256",
            Self::Bn254 => "bn254",
            Self::Bls12381 => "bls12_381",
        }
    }

    fn point_dwords(self) -> u32 {
        match self {
            Self::K256 | Self::P256 | Self::Bn254 => 8,
            Self::Bls12381 => 12,
        }
    }

    fn from_struct_name(struct_name: &str) -> Option<Self> {
        match struct_name {
            "Secp256k1Point" => Some(Self::K256),
            "P256Point" => Some(Self::P256),
            "Bn254G1Affine" => Some(Self::Bn254),
            "Bls12_381G1Affine" => Some(Self::Bls12381),
            _ => None,
        }
    }
}

// ── IR nodes ──────────────────────────────────────────────────────────────────

/// IR node for EC point addition (non-equal x-coordinates).
#[derive(Debug, Clone)]
pub struct EcAddNeInstr {
    pub rd_reg: Variable,
    pub rs1_reg: Variable,
    pub rs2_reg: Variable,
    curve: KnownCurve,
    pub is_setup: bool,
}

impl ExtInstr for EcAddNeInstr {
    fn opname(&self) -> &str {
        "ec_add_ne"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let checkpoint = ctx.is_checkpoint_preflight();
        let count_residuals = ctx.counts_checkpoint_residuals();
        let (rd, rs1, rs2) = if checkpoint {
            // Match the VecHeap adapter: source registers precede the destination register.
            let rs1 = ctx.read_var(self.rs1_reg);
            let rs2 = ctx.read_var(self.rs2_reg);
            let rd = ctx.read_var(self.rd_reg);
            (rd, rs1, rs2)
        } else {
            // Preserve the established pure, metered, and ValueTrace register order.
            let rd = ctx.read_var(self.rd_reg);
            let rs1 = ctx.read_var(self.rs1_reg);
            let rs2 = ctx.read_var(self.rs2_reg);
            (rd, rs1, rs2)
        };
        emit_pointer_alignment_guard(ctx, &[&rd, &rs1, &rs2]);
        let point_dwords = self.curve.point_dwords();
        if checkpoint {
            // Two point reads followed by one point write happen inside the opaque call.
            ctx.advance_checkpoint_timestamp(3 * point_dwords);
        }
        let setup_prefix = if self.is_setup { "setup_" } else { "" };
        let suffix = self.curve.c_suffix();
        let name = format!("rvr_ext_{setup_prefix}ec_add_ne_{suffix}");
        if self.is_setup {
            ctx.emit_checked_call(&name, &["state", &rd, &rs1, &rs2]);
        } else {
            ctx.emit_call(&name, &["state", &rd, &rs1, &rs2]);
        }
        if count_residuals {
            // Add setup constrains only its modulus input; y1, x2, and y2 remain execution data.
            // Its postimage is therefore no less authoritative than a regular add postimage.
            for word in 0..point_dwords {
                ctx.append_replay_value(&format!("peek_mem_u64(state, {rd} + {}ull)", word * 8));
            }
        }
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

/// IR node for EC point doubling.
#[derive(Debug, Clone)]
pub struct EcDoubleInstr {
    pub rd_reg: Variable,
    pub rs1_reg: Variable,
    curve: KnownCurve,
    pub is_setup: bool,
}

impl ExtInstr for EcDoubleInstr {
    fn opname(&self) -> &str {
        "ec_double"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let checkpoint = ctx.is_checkpoint_preflight();
        let count_residuals = ctx.counts_checkpoint_residuals();
        let (rd, rs1) = if checkpoint {
            // Match the VecHeap adapter: the source register precedes the destination register.
            let rs1 = ctx.read_var(self.rs1_reg);
            let rd = ctx.read_var(self.rd_reg);
            (rd, rs1)
        } else {
            // Preserve the established pure, metered, and ValueTrace register order.
            let rd = ctx.read_var(self.rd_reg);
            let rs1 = ctx.read_var(self.rs1_reg);
            (rd, rs1)
        };
        emit_pointer_alignment_guard(ctx, &[&rd, &rs1]);
        let point_dwords = self.curve.point_dwords();
        if checkpoint {
            // One point read followed by one point write happens inside the opaque call.
            ctx.advance_checkpoint_timestamp(2 * point_dwords);
        }
        let setup_prefix = if self.is_setup { "setup_" } else { "" };
        let suffix = self.curve.c_suffix();
        let name = format!("rvr_ext_{setup_prefix}ec_double_{suffix}");
        if self.is_setup {
            ctx.emit_checked_call(&name, &["state", &rd, &rs1]);
        } else {
            ctx.emit_call(&name, &["state", &rd, &rs1]);
        }
        if count_residuals && !self.is_setup {
            // Regular-operation outputs are the only residuals. Setup replay derives its writes
            // from the timed reads and the configured field-expression program rather than
            // extending the transcript with setup-only values.
            for word in 0..point_dwords {
                ctx.append_replay_value(&format!("peek_mem_u64(state, {rd} + {}ull)", word * 8));
            }
        }
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

// ── Extension ─────────────────────────────────────────────────────────────────

/// Information about a registered curve (for Weierstrass ECC opcodes).
#[derive(Debug, Clone)]
pub struct CurveInfo {
    curve: Option<KnownCurve>,
}

/// The ECC extension: handles Weierstrass EC opcodes (EC_ADD_NE, EC_DOUBLE + setups).
pub struct EccExtension {
    curves: Vec<CurveInfo>,
}

impl EccExtension {
    fn from_struct_names(struct_names: Vec<String>) -> Self {
        let curves = struct_names
            .into_iter()
            .map(|name| CurveInfo {
                curve: KnownCurve::from_struct_name(&name),
            })
            .collect();
        Self { curves }
    }

    fn from_curve_ids(curves: Vec<u32>) -> Self {
        let curves = curves
            .into_iter()
            .map(|curve_id| CurveInfo {
                curve: KnownCurve::from_id(curve_id),
            })
            .collect();
        Self { curves }
    }

    pub fn new(curves_info: Vec<u32>) -> Self {
        Self::from_curve_ids(curves_info)
    }

    pub fn new_from_struct_names(struct_names: Vec<String>) -> Self {
        Self::from_struct_names(struct_names)
    }
}

impl RvrExtension for EccExtension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        let ecc_base = Rv64WeierstrassOpcode::CLASS_OFFSET;
        let ecc_count = Rv64WeierstrassOpcode::COUNT;

        if opcode < ecc_base {
            return None;
        }
        let offset = opcode - ecc_base;
        let curve_idx = offset / ecc_count;
        let local_op = offset % ecc_count;

        let curve = self.curves.get(curve_idx)?.curve?;

        let rd_reg = decode_reg(insn.a);
        let rs1_reg = decode_reg(insn.b);

        let local_opcode = Rv64WeierstrassOpcode::from_repr(local_op)?;
        let instr: Box<dyn ExtInstr> = match local_opcode {
            EC_ADD_NE | SETUP_EC_ADD_NE => {
                let rs2_reg = decode_reg(insn.c);
                Box::new(EcAddNeInstr {
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    curve,
                    is_setup: local_opcode == SETUP_EC_ADD_NE,
                })
            }
            EC_DOUBLE | SETUP_EC_DOUBLE => Box::new(EcDoubleInstr {
                rd_reg,
                rs1_reg,
                curve,
                is_setup: local_opcode == SETUP_EC_DOUBLE,
            }),
        };

        Some(LiftedInstr::Body(InstrAt {
            pc,
            instr,
            source_loc: None,
        }))
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        // The modular extension supplies the native K-256 and BLS12-381 point
        // functions declared in this header.
        vec![("rvr_ext_ecc.h", include_str!("../c/rvr_ext_ecc.h"))]
    }

    fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        vec![(
            "librvr_openvm_ext_ecc_ffi.a",
            include_bytes!(env!("RVR_ECC_FFI_STATICLIB")),
        )]
    }

    fn uses_memory_wrappers(&self) -> bool {
        true
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        ECC_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION
    }
}

#[cfg(test)]
mod tests {
    use openvm_instructions::VmOpcode;
    use rvr_openvm_ir::{MemWidth, PageAddressSpace};

    use super::*;

    struct TestEmitCtx {
        operations: Vec<String>,
        checkpoint: bool,
        next_tmp: usize,
    }

    impl TestEmitCtx {
        fn checkpoint() -> Self {
            Self {
                operations: Vec::new(),
                checkpoint: true,
                next_tmp: 0,
            }
        }

        fn legacy() -> Self {
            Self {
                operations: Vec::new(),
                checkpoint: false,
                next_tmp: 0,
            }
        }
    }

    impl ExtEmitCtx for TestEmitCtx {
        fn is_checkpoint_preflight(&self) -> bool {
            self.checkpoint
        }

        fn read_var(&mut self, var: Variable) -> String {
            let value = format!("r{}", var.index());
            self.operations.push(format!("read({value})"));
            value
        }

        fn peek_var(&mut self, _var: Variable) -> String {
            unreachable!()
        }

        fn advance_timestamp(&mut self, _slots: u32) {
            unreachable!()
        }

        fn advance_checkpoint_timestamp(&mut self, slots: u32) {
            self.operations.push(format!("checkpoint_slots({slots})"));
        }

        fn write_var(&mut self, _var: Variable, _val: &str) {
            unreachable!()
        }

        fn write_line(&mut self, line: &str) {
            self.operations.push(line.to_string());
        }

        fn emit_trap(&mut self) {
            self.operations.push("trap".to_string());
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

        fn append_replay_value(&mut self, value: &str) {
            self.operations.push(format!("residual({value})"));
        }

        fn emit_call(&mut self, name: &str, args: &[&str]) {
            self.operations.push(format!("{name}({})", args.join(", ")));
        }

        fn emit_call_without_page_flush(&mut self, _name: &str, _args: &[&str]) {
            unreachable!()
        }

        fn emit_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String {
            let value = format!("tmp{}", self.next_tmp);
            self.next_tmp += 1;
            self.operations
                .push(format!("{ret_ty} {value} = {name}({})", args.join(", ")));
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

    fn expected_residuals(rd: &str, point_dwords: u32) -> Vec<String> {
        (0..point_dwords)
            .map(|word| format!("residual(peek_mem_u64(state, {rd} + {}ull))", word * 8))
            .collect()
    }

    #[test]
    fn ignores_opcodes_outside_configured_curves() {
        let extension = EccExtension::new(vec![0]);
        let opcode = VmOpcode::from_usize(
            Rv64WeierstrassOpcode::CLASS_OFFSET + Rv64WeierstrassOpcode::COUNT,
        );
        let insn = RvrInstruction::from_canonical(opcode, [0; 7], u32::MAX);

        assert!(extension.try_lift(&insn, 0x100).is_none());
    }

    #[test]
    fn add_checkpoint_matches_schedule_and_minimal_residuals() {
        for (curve, point_dwords) in [(KnownCurve::K256, 8), (KnownCurve::Bls12381, 12)] {
            for is_setup in [false, true] {
                let instruction = EcAddNeInstr {
                    rd_reg: Variable::new(1),
                    rs1_reg: Variable::new(2),
                    rs2_reg: Variable::new(3),
                    curve,
                    is_setup,
                };
                assert!(!instruction.supports_preflight());
                assert!(instruction.supports_checkpoint_preflight());

                let mut checkpoint = TestEmitCtx::checkpoint();
                instruction.emit_c(&mut checkpoint);
                let mut expected = vec![
                    "read(r2)".to_string(),
                    "read(r3)".to_string(),
                    "read(r1)".to_string(),
                    "if (unlikely(((r1 | r2 | r3) & 7ull) != 0ull)) {".to_string(),
                    "trap".to_string(),
                    "}".to_string(),
                    format!("checkpoint_slots({})", 3 * point_dwords),
                ];
                let name = format!(
                    "rvr_ext_{}ec_add_ne_{}",
                    if is_setup { "setup_" } else { "" },
                    curve.c_suffix()
                );
                if is_setup {
                    expected.extend([
                        format!("bool tmp0 = {name}(state, r1, r2, r3)"),
                        "if (unlikely(!tmp0)) {".to_string(),
                        "trap".to_string(),
                        "}".to_string(),
                    ]);
                } else {
                    expected.push(format!("{name}(state, r1, r2, r3)"));
                }
                expected.extend(expected_residuals("r1", point_dwords));
                assert_eq!(checkpoint.operations, expected);
            }
        }
    }

    #[test]
    fn double_checkpoint_matches_schedule_and_minimal_residuals() {
        for (curve, point_dwords) in [(KnownCurve::P256, 8), (KnownCurve::Bls12381, 12)] {
            for is_setup in [false, true] {
                let instruction = EcDoubleInstr {
                    rd_reg: Variable::new(1),
                    rs1_reg: Variable::new(2),
                    curve,
                    is_setup,
                };
                assert!(!instruction.supports_preflight());
                assert!(instruction.supports_checkpoint_preflight());

                let mut checkpoint = TestEmitCtx::checkpoint();
                instruction.emit_c(&mut checkpoint);
                let mut expected = vec![
                    "read(r2)".to_string(),
                    "read(r1)".to_string(),
                    "if (unlikely(((r1 | r2) & 7ull) != 0ull)) {".to_string(),
                    "trap".to_string(),
                    "}".to_string(),
                    format!("checkpoint_slots({})", 2 * point_dwords),
                ];
                let name = format!(
                    "rvr_ext_{}ec_double_{}",
                    if is_setup { "setup_" } else { "" },
                    curve.c_suffix()
                );
                if is_setup {
                    expected.extend([
                        format!("bool tmp0 = {name}(state, r1, r2)"),
                        "if (unlikely(!tmp0)) {".to_string(),
                        "trap".to_string(),
                        "}".to_string(),
                    ]);
                } else {
                    expected.push(format!("{name}(state, r1, r2)"));
                }
                if !is_setup {
                    expected.extend(expected_residuals("r1", point_dwords));
                }
                assert_eq!(checkpoint.operations, expected);
            }
        }
    }

    #[test]
    fn legacy_emission_preserves_destination_first_order_without_checkpoint_data() {
        let add = EcAddNeInstr {
            rd_reg: Variable::new(1),
            rs1_reg: Variable::new(2),
            rs2_reg: Variable::new(3),
            curve: KnownCurve::K256,
            is_setup: false,
        };
        let mut legacy = TestEmitCtx::legacy();
        add.emit_c(&mut legacy);
        assert_eq!(
            legacy.operations,
            [
                "read(r1)",
                "read(r2)",
                "read(r3)",
                "if (unlikely(((r1 | r2 | r3) & 7ull) != 0ull)) {",
                "trap",
                "}",
                "rvr_ext_ec_add_ne_k256(state, r1, r2, r3)",
            ]
        );

        let double = EcDoubleInstr {
            rd_reg: Variable::new(1),
            rs1_reg: Variable::new(2),
            curve: KnownCurve::P256,
            is_setup: false,
        };
        let mut legacy = TestEmitCtx::legacy();
        double.emit_c(&mut legacy);
        assert_eq!(
            legacy.operations,
            [
                "read(r1)",
                "read(r2)",
                "if (unlikely(((r1 | r2) & 7ull) != 0ull)) {",
                "trap",
                "}",
                "rvr_ext_ec_double_p256(state, r1, r2)",
            ]
        );
    }
}
