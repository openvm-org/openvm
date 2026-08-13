//! ECC extension for rvr-openvm.
//!
//! Provides IR nodes and extension trait implementation for the short Weierstrass
//! elliptic curve opcodes (EC_ADD_NE, EC_DOUBLE, EC_MUL + setups).
//!
//! Modular arithmetic opcodes are handled separately by the algebra extension.

use openvm_ecc_transpiler::WeierstrassOpcode::{
    self, EC_ADD_NE, EC_DOUBLE, EC_MUL, SETUP_EC_ADD_NE, SETUP_EC_DOUBLE, SETUP_EC_MUL,
};
use openvm_instructions::{
    riscv::{NUM_REGISTERS, REGISTER_BYTES},
    LocalOpcode, VmOpcode,
};
use rvr_openvm_ir::{
    CfgEffect, ExtEmitCtx, ExtInstr, FixedTraceRows, InstrAt, LiftedInstr, Variable,
};
use rvr_openvm_lift::{
    decode_variable, fixed_trace_rows_for_chip, max_main_memory_pages_for_contiguous_range,
    AirIndex, ExtensionError, RvrExtension, RvrExtensionCtx, RvrInstruction,
};
use strum::EnumCount;

// An ECC addition can read two independent 96-byte points and write one. `EC_MUL` reads one point
// and a 32-byte scalar and writes one point, which this also covers.
const ECC_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION: usize =
    3 * max_main_memory_pages_for_contiguous_range(96);

/// `EC_MUL`'s scalar operand, in RV64 words. The opcode takes a fixed 256-bit scalar on every
/// curve.
pub const EC_MUL_SCALAR_DWORDS: u32 = 4;

/// Trace rows the `EC_MUL` chip consumes per instruction: one ladder row per two scalar digits,
/// the last of which carries the instruction's memory accesses.
///
/// Restated here rather than imported because `openvm-ecc-circuit` — which owns the chip and is the
/// authority on this value — depends on this crate. That crate statically asserts the two agree.
pub const EC_MUL_TRACE_ROWS: u32 = 128;

fn decode_reg(value: u32) -> Variable {
    decode_variable(value, REGISTER_BYTES as u32, NUM_REGISTERS as u32)
}

fn emit_pointer_alignment_guard(ctx: &mut dyn ExtEmitCtx, pointers: &[&str]) {
    let pointers = pointers.join(" | ");
    ctx.write_line(&format!(
        "if (unlikely((({pointers}) & {}ull) != 0ull)) {{",
        REGISTER_BYTES - 1
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

/// Resolve the AIR index of the `EC_MUL` chip for the curve registered at `curve_idx`.
///
/// `EC_MUL` and `SETUP_EC_MUL` share one executor, so a single index serves both. Returns
/// `Ok(None)` in pure mode, where no metering context exists.
fn ec_mul_air_idx(
    ctx: Option<&RvrExtensionCtx>,
    curve_idx: usize,
) -> Result<Option<AirIndex>, ExtensionError> {
    let Some(ctx) = ctx else {
        return Ok(None);
    };
    let opcode = VmOpcode::from_usize(
        WeierstrassOpcode::CLASS_OFFSET + curve_idx * WeierstrassOpcode::COUNT + EC_MUL as usize,
    );
    let executor_idx = ctx
        .resolve_opcode_executor_idx(opcode)
        .ok_or(ExtensionError::UnknownOpcode(opcode))?;
    let raw = ctx.executor_idx_to_air_idx.get(executor_idx).ok_or(
        ExtensionError::ExecutorIndexOutOfBounds {
            opcode,
            executor_idx,
        },
    )?;
    let air_idx = u32::try_from(*raw).map_err(|_| ExtensionError::AirIndexOutOfBounds {
        opcode,
        air_idx: *raw,
    })?;
    Ok(Some(AirIndex::new(air_idx)))
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
        let is_preflight = ctx.is_preflight();
        let rs1 = ctx.read_var(self.rs1_reg);
        let rs2 = ctx.read_var(self.rs2_reg);
        let rd = ctx.read_var(self.rd_reg);
        emit_pointer_alignment_guard(ctx, &[&rd, &rs1, &rs2]);
        let point_dwords = self.curve.point_dwords();
        if is_preflight {
            // Two point reads followed by one point write happen inside the opaque call.
            ctx.advance_timestamp(3 * point_dwords);
        }
        let setup_prefix = if self.is_setup { "setup_" } else { "" };
        let suffix = self.curve.c_suffix();
        let name = format!("rvr_ext_{setup_prefix}ec_add_ne_{suffix}");
        if self.is_setup {
            ctx.emit_checked_call(&name, &["state", &rd, &rs1, &rs2]);
        } else {
            ctx.emit_call(&name, &["state", &rd, &rs1, &rs2]);
        }
        // Add setup constrains only its modulus input; y1, x2, and y2 remain execution data.
        // Its postimage is therefore no less authoritative than a regular add postimage.
        for word in 0..point_dwords {
            ctx.append_replay_value(&format!("peek_mem_u64(state, {rd} + {}ull)", word * 8));
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
        let is_preflight = ctx.is_preflight();
        let rs1 = ctx.read_var(self.rs1_reg);
        let rd = ctx.read_var(self.rd_reg);
        emit_pointer_alignment_guard(ctx, &[&rd, &rs1]);
        let point_dwords = self.curve.point_dwords();
        if is_preflight {
            // One point read followed by one point write happens inside the opaque call.
            ctx.advance_timestamp(2 * point_dwords);
        }
        let setup_prefix = if self.is_setup { "setup_" } else { "" };
        let suffix = self.curve.c_suffix();
        let name = format!("rvr_ext_{setup_prefix}ec_double_{suffix}");
        if self.is_setup {
            ctx.emit_checked_call(&name, &["state", &rd, &rs1]);
        } else {
            ctx.emit_call(&name, &["state", &rd, &rs1]);
        }
        if !self.is_setup {
            // Regular-operation outputs are the only replay values. Setup replay derives its writes
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

    fn supports_preflight(&self) -> bool {
        true
    }
}

/// IR node for EC scalar multiplication.
#[derive(Debug, Clone)]
pub struct EcMulInstr {
    pub rd_reg: Variable,
    pub rs1_reg: Variable,
    pub rs2_reg: Variable,
    curve: KnownCurve,
    pub is_setup: bool,
    /// The `EC_MUL` chip, which spends [`EC_MUL_TRACE_ROWS`] rows per instruction.
    pub mul_chip_idx: Option<AirIndex>,
}

impl ExtInstr for EcMulInstr {
    fn opname(&self) -> &str {
        "ec_mul"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let is_preflight = ctx.is_preflight();
        let rs1 = ctx.read_var(self.rs1_reg);
        let rs2 = ctx.read_var(self.rs2_reg);
        let rd = ctx.read_var(self.rd_reg);
        emit_pointer_alignment_guard(ctx, &[&rd, &rs1, &rs2]);
        let point_dwords = self.curve.point_dwords();
        if is_preflight {
            // One point read, one scalar read, and one point write happen inside the opaque call.
            // The chip performs the same accesses on its final compute row.
            ctx.advance_timestamp(2 * point_dwords + EC_MUL_SCALAR_DWORDS);
        }
        let suffix = self.curve.c_suffix();
        if self.is_setup {
            // Setup reads `(modulus, a)` from the point operand and ignores the scalar's value, but
            // still reads it: the chip reads the scalar on every row, so the access sequences
            // match.
            let name = format!("rvr_ext_setup_ec_mul_{suffix}");
            ctx.emit_checked_call(&name, &["state", &rd, &rs1, &rs2]);
        } else {
            let name = format!("rvr_ext_ec_mul_{suffix}");
            ctx.emit_call(&name, &["state", &rd, &rs1, &rs2]);
        }
        // Setup constrains its modulus and curve coefficient inputs, so its postimage is no less
        // authoritative than a regular multiplication's.
        for word in 0..point_dwords {
            ctx.append_replay_value(&format!("peek_mem_u64(state, {rd} + {}ull)", word * 8));
        }
    }

    fn fixed_trace_rows(&self) -> Vec<FixedTraceRows> {
        // The PC-to-chip mapping already counts one row for this instruction, so declare the rest.
        // Setup rows are no cheaper: the chip lays out a full ladder either way.
        fixed_trace_rows_for_chip(self.mul_chip_idx, EC_MUL_TRACE_ROWS - 1)
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

// ── Extension ─────────────────────────────────────────────────────────────────

/// Information about a registered curve (for Weierstrass ECC opcodes).
#[derive(Debug, Clone)]
pub struct CurveInfo {
    curve: Option<KnownCurve>,
    /// AIR index of this curve's `EC_MUL` chip, resolved once at lift time. `None` for curves
    /// whose `EC_MUL` this extension does not lift, and in pure mode.
    mul_chip_idx: Option<AirIndex>,
}

/// The ECC extension: handles Weierstrass EC opcodes (EC_ADD_NE, EC_DOUBLE, EC_MUL + setups).
pub struct EccExtension {
    curves: Vec<CurveInfo>,
}

impl EccExtension {
    fn has_bn254(&self) -> bool {
        self.curves
            .iter()
            .any(|info| matches!(info.curve, Some(KnownCurve::Bn254)))
    }

    /// Record the curves in configuration order and resolve the `EC_MUL` chip index for each one
    /// whose scalar multiplication this extension lifts.
    fn build(
        ctx: Option<&RvrExtensionCtx>,
        curves: Vec<Option<KnownCurve>>,
    ) -> Result<Self, ExtensionError> {
        let curves = curves
            .into_iter()
            .enumerate()
            .map(|(curve_idx, curve)| {
                let mul_chip_idx = match curve {
                    Some(_) => ec_mul_air_idx(ctx, curve_idx)?,
                    None => None,
                };
                Ok(CurveInfo {
                    curve,
                    mul_chip_idx,
                })
            })
            .collect::<Result<Vec<_>, ExtensionError>>()?;
        Ok(Self { curves })
    }

    pub fn new(
        ctx: Option<&RvrExtensionCtx>,
        curves_info: Vec<u32>,
    ) -> Result<Self, ExtensionError> {
        Self::build(
            ctx,
            curves_info.into_iter().map(KnownCurve::from_id).collect(),
        )
    }

    pub fn new_from_struct_names(
        ctx: Option<&RvrExtensionCtx>,
        struct_names: Vec<String>,
    ) -> Result<Self, ExtensionError> {
        Self::build(
            ctx,
            struct_names
                .iter()
                .map(|name| KnownCurve::from_struct_name(name))
                .collect(),
        )
    }
}

impl RvrExtension for EccExtension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        let ecc_base = WeierstrassOpcode::CLASS_OFFSET;
        let ecc_count = WeierstrassOpcode::COUNT;

        if opcode < ecc_base {
            return None;
        }
        let offset = opcode - ecc_base;
        let curve_idx = offset / ecc_count;
        let local_op = offset % ecc_count;

        let info = self.curves.get(curve_idx)?;
        let curve = info.curve?;

        let rd_reg = decode_reg(insn.a);
        let rs1_reg = decode_reg(insn.b);

        let local_opcode = WeierstrassOpcode::from_repr(local_op)?;
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
            EC_MUL | SETUP_EC_MUL => {
                let rs2_reg = decode_reg(insn.c);
                Box::new(EcMulInstr {
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    curve,
                    is_setup: local_opcode == SETUP_EC_MUL,
                    mul_chip_idx: info.mul_chip_idx,
                })
            }
        };

        Some(LiftedInstr::Body(InstrAt {
            pc,
            instr,
            source_loc: None,
        }))
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        // The modular extension supplies the native K-256 and BLS12-381 point
        // functions declared in this header. ECC supplies BN254 EC_MUL.
        vec![("rvr_ext_ecc.h", include_str!("../c/rvr_ext_ecc.h"))]
    }

    fn c_sources(&self) -> Vec<(&'static str, &'static str)> {
        if self.has_bn254() {
            vec![(
                "rvr_ext_bn254.c",
                include_str!("../ffi/native/c/rvr_ext_bn254.c"),
            )]
        } else {
            Vec::new()
        }
    }

    fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        let mut files = vec![(
            "librvr_openvm_ext_ecc_ffi.a",
            include_bytes!(env!("RVR_ECC_FFI_STATICLIB")).as_slice(),
        )];
        if self.has_bn254() {
            files.push((
                "libmcl.a",
                include_bytes!(env!("RVR_ECC_MCL_STATICLIB")).as_slice(),
            ));
        }
        files
    }

    fn extra_c_include_files(&self) -> Vec<(&'static str, &'static str)> {
        if self.has_bn254() {
            vec![
                (
                    "mcl/include/mcl/bn.h",
                    include_str!("../ffi/native/mcl/include/mcl/bn.h"),
                ),
                (
                    "mcl/include/mcl/bn_c384_256.h",
                    include_str!("../ffi/native/mcl/include/mcl/bn_c384_256.h"),
                ),
                (
                    "mcl/include/mcl/curve_type.h",
                    include_str!("../ffi/native/mcl/include/mcl/curve_type.h"),
                ),
            ]
        } else {
            Vec::new()
        }
    }

    fn extra_cflags(&self) -> Vec<String> {
        if self.has_bn254() {
            vec![
                "-isystem".to_string(),
                "mcl/include".to_string(),
                "-Wno-global-constructors".to_string(),
            ]
        } else {
            Vec::new()
        }
    }

    fn requires_cxx_linker(&self) -> bool {
        self.has_bn254()
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
        preflight: bool,
        next_tmp: usize,
    }

    impl TestEmitCtx {
        fn preflight() -> Self {
            Self {
                operations: Vec::new(),
                preflight: true,
                next_tmp: 0,
            }
        }

        fn legacy() -> Self {
            Self {
                operations: Vec::new(),
                preflight: false,
                next_tmp: 0,
            }
        }
    }

    impl ExtEmitCtx for TestEmitCtx {
        fn is_preflight(&self) -> bool {
            self.preflight
        }

        fn read_var(&mut self, var: Variable) -> String {
            let value = format!("r{}", var.index());
            self.operations.push(format!("read({value})"));
            value
        }

        fn peek_var(&mut self, _var: Variable) -> String {
            unreachable!()
        }

        fn advance_timestamp(&mut self, slots: u32) {
            self.operations.push(format!("timestamp_slots({slots})"));
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

        fn reserve_preflight_timestamp_slots(&mut self, _slots: &str) {
            unreachable!()
        }

        fn append_replay_value(&mut self, value: &str) {
            if self.preflight {
                self.operations.push(format!("replay_value({value})"));
            }
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

    fn expected_replay_values(rd: &str, point_dwords: u32) -> Vec<String> {
        (0..point_dwords)
            .map(|word| format!("replay_value(peek_mem_u64(state, {rd} + {}ull))", word * 8))
            .collect()
    }

    #[test]
    fn ignores_opcodes_outside_configured_curves() {
        let extension = EccExtension::new(None, vec![0]).unwrap();
        let opcode =
            VmOpcode::from_usize(WeierstrassOpcode::CLASS_OFFSET + WeierstrassOpcode::COUNT);
        let insn = RvrInstruction::from_canonical(opcode, [0; 7], u32::MAX);

        assert!(extension.try_lift(&insn, 0x100).is_none());
    }

    #[test]
    fn add_preflight_matches_schedule_and_minimal_replay_values() {
        for (curve, point_dwords) in [(KnownCurve::K256, 8), (KnownCurve::Bls12381, 12)] {
            for is_setup in [false, true] {
                let instruction = EcAddNeInstr {
                    rd_reg: Variable::new(1),
                    rs1_reg: Variable::new(2),
                    rs2_reg: Variable::new(3),
                    curve,
                    is_setup,
                };
                assert!(instruction.supports_preflight());

                let mut preflight = TestEmitCtx::preflight();
                instruction.emit_c(&mut preflight);
                let mut expected = vec![
                    "read(r2)".to_string(),
                    "read(r3)".to_string(),
                    "read(r1)".to_string(),
                    "if (unlikely(((r1 | r2 | r3) & 7ull) != 0ull)) {".to_string(),
                    "trap".to_string(),
                    "}".to_string(),
                    format!("timestamp_slots({})", 3 * point_dwords),
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
                expected.extend(expected_replay_values("r1", point_dwords));
                assert_eq!(preflight.operations, expected);
            }
        }
    }

    #[test]
    fn double_preflight_matches_schedule_and_minimal_replay_values() {
        for (curve, point_dwords) in [(KnownCurve::P256, 8), (KnownCurve::Bls12381, 12)] {
            for is_setup in [false, true] {
                let instruction = EcDoubleInstr {
                    rd_reg: Variable::new(1),
                    rs1_reg: Variable::new(2),
                    curve,
                    is_setup,
                };
                assert!(instruction.supports_preflight());

                let mut preflight = TestEmitCtx::preflight();
                instruction.emit_c(&mut preflight);
                let mut expected = vec![
                    "read(r2)".to_string(),
                    "read(r1)".to_string(),
                    "if (unlikely(((r1 | r2) & 7ull) != 0ull)) {".to_string(),
                    "trap".to_string(),
                    "}".to_string(),
                    format!("timestamp_slots({})", 2 * point_dwords),
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
                    expected.extend(expected_replay_values("r1", point_dwords));
                }
                assert_eq!(preflight.operations, expected);
            }
        }
    }

    #[test]
    fn mul_preflight_matches_schedule_and_minimal_replay_values() {
        for curve in [
            KnownCurve::K256,
            KnownCurve::P256,
            KnownCurve::Bn254,
            KnownCurve::Bls12381,
        ] {
            let point_dwords = curve.point_dwords();
            for is_setup in [false, true] {
                let instruction = EcMulInstr {
                    rd_reg: Variable::new(1),
                    rs1_reg: Variable::new(2),
                    rs2_reg: Variable::new(3),
                    curve,
                    is_setup,
                    mul_chip_idx: Some(AirIndex::new(7)),
                };
                assert!(instruction.supports_preflight());

                let mut preflight = TestEmitCtx::preflight();
                instruction.emit_c(&mut preflight);
                let mut expected = vec![
                    "read(r2)".to_string(),
                    "read(r3)".to_string(),
                    "read(r1)".to_string(),
                    "if (unlikely(((r1 | r2 | r3) & 7ull) != 0ull)) {".to_string(),
                    "trap".to_string(),
                    "}".to_string(),
                    // One point read, a four-word scalar read, and one point write, matching the
                    // accesses the chip's final compute row performs.
                    format!(
                        "timestamp_slots({})",
                        2 * point_dwords + EC_MUL_SCALAR_DWORDS
                    ),
                ];
                if is_setup {
                    let name = format!("rvr_ext_setup_ec_mul_{}", curve.c_suffix());
                    expected.extend([
                        format!("bool tmp0 = {name}(state, r1, r2, r3)"),
                        "if (unlikely(!tmp0)) {".to_string(),
                        "trap".to_string(),
                        "}".to_string(),
                    ]);
                } else {
                    let name = format!("rvr_ext_ec_mul_{}", curve.c_suffix());
                    expected.push(format!("{name}(state, r1, r2, r3)"));
                }
                expected.extend(expected_replay_values("r1", point_dwords));
                assert_eq!(preflight.operations, expected);

                // The PC-to-chip mapping counts the first row; the rest are declared here, so
                // metered execution predicts the same height the chip fills.
                assert_eq!(
                    instruction.fixed_trace_rows(),
                    vec![FixedTraceRows {
                        chip_idx: 7,
                        count: EC_MUL_TRACE_ROWS - 1,
                    }]
                );
            }
        }
    }

    #[test]
    fn mul_lifts_for_every_configured_curve() {
        // `set_up_once` emits `SETUP_EC_MUL` for every declared curve, including those with a
        // cofactor whose scalar multiplication no guest library exposes, so all of them must lift.
        for curve_id in 0..4 {
            let extension = EccExtension::new(None, vec![curve_id]).unwrap();
            for local in [EC_MUL, SETUP_EC_MUL] {
                let opcode = VmOpcode::from_usize(WeierstrassOpcode::CLASS_OFFSET + local as usize);
                let insn = RvrInstruction::from_canonical(opcode, [0; 7], u32::MAX);
                assert!(
                    extension.try_lift(&insn, 0x100).is_some(),
                    "curve {curve_id}, {local:?}"
                );
            }
        }
    }

    #[test]
    fn execution_modes_use_air_operand_order_without_preflight_data() {
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
                "read(r2)",
                "read(r3)",
                "read(r1)",
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
                "read(r2)",
                "read(r1)",
                "if (unlikely(((r1 | r2) & 7ull) != 0ull)) {",
                "trap",
                "}",
                "rvr_ext_ec_double_p256(state, r1, r2)",
            ]
        );
    }
}
