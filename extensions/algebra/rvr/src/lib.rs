//! Algebra extension for rvr-openvm.
//!
//! Provides IR nodes for modular arithmetic (ADD, SUB, MUL, DIV, IS_EQ, SETUP),
//! Fp2 (complex extension field) operations, and phantom instructions
//! (HintNonQr, HintSqrt), plus the `AlgebraExtension` for lifting and
//! executing them via double FFI.

use std::path::{Path, PathBuf};

use num_bigint::BigUint;
use openvm_algebra_circuit::find_non_qr;
use openvm_algebra_transpiler::{Fp2Opcode, ModularPhantom, Rv32ModularArithmeticOpcode};
use openvm_instructions::instruction::Instruction;
use openvm_instructions::{LocalOpcode, SystemOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{helpers::decode_reg, RvrExtension, RvrExtensionCtx};
use strum::EnumCount;

// ── Modular arithmetic operations ────────────────────────────────────────────

/// Operation type for modular arithmetic.
#[derive(Debug, Clone, Copy)]
pub enum ModOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl ModOp {
    /// Lower-case op name used as a suffix in the generated C function name
    /// (e.g. `rvr_ext_mod_add`, `rvr_ext_mod_sub_k256_coord`).
    fn c_name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
        }
    }
}

// ── Known curve detection for native field arithmetic ─────────────────────────

/// Known field types that have optimized native FFI implementations.
#[derive(Debug, Clone, Copy)]
enum KnownField {
    K256Coord,
    K256Scalar,
    P256Coord,
    P256Scalar,
    Bn254Fq,
    Bn254Fr,
    Bls12381Fq,
    Bls12381Fr,
}

impl KnownField {
    /// C function name suffix for this field.
    fn c_suffix(self) -> &'static str {
        match self {
            Self::K256Coord => "k256_coord",
            Self::K256Scalar => "k256_scalar",
            Self::P256Coord => "p256_coord",
            Self::P256Scalar => "p256_scalar",
            Self::Bn254Fq => "bn254_fq",
            Self::Bn254Fr => "bn254_fr",
            Self::Bls12381Fq => "bls12_381_fq",
            Self::Bls12381Fr => "bls12_381_fr",
        }
    }

    /// Fp2 C function name suffix (only valid for base fields of Fp2-capable curves).
    fn fp2_c_suffix(self) -> Option<&'static str> {
        match self {
            Self::Bn254Fq => Some("bn254"),
            Self::Bls12381Fq => Some("bls12_381"),
            _ => None,
        }
    }
}

/// Known moduli (LE, padded) mapped to their field types.
static KNOWN_FIELDS: &[(&[u8], KnownField)] = &[
    // secp256k1 coordinate field
    (
        &[
            0x2f, 0xfc, 0xff, 0xff, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff,
        ],
        KnownField::K256Coord,
    ),
    // secp256k1 scalar field
    (
        &[
            0x41, 0x41, 0x36, 0xd0, 0x8c, 0x5e, 0xd2, 0xbf, 0x3b, 0xa0, 0x48, 0xaf, 0xe6, 0xdc,
            0xae, 0xba, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff,
        ],
        KnownField::K256Scalar,
    ),
    // secp256r1 coordinate field
    (
        &[
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0xff, 0xff, 0xff, 0xff,
        ],
        KnownField::P256Coord,
    ),
    // secp256r1 scalar field
    (
        &[
            0x51, 0x25, 0x63, 0xfc, 0xc2, 0xca, 0xb9, 0xf3, 0x84, 0x9e, 0x17, 0xa7, 0xad, 0xfa,
            0xe6, 0xbc, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
            0xff, 0xff, 0xff, 0xff,
        ],
        KnownField::P256Scalar,
    ),
    // BN254 base field
    (
        &[
            0x47, 0xfd, 0x7c, 0xd8, 0x16, 0x8c, 0x20, 0x3c, 0x8d, 0xca, 0x71, 0x68, 0x91, 0x6a,
            0x81, 0x97, 0x5d, 0x58, 0x81, 0x81, 0xb6, 0x45, 0x50, 0xb8, 0x29, 0xa0, 0x31, 0xe1,
            0x72, 0x4e, 0x64, 0x30,
        ],
        KnownField::Bn254Fq,
    ),
    // BN254 scalar field
    (
        &[
            0x01, 0x00, 0x00, 0xf0, 0x93, 0xf5, 0xe1, 0x43, 0x91, 0x70, 0xb9, 0x79, 0x48, 0xe8,
            0x33, 0x28, 0x5d, 0x58, 0x81, 0x81, 0xb6, 0x45, 0x50, 0xb8, 0x29, 0xa0, 0x31, 0xe1,
            0x72, 0x4e, 0x64, 0x30,
        ],
        KnownField::Bn254Fr,
    ),
    // BLS12-381 base field (48 bytes)
    (
        &[
            0xab, 0xaa, 0xff, 0xff, 0xff, 0xff, 0xfe, 0xb9, 0xff, 0xff, 0x53, 0xb1, 0xfe, 0xff,
            0xab, 0x1e, 0x24, 0xf6, 0xb0, 0xf6, 0xa0, 0xd2, 0x30, 0x67, 0xbf, 0x12, 0x85, 0xf3,
            0x84, 0x4b, 0x77, 0x64, 0xd7, 0xac, 0x4b, 0x43, 0xb6, 0xa7, 0x1b, 0x4b, 0x9a, 0xe6,
            0x7f, 0x39, 0xea, 0x11, 0x01, 0x1a,
        ],
        KnownField::Bls12381Fq,
    ),
    // BLS12-381 scalar field
    (
        &[
            0x01, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xfe, 0x5b, 0xfe, 0xff, 0x02, 0xa4,
            0xbd, 0x53, 0x05, 0xd8, 0xa1, 0x09, 0x08, 0xd8, 0x39, 0x33, 0x48, 0x7d, 0x9d, 0x29,
            0x53, 0xa7, 0xed, 0x73,
        ],
        KnownField::Bls12381Fr,
    ),
];

/// Detect a known field from its modulus bytes (LE, padded).
fn detect_known_field(modulus_bytes: &[u8]) -> Option<KnownField> {
    KNOWN_FIELDS
        .iter()
        .find(|(bytes, _)| *bytes == modulus_bytes)
        .map(|(_, f)| *f)
}

/// IR node for modular arithmetic (ADD, SUB, MUL, DIV).
#[derive(Debug, Clone)]
pub struct ModArithInstr {
    pub op: ModOp,
    pub rd_reg: Reg,
    pub rs1_reg: Reg,
    pub rs2_reg: Reg,
    pub num_limbs: u32,
    pub modulus: Vec<u8>,
}

impl ExtInstr for ModArithInstr {
    fn opname(&self) -> &str {
        "mod_arith"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_reg(self.rd_reg);
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        let op_name = self.op.c_name();
        if let Some(field) = detect_known_field(&self.modulus) {
            let suffix = field.c_suffix();
            ctx.write_line(&format!(
                "rvr_ext_mod_{op_name}_{suffix}(state, {rd}, {rs1}, {rs2});",
            ));
        } else {
            let mod_literal = format_c_byte_array(&self.modulus);
            ctx.write_line("{");
            ctx.write_line(&format!("static const uint8_t mod_[] = {mod_literal};"));
            ctx.write_line(&format!(
                "rvr_ext_mod_{op_name}(state, {rd}, {rs1}, {rs2}, {}u, mod_);",
                self.num_limbs
            ));
            ctx.write_line("}");
        }
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// IR node for modular IS_EQ.
#[derive(Debug, Clone)]
pub struct ModIsEqInstr {
    pub rd_reg: Reg,
    pub rs1_reg: Reg,
    pub rs2_reg: Reg,
    pub num_limbs: u32,
    pub modulus: Vec<u8>,
}

impl ExtInstr for ModIsEqInstr {
    fn opname(&self) -> &str {
        "mod_iseq"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        if let Some(field) = detect_known_field(&self.modulus) {
            let suffix = field.c_suffix();
            ctx.write_reg(
                self.rd_reg,
                &format!("rvr_ext_mod_iseq_{suffix}(state, {rs1}, {rs2})"),
            );
        } else {
            let mod_literal = format_c_byte_array(&self.modulus);
            ctx.write_line("{");
            ctx.write_line(&format!("static const uint8_t mod_[] = {mod_literal};"));
            ctx.write_reg(
                self.rd_reg,
                &format!(
                    "rvr_ext_mod_iseq(state, {rs1}, {rs2}, {}u, mod_)",
                    self.num_limbs
                ),
            );
            ctx.write_line("}");
        }
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// IR node for modular SETUP (SETUP_ADDSUB, SETUP_MULDIV, SETUP_ISEQ).
#[derive(Debug, Clone)]
pub struct ModSetupInstr {
    pub rd_reg: Reg,
    pub rs1_reg: Reg,
    pub rs2_reg: Reg,
    pub num_limbs: u32,
}

impl ExtInstr for ModSetupInstr {
    fn opname(&self) -> &str {
        "mod_setup"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_reg(self.rd_reg);
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        ctx.write_line(&format!(
            "rvr_ext_mod_setup(state, {rd}, {rs1}, {rs2}, {}u);",
            self.num_limbs
        ));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

// ── Fp2 (complex extension field) operations ─────────────────────────────────

/// IR node for Fp2 arithmetic (ADD, SUB, MUL, DIV).
#[derive(Debug, Clone)]
pub struct Fp2ArithInstr {
    pub op: ModOp,
    pub rd_reg: Reg,
    pub rs1_reg: Reg,
    pub rs2_reg: Reg,
    pub num_limbs: u32,
    pub modulus: Vec<u8>,
}

impl ExtInstr for Fp2ArithInstr {
    fn opname(&self) -> &str {
        "fp2_arith"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_reg(self.rd_reg);
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        let op_name = self.op.c_name();
        let fp2_suffix = detect_known_field(&self.modulus).and_then(|f| f.fp2_c_suffix());
        if let Some(suffix) = fp2_suffix {
            ctx.write_line(&format!(
                "rvr_ext_fp2_{op_name}_{suffix}(state, {rd}, {rs1}, {rs2});",
            ));
        } else {
            let mod_literal = format_c_byte_array(&self.modulus);
            ctx.write_line("{");
            ctx.write_line(&format!("static const uint8_t mod_[] = {mod_literal};"));
            ctx.write_line(&format!(
                "rvr_ext_fp2_{op_name}(state, {rd}, {rs1}, {rs2}, {}u, mod_);",
                self.num_limbs
            ));
            ctx.write_line("}");
        }
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// IR node for Fp2 SETUP (SETUP_ADDSUB, SETUP_MULDIV).
#[derive(Debug, Clone)]
pub struct Fp2SetupInstr {
    pub rd_reg: Reg,
    pub rs1_reg: Reg,
    pub rs2_reg: Reg,
    pub num_limbs: u32,
}

impl ExtInstr for Fp2SetupInstr {
    fn opname(&self) -> &str {
        "fp2_setup"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_reg(self.rd_reg);
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        ctx.write_line(&format!(
            "rvr_ext_fp2_setup(state, {rd}, {rs1}, {rs2}, {}u);",
            self.num_limbs
        ));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
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

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let literal = format_c_byte_array(&self.non_qr_bytes);
        ctx.write_line("{");
        ctx.write_line(&format!("static const uint8_t nqr[] = {literal};"));
        ctx.write_line(&format!(
            "ext_hint_stream_set(nqr, {}u);",
            self.non_qr_bytes.len()
        ));
        ctx.write_line("}");
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// IR node for HintSqrt phantom instruction.
#[derive(Debug, Clone)]
pub struct HintSqrtInstr {
    pub rs1_reg: Reg,
    pub num_limbs: u32,
    pub modulus: Vec<u8>,
    pub non_qr_bytes: Vec<u8>,
}

impl ExtInstr for HintSqrtInstr {
    fn opname(&self) -> &str {
        "hint_sqrt"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rs1 = ctx.read_reg(self.rs1_reg);
        let mod_literal = format_c_byte_array(&self.modulus);
        let nqr_literal = format_c_byte_array(&self.non_qr_bytes);
        ctx.write_line("{");
        ctx.write_line(&format!("static const uint8_t mod_[] = {mod_literal};"));
        ctx.write_line(&format!("static const uint8_t nqr[] = {nqr_literal};"));
        ctx.write_line(&format!(
            "rvr_ext_algebra_hint_sqrt(state, {rs1}, {}u, mod_, nqr);",
            self.num_limbs
        ));
        ctx.write_line("}");
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

// ── Algebra extension ────────────────────────────────────────────────────────

/// Per-modulus info for the algebra extension.
struct ModulusInfo {
    modulus_bytes: Vec<u8>,
    non_qr_bytes: Vec<u8>,
    num_limbs: u32,
}

/// The Algebra extension (modular arithmetic + Fp2 + phantom hints).
pub struct AlgebraExtension {
    moduli: Vec<ModulusInfo>,
    fp2_moduli: Vec<ModulusInfo>,
    staticlib_path: PathBuf,
}

/// Path to the secp256k1 submodule (resolved from this crate's manifest dir).
fn secp256k1_dir() -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("ffi/secp256k1");
    assert!(
        dir.join("src/secp256k1.c").exists(),
        "missing secp256k1 submodule at {}. Run `git submodule update --init --recursive`.",
        dir.display()
    );
    dir
}

impl AlgebraExtension {
    pub fn new_pure(
        moduli: Vec<BigUint>,
        fp2_moduli: Vec<BigUint>,
        staticlib_path: PathBuf,
    ) -> Self {
        // Use the same deterministic seed as OpenVM for non-QR computation.
        let mut rng = StdRng::from_seed([0u8; 32]);
        let moduli = moduli
            .into_iter()
            .map(|m| make_modulus_info(&m, &mut rng))
            .collect();
        // Reuse the same deterministic seed here to match OpenVM's fixed NQR choice.
        let mut rng2 = StdRng::from_seed([0u8; 32]);
        let fp2_moduli = fp2_moduli
            .into_iter()
            .map(|m| make_modulus_info(&m, &mut rng2))
            .collect();
        Self {
            moduli,
            fp2_moduli,
            staticlib_path,
        }
    }

    pub fn new(
        moduli: Vec<BigUint>,
        fp2_moduli: Vec<BigUint>,
        _ctx: &RvrExtensionCtx,
        staticlib_path: PathBuf,
    ) -> Self {
        // Algebra currently uses the pure fast path for both metered and non-metered
        // lifting, so chip mappings are intentionally unused here.
        Self::new_pure(moduli, fp2_moduli, staticlib_path)
    }
}

fn make_modulus_info(modulus: &BigUint, rng: &mut StdRng) -> ModulusInfo {
    let bytes = modulus.bits().div_ceil(8) as usize;
    assert!(
        bytes <= 48,
        "modulus exceeds maximum supported size of 384 bits"
    );
    let num_limbs = if bytes <= 32 { 32u32 } else { 48u32 };
    let mut modulus_bytes = modulus.to_bytes_le();
    modulus_bytes.resize(num_limbs as usize, 0);
    let non_qr = find_non_qr(modulus, rng);
    let mut non_qr_bytes = non_qr.to_bytes_le();
    non_qr_bytes.resize(num_limbs as usize, 0);
    ModulusInfo {
        modulus_bytes,
        non_qr_bytes,
        num_limbs,
    }
}

/// Format a byte slice as a C array initializer: `{0x2f, 0xfc, ...}`
fn format_c_byte_array(bytes: &[u8]) -> String {
    let inner: Vec<String> = bytes.iter().map(|b| format!("0x{b:02x}")).collect();
    format!("{{{}}}", inner.join(","))
}

impl<F: PrimeField32> RvrExtension<F> for AlgebraExtension {
    fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if let Some(lifted) = self.try_lift_modular(insn, pc, opcode) {
            return Some(lifted);
        }

        if let Some(lifted) = self.try_lift_fp2(insn, pc, opcode) {
            return Some(lifted);
        }

        if opcode == SystemOpcode::PHANTOM.global_opcode_usize() {
            if let Some(lifted) = self.try_lift_phantom(insn, pc) {
                return Some(lifted);
            }
        }

        None
    }

    fn c_headers(&self) -> Vec<(&str, &str)> {
        vec![
            ("rvr_ext_algebra.h", include_str!("../c/rvr_ext_algebra.h")),
            ("rvr_ext_k256_fe.h", include_str!("../c/rvr_ext_k256_fe.h")),
        ]
    }

    fn c_sources(&self) -> Vec<(&str, &str)> {
        vec![("rvr_ext_k256.c", include_str!("../c/rvr_ext_k256.c"))]
    }

    fn staticlib_path(&self) -> &Path {
        &self.staticlib_path
    }

    fn extra_c_source_paths(&self) -> Vec<PathBuf> {
        let dir = secp256k1_dir();
        vec![
            dir.join("src/precomputed_ecmult.c"),
            dir.join("src/precomputed_ecmult_gen.c"),
        ]
    }

    fn extra_cflags(&self) -> Vec<String> {
        let dir = secp256k1_dir();
        vec![
            format!("-I{}", dir.join("src").display()),
            format!("-I{}", dir.display()),
            "-DSECP256K1_BUILD".to_string(),
            "-Wno-unused-function".to_string(),
            "-Wno-unused-parameter".to_string(),
        ]
    }
}

impl AlgebraExtension {
    fn try_lift_modular<F: PrimeField32>(
        &self,
        insn: &Instruction<F>,
        pc: u32,
        opcode: usize,
    ) -> Option<LiftedInstr> {
        let base_offset = Rv32ModularArithmeticOpcode::CLASS_OFFSET;
        let count = Rv32ModularArithmeticOpcode::COUNT;

        if opcode < base_offset {
            return None;
        }
        let relative = opcode - base_offset;
        let mod_idx = relative / count;
        let local = relative % count;

        if mod_idx >= self.moduli.len() {
            return None;
        }

        let info = &self.moduli[mod_idx];
        let rd_reg = decode_reg(insn.a);
        let rs1_reg = decode_reg(insn.b);
        let rs2_reg = decode_reg(insn.c);

        let instr: Instr = match local {
            x if x == Rv32ModularArithmeticOpcode::ADD as usize => {
                Instr::Ext(Box::new(ModArithInstr {
                    op: ModOp::Add,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                    modulus: info.modulus_bytes.clone(),
                }))
            }
            x if x == Rv32ModularArithmeticOpcode::SUB as usize => {
                Instr::Ext(Box::new(ModArithInstr {
                    op: ModOp::Sub,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                    modulus: info.modulus_bytes.clone(),
                }))
            }
            x if x == Rv32ModularArithmeticOpcode::SETUP_ADDSUB as usize => {
                Instr::Ext(Box::new(ModSetupInstr {
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                }))
            }
            x if x == Rv32ModularArithmeticOpcode::MUL as usize => {
                Instr::Ext(Box::new(ModArithInstr {
                    op: ModOp::Mul,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                    modulus: info.modulus_bytes.clone(),
                }))
            }
            x if x == Rv32ModularArithmeticOpcode::DIV as usize => {
                Instr::Ext(Box::new(ModArithInstr {
                    op: ModOp::Div,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                    modulus: info.modulus_bytes.clone(),
                }))
            }
            x if x == Rv32ModularArithmeticOpcode::SETUP_MULDIV as usize => {
                Instr::Ext(Box::new(ModSetupInstr {
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                }))
            }
            x if x == Rv32ModularArithmeticOpcode::IS_EQ as usize => {
                Instr::Ext(Box::new(ModIsEqInstr {
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                    modulus: info.modulus_bytes.clone(),
                }))
            }
            x if x == Rv32ModularArithmeticOpcode::SETUP_ISEQ as usize => {
                Instr::Ext(Box::new(ModSetupInstr {
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                }))
            }
            _ => return None,
        };

        Some(LiftedInstr::Body(InstrAt {
            pc,
            instr,
            source_loc: None,
        }))
    }

    fn try_lift_fp2<F: PrimeField32>(
        &self,
        insn: &Instruction<F>,
        pc: u32,
        opcode: usize,
    ) -> Option<LiftedInstr> {
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

        let instr: Instr = match local {
            x if x == Fp2Opcode::ADD as usize => Instr::Ext(Box::new(Fp2ArithInstr {
                op: ModOp::Add,
                rd_reg,
                rs1_reg,
                rs2_reg,
                num_limbs: info.num_limbs,
                modulus: info.modulus_bytes.clone(),
            })),
            x if x == Fp2Opcode::SUB as usize => Instr::Ext(Box::new(Fp2ArithInstr {
                op: ModOp::Sub,
                rd_reg,
                rs1_reg,
                rs2_reg,
                num_limbs: info.num_limbs,
                modulus: info.modulus_bytes.clone(),
            })),
            x if x == Fp2Opcode::SETUP_ADDSUB as usize => Instr::Ext(Box::new(Fp2SetupInstr {
                rd_reg,
                rs1_reg,
                rs2_reg,
                num_limbs: info.num_limbs,
            })),
            x if x == Fp2Opcode::MUL as usize => Instr::Ext(Box::new(Fp2ArithInstr {
                op: ModOp::Mul,
                rd_reg,
                rs1_reg,
                rs2_reg,
                num_limbs: info.num_limbs,
                modulus: info.modulus_bytes.clone(),
            })),
            x if x == Fp2Opcode::DIV as usize => Instr::Ext(Box::new(Fp2ArithInstr {
                op: ModOp::Div,
                rd_reg,
                rs1_reg,
                rs2_reg,
                num_limbs: info.num_limbs,
                modulus: info.modulus_bytes.clone(),
            })),
            x if x == Fp2Opcode::SETUP_MULDIV as usize => Instr::Ext(Box::new(Fp2SetupInstr {
                rd_reg,
                rs1_reg,
                rs2_reg,
                num_limbs: info.num_limbs,
            })),
            _ => return None,
        };

        Some(LiftedInstr::Body(InstrAt {
            pc,
            instr,
            source_loc: None,
        }))
    }

    fn try_lift_phantom<F: PrimeField32>(
        &self,
        insn: &Instruction<F>,
        pc: u32,
    ) -> Option<LiftedInstr> {
        let c_val = insn.c.as_canonical_u32();
        let discriminant = (c_val & 0xffff) as u16;
        let mod_idx = (c_val >> 16) as usize;

        match ModularPhantom::from_repr(discriminant) {
            Some(ModularPhantom::HintNonQr) => {
                let info = self.moduli.get(mod_idx)?;
                Some(LiftedInstr::Body(InstrAt {
                    pc,
                    instr: Instr::Ext(Box::new(HintNonQrInstr {
                        non_qr_bytes: info.non_qr_bytes.clone(),
                    })),
                    source_loc: None,
                }))
            }
            Some(ModularPhantom::HintSqrt) => {
                let info = self.moduli.get(mod_idx)?;
                let rs1_reg = decode_reg(insn.a);
                Some(LiftedInstr::Body(InstrAt {
                    pc,
                    instr: Instr::Ext(Box::new(HintSqrtInstr {
                        rs1_reg,
                        num_limbs: info.num_limbs,
                        modulus: info.modulus_bytes.clone(),
                        non_qr_bytes: info.non_qr_bytes.clone(),
                    })),
                    source_loc: None,
                }))
            }
            None => None,
        }
    }
}
