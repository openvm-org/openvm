//! Modular arithmetic IR nodes, phantom hints, and the
//! [`ModularRvrExtension`] lifter.

use std::path::{Path, PathBuf};

use num_bigint::BigUint;
use openvm_algebra_transpiler::{ModularPhantom, Rv32ModularArithmeticOpcode};
use openvm_instructions::{instruction::Instruction, LocalOpcode, SystemOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{helpers::decode_reg, RvrExtension};
use strum::EnumCount;

use crate::{detect_known_field, format_c_byte_array, make_moduli, ModOp, ModulusInfo};

// ── Modular arithmetic IR ────────────────────────────────────────────────────

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

// ── Modular extension ────────────────────────────────────────────────────────

/// Default path to the modular Rust FFI staticlib, populated by `build.rs`.
fn default_modular_staticlib_path() -> PathBuf {
    PathBuf::from(env!("RVR_ALGEBRA_MODULAR_FFI_STATICLIB"))
}

/// Path to the secp256k1 submodule, consumed by `ModularRvrExtension`'s
/// lift-time C registration.
fn secp256k1_dir() -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("ffi/modular/secp256k1");
    assert!(
        dir.join("src/secp256k1.c").exists(),
        "missing secp256k1 submodule at {}. Run `git submodule update --init --recursive`.",
        dir.display()
    );
    dir
}

/// Modular arithmetic + phantom hints. Owns the modular Rust FFI staticlib
/// (built at repo build time) and registers `rvr_ext_modular.c` plus its
/// libsecp256k1 inputs for lift-time compilation. Independent of
/// [`crate::Fp2RvrExtension`].
pub struct ModularRvrExtension {
    moduli: Vec<ModulusInfo>,
    staticlib_path: PathBuf,
}

impl ModularRvrExtension {
    /// Pure-mode constructor; equivalent to [`Self::new`] today (chip indices
    /// are unused by this extension).
    pub fn new_pure(moduli: Vec<BigUint>) -> Self {
        Self {
            moduli: make_moduli(moduli),
            staticlib_path: default_modular_staticlib_path(),
        }
    }

    /// Standard constructor.
    pub fn new(moduli: Vec<BigUint>) -> Self {
        Self::new_pure(moduli)
    }
}

impl<F: PrimeField32> RvrExtension<F> for ModularRvrExtension {
    fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr> {
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

    fn c_headers(&self) -> Vec<(&str, &str)> {
        vec![("rvr_ext_mod.h", include_str!("../c/rvr_ext_mod.h"))]
    }

    fn c_sources(&self) -> Vec<(&str, &str)> {
        vec![(
            "rvr_ext_modular.c",
            include_str!("../ffi/modular/c/rvr_ext_modular.c"),
        )]
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
            // ENABLE_MODULE_RECOVERY keeps the ECC modules compiled in so the
            // k256 EC ops in rvr_ext_modular.c can call into libsecp256k1.
            // (-DSECP256K1_BUILD is not set here — secp256k1.c defines it
            // internally.)
            "-DENABLE_MODULE_RECOVERY".to_string(),
            "-Wno-unused-function".to_string(),
            "-Wno-unused-parameter".to_string(),
            "-Wno-unused-variable".to_string(),
            "-Wno-strict-prototypes".to_string(),
        ]
    }
}

impl ModularRvrExtension {
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
