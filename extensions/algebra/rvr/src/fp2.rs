//! Fp2 (complex extension field) IR nodes and the [`Fp2RvrExtension`] lifter.

use std::path::{Path, PathBuf};

use num_bigint::BigUint;
use openvm_algebra_transpiler::Fp2Opcode;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{helpers::decode_reg, RvrExtension};
use strum::EnumCount;

use crate::{detect_known_field, format_c_byte_array, ModOp};

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
    let bytes = modulus.bits().div_ceil(8) as usize;
    assert!(
        bytes <= 48,
        "modulus exceeds maximum supported size of 384 bits"
    );
    let num_limbs = if bytes <= 32 { 32u32 } else { 48u32 };
    let mut modulus_bytes = modulus.to_bytes_le();
    modulus_bytes.resize(num_limbs as usize, 0);
    ModulusInfo {
        modulus_bytes,
        num_limbs,
    }
}

// ── Fp2 arithmetic IR ────────────────────────────────────────────────────────

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

// ── Fp2 extension ────────────────────────────────────────────────────────────

/// Default path to the fp2 FFI staticlib, populated by `build.rs`.
fn default_fp2_staticlib_path() -> PathBuf {
    PathBuf::from(env!("RVR_ALGEBRA_FP2_FFI_STATICLIB"))
}

/// Fp2 (complex extension field) arithmetic. Self-contained: owns its own
/// Rust FFI staticlib and ships only `rvr_ext_fp2.h`. No lift-time C, no
/// dependency on [`crate::ModularRvrExtension`].
pub struct Fp2RvrExtension {
    fp2_moduli: Vec<ModulusInfo>,
    staticlib_path: PathBuf,
}

impl Fp2RvrExtension {
    pub fn new_pure(fp2_moduli: Vec<BigUint>) -> Self {
        Self {
            fp2_moduli: make_moduli(fp2_moduli),
            staticlib_path: default_fp2_staticlib_path(),
        }
    }

    pub fn new(fp2_moduli: Vec<BigUint>) -> Self {
        Self::new_pure(fp2_moduli)
    }
}

impl<F: PrimeField32> RvrExtension<F> for Fp2RvrExtension {
    fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();
        self.try_lift_fp2(insn, pc, opcode)
    }

    fn c_headers(&self) -> Vec<(&str, &str)> {
        vec![("rvr_ext_fp2.h", include_str!("../c/rvr_ext_fp2.h"))]
    }

    fn staticlib_path(&self) -> &Path {
        &self.staticlib_path
    }
}

impl Fp2RvrExtension {
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
}
