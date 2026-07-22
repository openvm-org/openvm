//! Fp2 (complex extension field) IR nodes and the [`Fp2RvrExtension`] lifter.

use num_bigint::BigUint;
use openvm_algebra_transpiler::Fp2Opcode;
use openvm_instructions::LocalOpcode;
use rvr_openvm_ir::{ExtInstr, InstrAt, LiftedInstr};
use rvr_openvm_lift::{RvrExtension, RvrInstruction};
use strum::EnumCount;

use crate::{
    decode_reg, pad_modulus, ArithKind, FieldArithInstr, FieldKind, FieldSetupInstr, KnownField,
    ModOp, SetupKind,
};

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
