//! Pairing extension for rvr-openvm.
//!
//! Provides an IR node for the `HintFinalExp` phantom instruction and the
//! `PairingExtension` for lifting and executing it via FFI.

use std::path::{Path, PathBuf};

use openvm_instructions::{instruction::Instruction, LocalOpcode, SystemOpcode};
use openvm_pairing_transpiler::PairingPhantom;
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{helpers::decode_reg, RvrExtension};

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
    pub rs1_reg: Reg,
    /// Register holding pointer to Q slice header (data_ptr, len).
    pub rs2_reg: Reg,
    /// Pairing curve, resolved at lift time.
    curve: KnownPairingCurve,
}

impl ExtInstr for HintFinalExpInstr {
    fn opname(&self) -> &str {
        "hint_finalexp"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        ctx.write_line(&format!(
            "{}(state, {rs1}, {rs2});",
            self.curve.ffi_symbol(),
        ));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// The Pairing extension (HintFinalExp phantom instruction).
/// Register this with the `ExtensionRegistry`.
pub struct PairingExtension {
    staticlib_path: PathBuf,
}

impl PairingExtension {
    pub fn new(staticlib_path: PathBuf) -> Self {
        Self { staticlib_path }
    }
}

impl<F: PrimeField32> RvrExtension<F> for PairingExtension {
    fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode != SystemOpcode::PHANTOM.global_opcode_usize() {
            return None;
        }

        let c_val = insn.c.as_canonical_u32();
        let discriminant = (c_val & 0xffff) as u16;
        let curve_idx = (c_val >> 16) as u16;

        if discriminant != PairingPhantom::HintFinalExp as u16 {
            return None;
        }

        let rs1_reg = decode_reg(insn.a);
        let rs2_reg = decode_reg(insn.b);
        let curve = KnownPairingCurve::from_idx(curve_idx)
            .unwrap_or_else(|| panic!("unsupported pairing curve idx: {curve_idx}"));

        Some(LiftedInstr::Body(InstrAt {
            pc,
            instr: Instr::Ext(Box::new(HintFinalExpInstr {
                rs1_reg,
                rs2_reg,
                curve,
            })),
            source_loc: None,
        }))
    }

    fn c_headers(&self) -> Vec<(&str, &str)> {
        vec![("rvr_ext_pairing.h", include_str!("../c/rvr_ext_pairing.h"))]
    }

    fn staticlib_path(&self) -> &Path {
        &self.staticlib_path
    }
}
