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
        ctx.emit_call(self.curve.ffi_symbol(), &["state", &rs1, &rs2]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
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

    fn uses_memory_wrappers(&self) -> bool {
        true
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        // Pairing's guest-memory reads record zero main-memory page entries.
        0
    }
}
