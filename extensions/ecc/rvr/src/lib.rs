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
        // Match Rv64VecHeapAdapter's memory-bus order: rs1, rs2, then rd.
        let rs1 = ctx.read_var(self.rs1_reg);
        let rs2 = ctx.read_var(self.rs2_reg);
        let rd = ctx.read_var(self.rd_reg);
        let setup_prefix = if self.is_setup { "setup_" } else { "" };
        let suffix = self.curve.c_suffix();
        let name = format!("rvr_ext_{setup_prefix}ec_add_ne_{suffix}");
        if self.is_setup {
            ctx.emit_checked_call(&name, &["state", &rd, &rs1, &rs2]);
        } else {
            ctx.emit_call(&name, &["state", &rd, &rs1, &rs2]);
        }
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
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
        // Match Rv64VecHeapAdapter's memory-bus order: rs1, then rd.
        let rs1 = ctx.read_var(self.rs1_reg);
        let rd = ctx.read_var(self.rd_reg);
        let setup_prefix = if self.is_setup { "setup_" } else { "" };
        let suffix = self.curve.c_suffix();
        let name = format!("rvr_ext_{setup_prefix}ec_double_{suffix}");
        if self.is_setup {
            ctx.emit_checked_call(&name, &["state", &rd, &rs1]);
        } else {
            ctx.emit_call(&name, &["state", &rd, &rs1]);
        }
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
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

    use super::*;

    #[test]
    fn ignores_opcodes_outside_configured_curves() {
        let extension = EccExtension::new(vec![0]);
        let opcode = VmOpcode::from_usize(
            Rv64WeierstrassOpcode::CLASS_OFFSET + Rv64WeierstrassOpcode::COUNT,
        );
        let insn = RvrInstruction::from_canonical(opcode, [0; 7], u32::MAX);

        assert!(extension.try_lift(&insn, 0x100).is_none());
    }
}
