//! ECC extension for rvr-openvm.
//!
//! Provides IR nodes and extension trait implementation for the short Weierstrass
//! elliptic curve opcodes (EC_ADD_NE, EC_DOUBLE + setups).
//!
//! Modular arithmetic opcodes are handled separately by the algebra extension.

use std::path::{Path, PathBuf};

use openvm_ecc_transpiler::Rv32WeierstrassOpcode::{
    self, EC_ADD_NE, EC_DOUBLE, SETUP_EC_ADD_NE, SETUP_EC_DOUBLE,
};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{helpers::decode_reg, RvrExtension};
use strum::EnumCount;

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
    pub rd_reg: Reg,
    pub rs1_reg: Reg,
    pub rs2_reg: Reg,
    curve: KnownCurve,
    pub is_setup: bool,
}

impl ExtInstr for EcAddNeInstr {
    fn opname(&self) -> &str {
        "ec_add_ne"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_reg(self.rd_reg);
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        let setup_prefix = if self.is_setup { "setup_" } else { "" };
        let suffix = self.curve.c_suffix();
        ctx.write_line(&format!(
            "rvr_ext_{setup_prefix}ec_add_ne_{suffix}(state, {rd}, {rs1}, {rs2});",
        ));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// IR node for EC point doubling.
#[derive(Debug, Clone)]
pub struct EcDoubleInstr {
    pub rd_reg: Reg,
    pub rs1_reg: Reg,
    curve: KnownCurve,
    pub is_setup: bool,
}

impl ExtInstr for EcDoubleInstr {
    fn opname(&self) -> &str {
        "ec_double"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_reg(self.rd_reg);
        let rs1 = ctx.read_reg(self.rs1_reg);
        let setup_prefix = if self.is_setup { "setup_" } else { "" };
        let suffix = self.curve.c_suffix();
        ctx.write_line(&format!(
            "rvr_ext_{setup_prefix}ec_double_{suffix}(state, {rd}, {rs1});",
        ));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
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
    staticlib_path: PathBuf,
}

impl EccExtension {
    fn from_struct_names(struct_names: Vec<String>, staticlib_path: PathBuf) -> Self {
        let curves = struct_names
            .into_iter()
            .map(|name| CurveInfo {
                curve: KnownCurve::from_struct_name(&name),
            })
            .collect();
        Self {
            curves,
            staticlib_path,
        }
    }

    fn from_curve_ids(curves: Vec<u32>, staticlib_path: PathBuf) -> Self {
        let curves = curves
            .into_iter()
            .map(|curve_id| CurveInfo {
                curve: KnownCurve::from_id(curve_id),
            })
            .collect();
        Self {
            curves,
            staticlib_path,
        }
    }

    /// Create for pure execution (chip indices are unused).
    pub fn new_pure(curves: Vec<u32>) -> Self {
        Self::from_curve_ids(curves, default_staticlib_path())
    }

    pub fn new_pure_from_struct_names(struct_names: Vec<String>) -> Self {
        Self::from_struct_names(struct_names, default_staticlib_path())
    }

    pub fn new(curves_info: Vec<u32>) -> Self {
        Self::from_curve_ids(curves_info, default_staticlib_path())
    }

    pub fn new_from_struct_names(struct_names: Vec<String>) -> Self {
        Self::from_struct_names(struct_names, default_staticlib_path())
    }
}

/// Default path to the ECC FFI staticlib, populated by `extensions/ecc/rvr/build.rs`.
fn default_staticlib_path() -> PathBuf {
    PathBuf::from(env!("RVR_ECC_FFI_STATICLIB"))
}

impl<F: PrimeField32> RvrExtension<F> for EccExtension {
    fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        let ecc_base = Rv32WeierstrassOpcode::CLASS_OFFSET;
        let ecc_count = Rv32WeierstrassOpcode::COUNT;

        if opcode < ecc_base {
            return None;
        }
        let offset = opcode - ecc_base;
        let curve_idx = offset / ecc_count;
        let local_op = offset % ecc_count;

        assert!(
            curve_idx < self.curves.len(),
            "ECC opcode references unregistered curve_idx {curve_idx} (only {} curves registered)",
            self.curves.len(),
        );
        // Skip lifting opcodes for curves not in the rvr-known set.
        let curve = self.curves[curve_idx].curve?;

        let rd_reg = decode_reg(insn.a);
        let rs1_reg = decode_reg(insn.b);

        let local_opcode = Rv32WeierstrassOpcode::from_repr(local_op)?;
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
            instr: Instr::Ext(instr),
            source_loc: None,
        }))
    }

    fn c_headers(&self) -> Vec<(&str, &str)> {
        // K-256 EC ops are bundled into the modular staticlib via libsecp256k1
        // (see `extensions/algebra/rvr/ffi/modular/c/rvr_ext_modular.c`); their
        // declarations live in `rvr_ext_ecc.h` alongside the other curves'.
        vec![("rvr_ext_ecc.h", include_str!("../c/rvr_ext_ecc.h"))]
    }

    fn staticlib_path(&self) -> &Path {
        &self.staticlib_path
    }
}
