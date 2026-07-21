//! ECC extension for rvr-openvm.
//!
//! Provides IR nodes and extension trait implementation for the short Weierstrass
//! elliptic curve opcodes (EC_ADD_NE, EC_DOUBLE + setups).
//!
//! Modular arithmetic opcodes are handled separately by the algebra extension.

use openvm_ecc_transpiler::Rv64WeierstrassOpcode::{
    self, EC_ADD_NE, EC_DOUBLE, SETUP_EC_ADD_NE, SETUP_EC_DOUBLE,
};
use openvm_instructions::LocalOpcode;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{helpers::decode_reg, RvrExtension, RvrInstruction};
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

        assert!(
            curve_idx < self.curves.len(),
            "ECC opcode references unregistered curve_idx {curve_idx} (only {} curves registered)",
            self.curves.len(),
        );
        // Skip lifting opcodes for curves not in the rvr-known set.
        let curve = self.curves[curve_idx].curve?;

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
            instr: Instr::Ext(instr),
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
}
