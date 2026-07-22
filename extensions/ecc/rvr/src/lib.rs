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
    riscv::{RV64_MEMORY_AS, RV64_NUM_REGISTERS, RV64_REGISTER_AS, RV64_REGISTER_BYTES},
    LocalOpcode, VmOpcode,
};
use rvr_openvm_ext_algebra::{VecHeapRecordDescriptor, VEC_HEAP_RECORD_C_HEADER};
use rvr_openvm_ir::{
    CfgEffect, ExtEmitCtx, ExtInstr, InlineRecordShape, InstrAt, LiftedInstr, Variable,
};
use rvr_openvm_lift::{
    air_index_codegen_fingerprint, air_index_to_c, decode_variable,
    max_main_memory_pages_for_contiguous_range, AirIndex, ExtensionError, RvrExtension,
    RvrExtensionCtx, RvrInstruction,
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
    fn fingerprint_byte(self) -> u8 {
        match self {
            Self::K256 => 0,
            Self::P256 => 1,
            Self::Bn254 => 2,
            Self::Bls12381 => 3,
        }
    }

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

    fn coordinate_bytes(self) -> usize {
        match self {
            Self::K256 | Self::P256 | Self::Bn254 => 32,
            Self::Bls12381 => 48,
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
    pub from_pc: u32,
    pub local_opcode: u8,
    pub chip_idx: Option<AirIndex>,
    pub emit_inline: bool,
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
        let emit_inline = self.emit_inline && ctx.inline_record_enabled();
        if emit_inline {
            ctx.write_line("preflight_begin_custom_memory_capture(state);");
        }
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
        if emit_inline {
            emit_vec_heap_record(
                ctx,
                self.from_pc,
                self.local_opcode,
                self.curve.coordinate_bytes() * 2,
                2,
                self.chip_idx,
            );
        }
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        self.emit_inline.then(|| InlineRecordShape::Custom {
            record_size: VecHeapRecordDescriptor::new_with_reads(
                self.curve.coordinate_bytes() * 2,
                2,
            )
            .record_size,
        })
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
    pub from_pc: u32,
    pub local_opcode: u8,
    pub chip_idx: Option<AirIndex>,
    pub emit_inline: bool,
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
        let emit_inline = self.emit_inline && ctx.inline_record_enabled();
        if emit_inline {
            ctx.write_line("preflight_begin_custom_memory_capture(state);");
        }
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
        if emit_inline {
            emit_vec_heap_record(
                ctx,
                self.from_pc,
                self.local_opcode,
                self.curve.coordinate_bytes() * 2,
                1,
                self.chip_idx,
            );
        }
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        self.emit_inline.then(|| InlineRecordShape::Custom {
            record_size: VecHeapRecordDescriptor::new_with_reads(
                self.curve.coordinate_bytes() * 2,
                1,
            )
            .record_size,
        })
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
    air_indices: [Option<AirIndex>; Rv64WeierstrassOpcode::COUNT],
}

/// The ECC extension: handles Weierstrass EC opcodes (EC_ADD_NE, EC_DOUBLE + setups).
pub struct EccExtension {
    curves: Vec<CurveInfo>,
}

impl EccExtension {
    fn from_struct_names(
        struct_names: Vec<String>,
        ctx: Option<&RvrExtensionCtx>,
    ) -> Result<Self, ExtensionError> {
        let curves = struct_names
            .into_iter()
            .map(|name| KnownCurve::from_struct_name(&name))
            .collect();
        Self::from_curves(curves, ctx)
    }

    fn from_curve_ids(
        curves: Vec<u32>,
        ctx: Option<&RvrExtensionCtx>,
    ) -> Result<Self, ExtensionError> {
        let curves = curves.into_iter().map(KnownCurve::from_id).collect();
        Self::from_curves(curves, ctx)
    }

    fn from_curves(
        curves: Vec<Option<KnownCurve>>,
        ctx: Option<&RvrExtensionCtx>,
    ) -> Result<Self, ExtensionError> {
        let mut infos = Vec::with_capacity(curves.len());
        for (curve_idx, curve) in curves.into_iter().enumerate() {
            let mut air_indices = [None; Rv64WeierstrassOpcode::COUNT];
            for (local, index) in air_indices.iter_mut().enumerate() {
                let opcode = VmOpcode::from_usize(
                    Rv64WeierstrassOpcode::CLASS_OFFSET
                        + curve_idx * Rv64WeierstrassOpcode::COUNT
                        + local,
                );
                *index = resolve_air_index(ctx, opcode)?;
            }
            infos.push(CurveInfo { curve, air_indices });
        }
        Ok(Self { curves: infos })
    }

    pub fn new(curves_info: Vec<u32>) -> Self {
        Self::from_curve_ids(curves_info, None).expect("pure ECC extension construction")
    }

    pub fn new_from_struct_names(struct_names: Vec<String>) -> Self {
        Self::from_struct_names(struct_names, None).expect("pure ECC extension construction")
    }

    pub fn new_from_struct_names_with_ctx(
        struct_names: Vec<String>,
        ctx: Option<&RvrExtensionCtx>,
    ) -> Result<Self, ExtensionError> {
        Self::from_struct_names(struct_names, ctx)
    }
}

fn resolve_air_index(
    ctx: Option<&RvrExtensionCtx>,
    opcode: VmOpcode,
) -> Result<Option<AirIndex>, ExtensionError> {
    let Some(ctx) = ctx else {
        return Ok(None);
    };
    let executor_idx = ctx
        .resolve_opcode_executor_idx(opcode)
        .ok_or(ExtensionError::UnknownOpcode(opcode))?;
    let air_idx = *ctx.executor_idx_to_air_idx.get(executor_idx).ok_or(
        ExtensionError::ExecutorIndexOutOfBounds {
            opcode,
            executor_idx,
        },
    )?;
    Ok(Some(AirIndex::new(air_idx as u32)))
}

impl RvrExtension for EccExtension {
    fn codegen_fingerprint(&self) -> Option<Vec<u8>> {
        let mut fingerprint = b"openvm-ecc-rvr-v1\0".to_vec();
        fingerprint.extend_from_slice(&(self.curves.len() as u64).to_le_bytes());
        fingerprint.extend(
            self.curves
                .iter()
                .map(|curve| curve.curve.map_or(u8::MAX, KnownCurve::fingerprint_byte)),
        );
        let indices = self
            .curves
            .iter()
            .flat_map(|curve| curve.air_indices)
            .collect::<Vec<_>>();
        fingerprint.extend_from_slice(&air_index_codegen_fingerprint(
            b"openvm-ecc-air-indices-v1",
            &indices,
        ));
        Some(fingerprint)
    }

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

        let curve_info = self.curves.get(curve_idx)?;
        let curve = curve_info.curve?;
        let chip_idx = curve_info.air_indices[local_op];

        let rd_reg = decode_reg(insn.a);
        let rs1_reg = decode_reg(insn.b);
        let emit_inline = insn.d == RV64_REGISTER_AS && insn.e == RV64_MEMORY_AS;
        let from_pc = pc as u32;
        let local_opcode_u8 = local_op as u8;

        let local_opcode = Rv64WeierstrassOpcode::from_repr(local_op)?;
        let instr: Box<dyn ExtInstr> = match local_opcode {
            EC_ADD_NE | SETUP_EC_ADD_NE => {
                let rs2_reg = decode_reg(insn.c);
                Box::new(EcAddNeInstr {
                    from_pc,
                    local_opcode: local_opcode_u8,
                    chip_idx,
                    emit_inline,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    curve,
                    is_setup: local_opcode == SETUP_EC_ADD_NE,
                })
            }
            EC_DOUBLE | SETUP_EC_DOUBLE => Box::new(EcDoubleInstr {
                from_pc,
                local_opcode: local_opcode_u8,
                chip_idx,
                emit_inline,
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
        vec![
            ("rvr_ext_vec_heap_record.h", VEC_HEAP_RECORD_C_HEADER),
            ("rvr_ext_ecc.h", include_str!("../c/rvr_ext_ecc.h")),
        ]
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

fn emit_vec_heap_record(
    ctx: &mut dyn ExtEmitCtx,
    from_pc: u32,
    local_opcode: u8,
    num_limbs: usize,
    num_reads: usize,
    chip_idx: Option<AirIndex>,
) {
    ctx.extern_call(
        "rvr_ext_emit_vec_heap_record",
        &[
            "state",
            &format!("{from_pc}u"),
            &format!("{local_opcode}u"),
            &format!("{num_limbs}u"),
            &format!("{num_reads}u"),
            &format!("{}u", air_index_to_c(chip_idx)),
        ],
    );
}
