//! Fp2 (complex extension field) IR nodes and the [`Fp2RvrExtension`] lifter.

use num_bigint::BigUint;
use openvm_algebra_transpiler::Fp2Opcode;
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode, VmOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, InlineRecordShape, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{
    air_index_codegen_fingerprint, air_index_to_c, helpers::decode_reg, AirIndex, ExtensionError,
    RvrExtension, RvrExtensionCtx,
};
use strum::EnumCount;

use crate::{
    detect_known_field, format_c_byte_array, ModOp, VecHeapRecordDescriptor,
    VEC_HEAP_RECORD_C_HEADER,
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
    pub from_pc: u32,
    pub local_opcode: u8,
    pub chip_idx: Option<AirIndex>,
    pub emit_inline: bool,
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
        let emit_inline = self.emit_inline && ctx.inline_record_enabled();
        if emit_inline {
            ctx.write_line("preflight_begin_custom_memory_capture(state);");
        }
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        let rd = ctx.read_reg(self.rd_reg);
        let op_name = self.op.c_name();
        let fp2_suffix = detect_known_field(&self.modulus).and_then(|f| f.fp2_c_suffix());
        if let Some(suffix) = fp2_suffix {
            let name = format!("rvr_ext_fp2_{op_name}_{suffix}");
            ctx.extern_call(&name, &["state", &rd, &rs1, &rs2]);
        } else {
            let mod_literal = format_c_byte_array(&self.modulus);
            ctx.write_line("{");
            ctx.write_line(&format!("static const uint8_t mod_[] = {mod_literal};"));
            let name = format!("rvr_ext_fp2_{op_name}");
            let num_limbs = format!("{}u", self.num_limbs);
            ctx.extern_call(&name, &["state", &rd, &rs1, &rs2, &num_limbs, "mod_"]);
            ctx.write_line("}");
        }
        if emit_inline {
            emit_vec_heap_record(
                ctx,
                self.from_pc,
                self.local_opcode,
                self.num_limbs,
                self.chip_idx,
            );
        }
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        self.emit_inline.then(|| InlineRecordShape::Custom {
            record_size: VecHeapRecordDescriptor::new(self.num_limbs as usize * 2).record_size,
        })
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
    pub from_pc: u32,
    pub local_opcode: u8,
    pub chip_idx: Option<AirIndex>,
    pub emit_inline: bool,
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
        let emit_inline = self.emit_inline && ctx.inline_record_enabled();
        if emit_inline {
            ctx.write_line("preflight_begin_custom_memory_capture(state);");
        }
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        let rd = ctx.read_reg(self.rd_reg);
        let num_limbs = format!("{}u", self.num_limbs);
        ctx.extern_call("rvr_ext_fp2_setup", &["state", &rd, &rs1, &rs2, &num_limbs]);
        if emit_inline {
            emit_vec_heap_record(
                ctx,
                self.from_pc,
                self.local_opcode,
                self.num_limbs,
                self.chip_idx,
            );
        }
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        self.emit_inline.then(|| InlineRecordShape::Custom {
            record_size: VecHeapRecordDescriptor::new(self.num_limbs as usize * 2).record_size,
        })
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

// ── Fp2 extension ────────────────────────────────────────────────────────────

/// Fp2 (complex extension field) arithmetic. Self-contained: owns its own
/// Rust FFI staticlib and ships only `rvr_ext_fp2.h`. No lift-time C, no
/// dependency on [`crate::ModularRvrExtension`].
pub struct Fp2RvrExtension {
    fp2_moduli: Vec<ModulusInfo>,
    air_indices: Vec<[Option<AirIndex>; Fp2Opcode::COUNT]>,
}

impl Fp2RvrExtension {
    pub fn new(
        fp2_moduli: Vec<BigUint>,
        ctx: Option<&RvrExtensionCtx>,
    ) -> Result<Self, ExtensionError> {
        let fp2_moduli = make_moduli(fp2_moduli);
        let mut air_indices = Vec::with_capacity(fp2_moduli.len());
        for mod_idx in 0..fp2_moduli.len() {
            let mut indices = [None; Fp2Opcode::COUNT];
            for (local, index) in indices.iter_mut().enumerate() {
                let opcode = VmOpcode::from_usize(
                    Fp2Opcode::CLASS_OFFSET + mod_idx * Fp2Opcode::COUNT + local,
                );
                *index = resolve_air_index(ctx, opcode)?;
            }
            air_indices.push(indices);
        }
        Ok(Self {
            fp2_moduli,
            air_indices,
        })
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

impl<F: PrimeField32> RvrExtension<F> for Fp2RvrExtension {
    fn codegen_fingerprint(&self) -> Option<Vec<u8>> {
        let mut fingerprint = b"openvm-fp2-rvr-v2\0".to_vec();
        fingerprint.extend_from_slice(&(self.fp2_moduli.len() as u64).to_le_bytes());
        for modulus in &self.fp2_moduli {
            fingerprint.extend_from_slice(&modulus.num_limbs.to_le_bytes());
            fingerprint.extend_from_slice(&(modulus.modulus_bytes.len() as u64).to_le_bytes());
            fingerprint.extend_from_slice(&modulus.modulus_bytes);
        }
        let indices = self
            .air_indices
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        fingerprint.extend_from_slice(&air_index_codegen_fingerprint(
            b"openvm-fp2-air-indices-v1",
            &indices,
        ));
        Some(fingerprint)
    }

    fn try_lift(&self, insn: &Instruction<F>, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();
        self.try_lift_fp2(insn, pc, opcode)
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("rvr_ext_vec_heap_record.h", VEC_HEAP_RECORD_C_HEADER),
            ("rvr_ext_fp2.h", include_str!("../c/rvr_ext_fp2.h")),
        ]
    }

    fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        vec![(
            "librvr_openvm_ext_algebra_fp2_ffi.a",
            include_bytes!(env!("RVR_ALGEBRA_FP2_FFI_STATICLIB")),
        )]
    }
}

impl Fp2RvrExtension {
    fn try_lift_fp2<F: PrimeField32>(
        &self,
        insn: &Instruction<F>,
        pc: u64,
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
        let chip_idx = self.air_indices[fp2_idx][local];
        let rd_reg = decode_reg(insn.a);
        let rs1_reg = decode_reg(insn.b);
        let rs2_reg = decode_reg(insn.c);
        let from_pc = pc as u32;
        let local_opcode = local as u8;
        let emit_inline = insn.d.as_canonical_u32() == RV64_REGISTER_AS
            && insn.e.as_canonical_u32() == RV64_MEMORY_AS;

        let instr: Instr = match local {
            x if x == Fp2Opcode::ADD as usize => Instr::Ext(Box::new(Fp2ArithInstr {
                from_pc,
                local_opcode,
                chip_idx,
                emit_inline,
                op: ModOp::Add,
                rd_reg,
                rs1_reg,
                rs2_reg,
                num_limbs: info.num_limbs,
                modulus: info.modulus_bytes.clone(),
            })),
            x if x == Fp2Opcode::SUB as usize => Instr::Ext(Box::new(Fp2ArithInstr {
                from_pc,
                local_opcode,
                chip_idx,
                emit_inline,
                op: ModOp::Sub,
                rd_reg,
                rs1_reg,
                rs2_reg,
                num_limbs: info.num_limbs,
                modulus: info.modulus_bytes.clone(),
            })),
            x if x == Fp2Opcode::SETUP_ADDSUB as usize => Instr::Ext(Box::new(Fp2SetupInstr {
                from_pc,
                local_opcode,
                chip_idx,
                emit_inline,
                rd_reg,
                rs1_reg,
                rs2_reg,
                num_limbs: info.num_limbs,
            })),
            x if x == Fp2Opcode::MUL as usize => Instr::Ext(Box::new(Fp2ArithInstr {
                from_pc,
                local_opcode,
                chip_idx,
                emit_inline,
                op: ModOp::Mul,
                rd_reg,
                rs1_reg,
                rs2_reg,
                num_limbs: info.num_limbs,
                modulus: info.modulus_bytes.clone(),
            })),
            x if x == Fp2Opcode::DIV as usize => Instr::Ext(Box::new(Fp2ArithInstr {
                from_pc,
                local_opcode,
                chip_idx,
                emit_inline,
                op: ModOp::Div,
                rd_reg,
                rs1_reg,
                rs2_reg,
                num_limbs: info.num_limbs,
                modulus: info.modulus_bytes.clone(),
            })),
            x if x == Fp2Opcode::SETUP_MULDIV as usize => Instr::Ext(Box::new(Fp2SetupInstr {
                from_pc,
                local_opcode,
                chip_idx,
                emit_inline,
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

fn emit_vec_heap_record(
    ctx: &mut dyn ExtEmitCtx,
    from_pc: u32,
    local_opcode: u8,
    num_limbs: u32,
    chip_idx: Option<AirIndex>,
) {
    ctx.extern_call(
        "rvr_ext_emit_vec_heap_record",
        &[
            "state",
            &format!("{from_pc}u"),
            &format!("{local_opcode}u"),
            &format!("{}u", num_limbs * 2),
            "2u",
            &format!("{}u", air_index_to_c(chip_idx)),
        ],
    );
}
