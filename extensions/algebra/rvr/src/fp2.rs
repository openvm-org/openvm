//! Fp2 (complex extension field) IR nodes and the [`Fp2RvrExtension`] lifter.

use num_bigint::BigUint;
use openvm_algebra_transpiler::Fp2Opcode;
use openvm_instructions::{
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode, VmOpcode,
};
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, InstrAt, LiftedInstr};
use rvr_openvm_lift::{
    air_index_codegen_fingerprint, air_index_to_c, max_main_memory_pages_for_contiguous_range,
    AirIndex, ExtensionError, RvrExtension, RvrExtensionCtx, RvrInstruction,
};
use strum::EnumCount;

use crate::{
    decode_reg, pad_modulus, ArithKind, FieldArithInstr, FieldKind, FieldSetupInstr, KnownField,
    ModOp, SetupKind, VecHeapRecordDescriptor, VEC_HEAP_RECORD_C_HEADER,
};

// An Fp2 operation can read two independent 96-byte values and write one.
const FP2_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION: usize =
    3 * max_main_memory_pages_for_contiguous_range(96);

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

    fn inline_record_size(num_limbs: u32) -> Option<usize> {
        Some(VecHeapRecordDescriptor::new(num_limbs as usize * 2).record_size)
    }

    fn emit_inline_record(
        ctx: &mut dyn ExtEmitCtx,
        from_pc: u32,
        local_opcode: u8,
        num_limbs: u32,
        chip_idx: Option<AirIndex>,
    ) {
        emit_vec_heap_record(ctx, from_pc, local_opcode, num_limbs, chip_idx);
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

impl RvrExtension for Fp2RvrExtension {
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

    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
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

    fn uses_memory_wrappers(&self) -> bool {
        true
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        FP2_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION
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
        let chip_idx = self.air_indices[fp2_idx][local];
        let rd_reg = decode_reg(insn.a);
        let rs1_reg = decode_reg(insn.b);
        let rs2_reg = decode_reg(insn.c);
        let from_pc = pc as u32;
        let local_opcode = local as u8;
        let emit_inline = insn.d == RV64_REGISTER_AS && insn.e == RV64_MEMORY_AS;

        let instr: Box<dyn ExtInstr> = match local {
            x if x == Fp2Opcode::ADD as usize => Box::new(
                Fp2ArithInstr::new(
                    ModOp::Add,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    info.num_limbs,
                    info.modulus_bytes.clone(),
                )
                .with_inline_record(from_pc, local_opcode, chip_idx, emit_inline),
            ),
            x if x == Fp2Opcode::SUB as usize => Box::new(
                Fp2ArithInstr::new(
                    ModOp::Sub,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    info.num_limbs,
                    info.modulus_bytes.clone(),
                )
                .with_inline_record(from_pc, local_opcode, chip_idx, emit_inline),
            ),
            x if x == Fp2Opcode::SETUP_ADDSUB as usize => Box::new(
                Fp2SetupInstr::new(
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    info.num_limbs,
                    info.modulus_bytes.clone(),
                )
                .with_inline_record(from_pc, local_opcode, chip_idx, emit_inline),
            ),
            x if x == Fp2Opcode::MUL as usize => Box::new(
                Fp2ArithInstr::new(
                    ModOp::Mul,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    info.num_limbs,
                    info.modulus_bytes.clone(),
                )
                .with_inline_record(from_pc, local_opcode, chip_idx, emit_inline),
            ),
            x if x == Fp2Opcode::DIV as usize => Box::new(
                Fp2ArithInstr::new(
                    ModOp::Div,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    info.num_limbs,
                    info.modulus_bytes.clone(),
                )
                .with_inline_record(from_pc, local_opcode, chip_idx, emit_inline),
            ),
            x if x == Fp2Opcode::SETUP_MULDIV as usize => Box::new(
                Fp2SetupInstr::new(
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    info.num_limbs,
                    info.modulus_bytes.clone(),
                )
                .with_inline_record(from_pc, local_opcode, chip_idx, emit_inline),
            ),
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
