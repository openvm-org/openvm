//! Modular arithmetic IR nodes, phantom hints, and the
//! [`ModularRvrExtension`] lifter.

use num_bigint::BigUint;
use openvm_algebra_transpiler::{ModularPhantom, Rv64ModularArithmeticOpcode};
use openvm_algebra_utils::{find_non_qr, NQR_RNG_SEED};
use openvm_instructions::{
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode, SystemOpcode, VmOpcode,
};
use rand::{rngs::StdRng, SeedableRng};
use rvr_openvm_ir::{
    CfgEffect, ExtEmitCtx, ExtInstr, InlineRecordShape, InstrAt, LiftedInstr, Variable,
};
use rvr_openvm_lift::{
    air_index_codegen_fingerprint, air_index_to_c, max_main_memory_pages_for_contiguous_range,
    AirIndex, ExtensionError, RvrExtension, RvrExtensionCtx, RvrInstruction,
};
use strum::EnumCount;

use crate::{
    decode_reg, format_c_byte_array, pad_modulus, ArithKind, FieldArithInstr, FieldIsEqInstr,
    FieldKind, FieldSetupInstr, IsEqKind, KnownField, ModOp, SetupKind,
};

include!(concat!(env!("OUT_DIR"), "/secp256k1_files.rs"));

// A modular operation can read two independent 48-byte values and write one.
const MODULAR_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION: usize =
    3 * max_main_memory_pages_for_contiguous_range(48);

const MODULAR_OPCODE_COUNT: usize = Rv64ModularArithmeticOpcode::COUNT;

/// Dense-arena ABI for a two-input VecHeap field-expression record.
///
/// The adapter is a fixed prefix followed by arrays whose lengths are all
/// derived from `blocks`; the core is one opcode byte plus the two input byte
/// strings. Circuit-side ABI assertions pin every generated value to the real
/// Rust record types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VecHeapRecordDescriptor {
    pub num_reads: usize,
    pub blocks: usize,
    pub adapter_size: usize,
    pub adapter_align: usize,
    pub core_size: usize,
    pub core_align: usize,
    pub core_off_dense: usize,
    pub record_size: usize,
    pub reads_aux: usize,
    pub writes_aux: usize,
}

impl VecHeapRecordDescriptor {
    pub const fn new(num_limbs: usize) -> Self {
        Self::new_with_reads(num_limbs, 2)
    }

    /// Describe a VecHeap record with `num_reads` equally-sized inputs.
    ///
    /// Modular and Fp2 use two inputs. Weierstrass doubling uses one, while
    /// addition uses two; keeping the derivation here gives every family the
    /// same dense-record ABI rather than duplicating offset arithmetic.
    pub const fn new_with_reads(num_limbs: usize, num_reads: usize) -> Self {
        assert!(num_limbs.is_multiple_of(8));
        assert!(num_reads > 0);
        let blocks = num_limbs / 8;
        let reads_aux = 20 + 12 * num_reads;
        let writes_aux = reads_aux + 4 * num_reads * blocks;
        let adapter_size = writes_aux + 12 * blocks;
        let core_size = 1 + 8 * num_reads * blocks;
        let record_size = (adapter_size + core_size + 3) & !3;
        Self {
            num_reads,
            blocks,
            adapter_size,
            adapter_align: 4,
            core_size,
            core_align: 1,
            core_off_dense: adapter_size,
            record_size,
            reads_aux,
            writes_aux,
        }
    }
}

/// Dense-arena ABI for the modular equality adapter/core pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModIsEqRecordDescriptor {
    pub blocks: usize,
    pub u16_limbs: usize,
    pub adapter_size: usize,
    pub adapter_align: usize,
    pub core_size: usize,
    pub core_align: usize,
    pub core_off_dense: usize,
    pub record_size: usize,
    pub heap_read_aux: usize,
    pub rd_ptr: usize,
    pub writes_aux: usize,
}

impl ModIsEqRecordDescriptor {
    pub const fn new(num_limbs: usize) -> Self {
        assert!(num_limbs.is_multiple_of(8));
        let blocks = num_limbs / 8;
        let u16_limbs = num_limbs / 2;
        let heap_read_aux = 32;
        let rd_ptr = heap_read_aux + 8 * blocks;
        let writes_aux = rd_ptr + 4;
        let adapter_size = writes_aux + 12;
        let core_size = 2 + 4 * u16_limbs;
        let record_size = (adapter_size + core_size + 3) & !3;
        Self {
            blocks,
            u16_limbs,
            adapter_size,
            adapter_align: 4,
            core_size,
            core_align: 2,
            core_off_dense: adapter_size,
            record_size,
            heap_read_aux,
            rd_ptr,
            writes_aux,
        }
    }
}

/// Per-modulus info for the modular extension. Includes a precomputed non-QR
/// for the `HintNonQr` / `HintSqrt` phantoms.
struct ModulusInfo {
    modulus_bytes: Vec<u8>,
    non_qr_bytes: Vec<u8>,
    num_limbs: u32,
}
fn make_moduli(moduli: Vec<BigUint>) -> Vec<ModulusInfo> {
    // Use the same deterministic seed as the circuit-side `NonQrHintSubEx::new`
    // (single rng across the full modulus list), so rvr-emitted NQRs match
    // what the circuit would compute.
    let mut rng = StdRng::from_seed(NQR_RNG_SEED);
    moduli
        .into_iter()
        .map(|m| make_modulus_info(&m, &mut rng))
        .collect()
}

fn make_modulus_info(modulus: &BigUint, rng: &mut StdRng) -> ModulusInfo {
    let (modulus_bytes, num_limbs) = pad_modulus(modulus);
    let non_qr = find_non_qr(modulus, rng);
    let mut non_qr_bytes = non_qr.to_bytes_le();
    non_qr_bytes.resize(num_limbs as usize, 0);
    ModulusInfo {
        modulus_bytes,
        non_qr_bytes,
        num_limbs,
    }
}

// ── Modular arithmetic IR ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct ModArithKind;

impl FieldKind for ModArithKind {
    fn c_prefix() -> &'static str {
        "mod"
    }

    fn known_suffix(field: KnownField) -> Option<&'static str> {
        Some(field.c_suffix())
    }

    fn inline_record_size(num_limbs: u32) -> Option<usize> {
        Some(VecHeapRecordDescriptor::new(num_limbs as usize).record_size)
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

impl ArithKind for ModArithKind {
    fn opname() -> &'static str {
        "mod_arith"
    }
}

impl SetupKind for ModArithKind {
    fn opname() -> &'static str {
        "mod_setup"
    }
}

impl IsEqKind for ModArithKind {
    fn opname() -> &'static str {
        "mod_iseq"
    }

    fn inline_record_size(num_limbs: u32) -> Option<usize> {
        Some(ModIsEqRecordDescriptor::new(num_limbs as usize).record_size)
    }

    fn emit_inline_record(
        ctx: &mut dyn ExtEmitCtx,
        from_pc: u32,
        local_opcode: u8,
        num_limbs: u32,
        chip_idx: Option<AirIndex>,
    ) {
        emit_mod_iseq_record(ctx, from_pc, local_opcode, num_limbs, chip_idx);
    }
}

/// IR node for modular arithmetic (ADD, SUB, MUL, DIV).
pub(crate) type ModArithInstr = FieldArithInstr<ModArithKind>;

/// IR node for modular IS_EQ.
pub(crate) type ModIsEqInstr = FieldIsEqInstr<ModArithKind>;

/// IR node for modular SETUP (SETUP_ADDSUB and SETUP_MULDIV).
pub(crate) type ModSetupInstr = FieldSetupInstr<ModArithKind>;

#[derive(Debug, Clone)]
struct ModSetupIsEqInstr {
    from_pc: u32,
    local_opcode: u8,
    chip_idx: Option<AirIndex>,
    emit_inline: bool,
    rd_reg: Variable,
    rs1_reg: Variable,
    rs2_reg: Variable,
    num_limbs: u32,
    modulus: Vec<u8>,
}

impl ExtInstr for ModSetupIsEqInstr {
    fn opname(&self) -> &str {
        "mod_setup_iseq"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rs1 = ctx.read_var(self.rs1_reg);
        let rs2 = ctx.read_var(self.rs2_reg);
        let mod_literal = format_c_byte_array(&self.modulus);
        let num_limbs = format!("{}u", self.num_limbs);
        ctx.write_line("{");
        ctx.write_line(&format!("static constexpr uint8_t mod_[] = {mod_literal};"));
        let result = ctx.emit_call_expr(
            "uint8_t",
            "rvr_ext_mod_setup_iseq",
            &["state", &rs1, &rs2, &num_limbs, "mod_"],
        );
        ctx.write_line(&format!("if (unlikely({result} > 1u)) {{"));
        ctx.emit_trap();
        ctx.write_line("}");
        ctx.write_var(self.rd_reg, &result);
        ctx.write_line("}");
        if self.emit_inline && ctx.inline_record_enabled() {
            emit_mod_iseq_record(
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
            record_size: ModIsEqRecordDescriptor::new(self.num_limbs as usize).record_size,
        })
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::WriteUnknown { dst: self.rd_reg }
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
            &format!("{num_limbs}u"),
            "2u",
            &format!("{}u", air_index_to_c(chip_idx)),
        ],
    );
}

fn emit_mod_iseq_record(
    ctx: &mut dyn ExtEmitCtx,
    from_pc: u32,
    local_opcode: u8,
    num_limbs: u32,
    chip_idx: Option<AirIndex>,
) {
    ctx.extern_call(
        "rvr_ext_emit_mod_iseq_record",
        &[
            "state",
            &format!("{from_pc}u"),
            &format!("{local_opcode}u"),
            &format!("{num_limbs}u"),
            &format!("{}u", air_index_to_c(chip_idx)),
        ],
    );
}

// ── Phantom instructions (HintNonQr, HintSqrt) ──────────────────────────────

/// IR node for HintNonQr phantom instruction.
#[derive(Debug, Clone)]
pub struct HintNonQrInstr {
    pub non_qr_bytes: Vec<u8>,
    pub operands: [u32; 3],
}

impl ExtInstr for HintNonQrInstr {
    fn opname(&self) -> &str {
        "hint_nonqr"
    }

    fn accesses_memory(&self) -> bool {
        false
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let literal = format_c_byte_array(&self.non_qr_bytes);
        ctx.write_line("{");
        ctx.write_line(&format!("static constexpr uint8_t nqr[] = {literal};"));
        let len = format!("{}u", self.non_qr_bytes.len());
        ctx.emit_call_without_page_flush("ext_hint_stream_set", &["nqr", &len]);
        ctx.write_line("}");
        ctx.trace_phantom_record(self.operands);
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        Some(InlineRecordShape::Custom { record_size: 20 })
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }
}

/// IR node for HintSqrt phantom instruction.
#[derive(Debug, Clone)]
pub struct HintSqrtInstr {
    pub rs1_reg: Variable,
    pub num_limbs: u32,
    pub modulus: Vec<u8>,
    pub non_qr_bytes: Vec<u8>,
    pub operands: [u32; 3],
}

impl ExtInstr for HintSqrtInstr {
    fn opname(&self) -> &str {
        "hint_sqrt"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        // HintSqrt is a phantom: its `rs1` pointer read must NOT be traced. The
        // reference executor reads it via untraced `GuestMemory` and advances the
        // clock by a single `increment_timestamp()`, so no memory-bus interaction
        // exists for the PhantomAir to consume. Using the tracing `read_reg` here
        // logged an orphan register access, breaking MemoryBus balance (the read
        // sat unconsumed between two AddSub writes to the same register). Mirror
        // the sibling phantoms (PrintStr, HintRandom): untraced read + a bare tick.
        let rs1 = ctx.read_var_raw(self.rs1_reg);
        let mod_literal = format_c_byte_array(&self.modulus);
        let nqr_literal = format_c_byte_array(&self.non_qr_bytes);
        ctx.write_line("{");
        ctx.write_line(&format!("static constexpr uint8_t mod_[] = {mod_literal};"));
        ctx.write_line(&format!("static constexpr uint8_t nqr[] = {nqr_literal};"));
        let num_limbs = format!("{}u", self.num_limbs);
        ctx.emit_call(
            "rvr_ext_algebra_hint_sqrt",
            &["state", &rs1, &num_limbs, "mod_", "nqr"],
        );
        ctx.write_line("}");
        ctx.trace_phantom_record(self.operands);
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        Some(InlineRecordShape::Custom { record_size: 20 })
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }
}

// ── Modular extension ────────────────────────────────────────────────────────

/// Modular arithmetic and phantom hints.
pub struct ModularRvrExtension {
    moduli: Vec<ModulusInfo>,
    air_indices: Vec<[Option<AirIndex>; MODULAR_OPCODE_COUNT]>,
}

impl ModularRvrExtension {
    pub fn new(
        moduli: Vec<BigUint>,
        ctx: Option<&RvrExtensionCtx>,
    ) -> Result<Self, ExtensionError> {
        let moduli = make_moduli(moduli);
        let mut air_indices = Vec::with_capacity(moduli.len());
        for mod_idx in 0..moduli.len() {
            let mut indices = [None; MODULAR_OPCODE_COUNT];
            for (local, index) in indices.iter_mut().enumerate() {
                let opcode = VmOpcode::from_usize(
                    Rv64ModularArithmeticOpcode::CLASS_OFFSET
                        + mod_idx * MODULAR_OPCODE_COUNT
                        + local,
                );
                *index = resolve_air_index(ctx, opcode)?;
            }
            air_indices.push(indices);
        }
        Ok(Self {
            moduli,
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

impl RvrExtension for ModularRvrExtension {
    fn codegen_fingerprint(&self) -> Option<Vec<u8>> {
        let mut fingerprint = b"openvm-modular-rvr-v3\0".to_vec();
        fingerprint.extend_from_slice(&(self.moduli.len() as u64).to_le_bytes());
        for modulus in &self.moduli {
            fingerprint.extend_from_slice(&modulus.num_limbs.to_le_bytes());
            fingerprint.extend_from_slice(&(modulus.modulus_bytes.len() as u64).to_le_bytes());
            fingerprint.extend_from_slice(&modulus.modulus_bytes);
            fingerprint.extend_from_slice(&(modulus.non_qr_bytes.len() as u64).to_le_bytes());
            fingerprint.extend_from_slice(&modulus.non_qr_bytes);
        }
        let indices = self
            .air_indices
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        fingerprint.extend_from_slice(&air_index_codegen_fingerprint(
            b"openvm-modular-air-indices-v1",
            &indices,
        ));
        Some(fingerprint)
    }

    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if let Some(lifted) = self.try_lift_modular(insn, pc, opcode) {
            return Some(lifted);
        }

        if opcode == SystemOpcode::PHANTOM.global_opcode_usize() {
            if let Some(lifted) = self.try_lift_phantom(insn, pc) {
                return Some(lifted);
            }
        }

        None
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "rvr_ext_bls12_381.h",
                include_str!("../c/rvr_ext_bls12_381.h"),
            ),
            (
                "rvr_ext_vec_heap_record.h",
                include_str!("../c/rvr_ext_vec_heap_record.h"),
            ),
            ("rvr_ext_mod.h", include_str!("../c/rvr_ext_mod.h")),
        ]
    }

    fn c_sources(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "rvr_ext_modular.c",
                include_str!("../ffi/modular/c/rvr_ext_modular.c"),
            ),
            (
                "rvr_ext_bls12_381.c",
                include_str!("../ffi/modular/c/rvr_ext_bls12_381.c"),
            ),
        ]
    }

    fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        vec![
            (
                "librvr_openvm_ext_algebra_modular_ffi.a",
                include_bytes!(env!("RVR_ALGEBRA_MODULAR_FFI_STATICLIB")),
            ),
            (
                "libblst.a",
                include_bytes!(env!("RVR_ALGEBRA_BLST_STATICLIB")),
            ),
        ]
    }

    fn uses_memory_wrappers(&self) -> bool {
        true
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        MODULAR_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION
    }

    fn vendored_c_sources(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "precomputed_ecmult.c",
                include_str!("../ffi/modular/secp256k1/src/precomputed_ecmult.c"),
            ),
            (
                "precomputed_ecmult_gen.c",
                include_str!("../ffi/modular/secp256k1/src/precomputed_ecmult_gen.c"),
            ),
        ]
    }

    fn extra_c_include_files(&self) -> Vec<(&'static str, &'static str)> {
        let mut files = SECP256K1_C_FILES.to_vec();
        files.extend([
            (
                "blst/blst.h",
                include_str!("../ffi/modular/blst/bindings/blst.h"),
            ),
            (
                "blst/blst_aux.h",
                include_str!("../ffi/modular/blst/bindings/blst_aux.h"),
            ),
        ]);
        files
    }

    fn extra_cflags(&self) -> Vec<String> {
        vec![
            "-isystem".to_string(),
            "secp256k1/src".to_string(),
            "-isystem".to_string(),
            "secp256k1".to_string(),
            "-isystem".to_string(),
            "blst".to_string(),
            // ENABLE_MODULE_RECOVERY keeps the ECC modules compiled in so the
            // k256 EC ops in rvr_ext_modular.c can call into libsecp256k1.
            // (-DSECP256K1_BUILD is not set here — secp256k1.c defines it
            // internally.)
            "-DENABLE_MODULE_RECOVERY".to_string(),
        ]
    }
}

impl ModularRvrExtension {
    fn try_lift_modular(
        &self,
        insn: &RvrInstruction,
        pc: u64,
        opcode: usize,
    ) -> Option<LiftedInstr> {
        let base_offset = Rv64ModularArithmeticOpcode::CLASS_OFFSET;
        let count = Rv64ModularArithmeticOpcode::COUNT;

        if opcode < base_offset {
            return None;
        }
        let relative = opcode - base_offset;
        let mod_idx = relative / count;
        let local = relative % count;

        if mod_idx >= self.moduli.len() {
            return None;
        }

        let info = &self.moduli[mod_idx];
        let chip_idx = self.air_indices[mod_idx][local];
        let from_pc = pc as u32;
        let local_opcode = local as u8;
        let vec_heap_inline = insn.d == RV64_REGISTER_AS && insn.e == RV64_MEMORY_AS;
        let is_eq_inline = vec_heap_inline && insn.a != 0;
        let rd_reg = decode_reg(insn.a);
        let rs1_reg = decode_reg(insn.b);
        let rs2_reg = decode_reg(insn.c);

        let instr: Box<dyn ExtInstr> = match local {
            x if x == Rv64ModularArithmeticOpcode::ADD as usize => Box::new(
                ModArithInstr::new(
                    ModOp::Add,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    info.num_limbs,
                    info.modulus_bytes.clone(),
                )
                .with_inline_record(from_pc, local_opcode, chip_idx, vec_heap_inline),
            ),
            x if x == Rv64ModularArithmeticOpcode::SUB as usize => Box::new(
                ModArithInstr::new(
                    ModOp::Sub,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    info.num_limbs,
                    info.modulus_bytes.clone(),
                )
                .with_inline_record(from_pc, local_opcode, chip_idx, vec_heap_inline),
            ),
            x if x == Rv64ModularArithmeticOpcode::SETUP_ADDSUB as usize => Box::new(
                ModSetupInstr::new(
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    info.num_limbs,
                    info.modulus_bytes.clone(),
                )
                .with_inline_record(from_pc, local_opcode, chip_idx, vec_heap_inline),
            ),
            x if x == Rv64ModularArithmeticOpcode::MUL as usize => Box::new(
                ModArithInstr::new(
                    ModOp::Mul,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    info.num_limbs,
                    info.modulus_bytes.clone(),
                )
                .with_inline_record(from_pc, local_opcode, chip_idx, vec_heap_inline),
            ),
            x if x == Rv64ModularArithmeticOpcode::DIV as usize => Box::new(
                ModArithInstr::new(
                    ModOp::Div,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    info.num_limbs,
                    info.modulus_bytes.clone(),
                )
                .with_inline_record(from_pc, local_opcode, chip_idx, vec_heap_inline),
            ),
            x if x == Rv64ModularArithmeticOpcode::SETUP_MULDIV as usize => Box::new(
                ModSetupInstr::new(
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    info.num_limbs,
                    info.modulus_bytes.clone(),
                )
                .with_inline_record(from_pc, local_opcode, chip_idx, vec_heap_inline),
            ),
            x if x == Rv64ModularArithmeticOpcode::IS_EQ as usize => Box::new(
                ModIsEqInstr::new(
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    info.num_limbs,
                    info.modulus_bytes.clone(),
                )
                .with_inline_record(from_pc, local_opcode, chip_idx, is_eq_inline),
            ),
            x if x == Rv64ModularArithmeticOpcode::SETUP_ISEQ as usize => {
                Box::new(ModSetupIsEqInstr {
                    from_pc,
                    local_opcode,
                    chip_idx,
                    emit_inline: is_eq_inline,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                    modulus: info.modulus_bytes.clone(),
                })
            }
            _ => return None,
        };

        Some(LiftedInstr::Body(InstrAt {
            pc,
            instr,
            source_loc: None,
        }))
    }

    fn try_lift_phantom(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        let c_val = insn.c;
        let discriminant = (c_val & 0xffff) as u16;
        let mod_idx = (c_val >> 16) as usize;
        let operands = [insn.a, insn.b, insn.c];

        match ModularPhantom::from_repr(discriminant) {
            Some(ModularPhantom::HintNonQr) => {
                let info = self.moduli.get(mod_idx)?;
                Some(LiftedInstr::Body(InstrAt {
                    pc,
                    instr: Box::new(HintNonQrInstr {
                        non_qr_bytes: info.non_qr_bytes.clone(),
                        operands,
                    }),
                    source_loc: None,
                }))
            }
            Some(ModularPhantom::HintSqrt) => {
                let info = self.moduli.get(mod_idx)?;
                let rs1_reg = decode_reg(insn.a);
                Some(LiftedInstr::Body(InstrAt {
                    pc,
                    instr: Box::new(HintSqrtInstr {
                        rs1_reg,
                        num_limbs: info.num_limbs,
                        modulus: info.modulus_bytes.clone(),
                        non_qr_bytes: info.non_qr_bytes.clone(),
                        operands,
                    }),
                    source_loc: None,
                }))
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use rvr_openvm_ir::{MemWidth, PageAddressSpace};

    use super::*;

    /// Minimal capturing [`ExtEmitCtx`]. A *traced* register read (`read_reg`)
    /// records a `trace_reg_read(...)` line — in generated C this emits a
    /// MemoryBus event; an *untraced* read (`read_reg_raw`) records nothing.
    /// Tests can therefore assert whether a phantom logged a bus-visible
    /// register access.
    #[derive(Default)]
    struct TestEmitCtx {
        lines: Vec<String>,
    }

    impl ExtEmitCtx for TestEmitCtx {
        fn read_var(&mut self, var: Variable) -> String {
            self.lines
                .push(format!("trace_reg_read(state, {});", var.index()));
            format!("r{}", var.index())
        }
        fn peek_var(&mut self, var: Variable) -> String {
            format!("r{}", var.index())
        }

        fn read_var_raw(&mut self, var: Variable) -> String {
            format!("r{}", var.index())
        }

        fn write_var(&mut self, _var: Variable, _val: &str) {}

        fn write_line(&mut self, s: &str) {
            self.lines.push(s.to_string());
        }

        fn emit_trap(&mut self) {
            self.write_line("trap;");
        }

        fn read_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String {
            let tmp = format!("tmp{}", self.lines.len());
            self.write_line(&format!(
                "uint32_t {tmp} = read_mem({base}, {offset}, {width}, {signed});"
            ));
            tmp
        }
        fn write_mem(&mut self, base: &str, offset: i16, val: &str, width: u8) {
            self.write_line(&format!("write_mem({base}, {offset}, {val}, {width});"));
        }

        fn emit_call(&mut self, name: &str, args: &[&str]) {
            self.write_line(&format!("{name}({});", args.join(", ")));
        }

        fn emit_call_without_page_flush(&mut self, name: &str, args: &[&str]) {
            self.emit_call(name, args);
        }

        fn emit_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String {
            let tmp = format!("tmp{}", self.lines.len());
            self.write_line(&format!("{ret_ty} {tmp} = {name}({});", args.join(", ")));
            tmp
        }

        fn emit_call_with_trace_result(
            &mut self,
            ret_ty: &str,
            name: &str,
            args: &[&str],
        ) -> Option<String> {
            Some(self.emit_call_expr(ret_ty, name, args))
        }

        fn trace_chip(&mut self, chip_idx: u32, count_expr: &str) {
            self.write_line(&format!("trace_chip(state, {chip_idx}u, {count_expr});"));
        }

        fn trace_chip_if_nonzero(&mut self, chip_idx: u32, count_expr: &str) {
            self.trace_chip(chip_idx, count_expr);
        }

        fn trace_page_access(
            &mut self,
            addr: &str,
            width: MemWidth,
            addr_space: PageAddressSpace,
        ) {
            self.write_line(&format!(
                "trace_page_access(state, {addr}, {}u, {}u);",
                width.bytes(),
                addr_space.id()
            ));
        }

        fn trace_page_access_u64_range(
            &mut self,
            base_addr: &str,
            num_dwords: &str,
            addr_space: PageAddressSpace,
        ) {
            self.write_line(&format!(
                "trace_page_access_u64_range(state, {base_addr}, {num_dwords}, {}u);",
                addr_space.id()
            ));
        }

        fn trace_mem_access_u64_range(
            &mut self,
            base_addr: &str,
            num_dwords: &str,
            addr_space: PageAddressSpace,
        ) {
            self.write_line(&format!(
                "trace_mem_access_u64_range(state, {base_addr}, {num_dwords}, {}u);",
                addr_space.id()
            ));
        }
        fn trace_wr_as_u64(&mut self, addr: &str, val: &str, addr_space: u32) {
            self.write_line(&format!(
                "trace_wr_as_u64(state, {addr}, {val}, {addr_space}u);"
            ));
        }
        fn trace_timestamp(&mut self) {
            self.write_line("trace_timestamp(state);");
        }
    }

    /// Regression guard for bug #2 (reth CPU seg-28 MemoryBus imbalance).
    ///
    /// `HintSqrt` is a phantom: the reference executor reads its pointer register
    /// through untraced `GuestMemory` and advances the clock with a single
    /// `increment_timestamp()`, so no MemoryBus interaction exists for the
    /// memory-unconstrained `PhantomAir` to consume. Emitting a *traced* read
    /// (`read_reg`) here logged an orphan register access that sat unconsumed
    /// between two AddSub writes to the same register, leaving bus 1 one-sided
    /// (+1 send at T, -1 receive at T+gap). The phantom must instead read
    /// untraced and emit one bare timestamp tick, exactly like every other
    /// phantom (PrintStr, HintRandom, HintFinalExp).
    #[test]
    fn hint_sqrt_phantom_reads_pointer_untraced_and_ticks_once() {
        let mut ctx = TestEmitCtx::default();
        HintSqrtInstr {
            rs1_reg: Variable::new(10),
            num_limbs: 4,
            modulus: vec![0u8; 32],
            non_qr_bytes: vec![0u8; 32],
            operands: [80, 0, ModularPhantom::HintSqrt as u32],
        }
        .emit_c(&mut ctx);

        // A phantom must emit no traced (bus-visible) register access.
        assert!(
            !ctx.lines.iter().any(|l| l.contains("trace_reg_read")),
            "HintSqrt emitted a traced register read (orphan MemoryBus access); \
             phantoms must use read_reg_raw. lines: {:#?}",
            ctx.lines
        );
        // The pointer still reaches the host callback via the untraced value r10.
        assert!(
            ctx.lines
                .iter()
                .any(|l| l.contains("rvr_ext_algebra_hint_sqrt(state, r10,")),
            "expected untraced pointer r10 passed to hint_sqrt callback. lines: {:#?}",
            ctx.lines
        );
        // The phantom still advances the clock by exactly one bare tick, matching
        // the reference `increment_timestamp()` so downstream timestamps are
        // byte-identical.
        assert_eq!(
            ctx.lines
                .iter()
                .filter(|l| l.as_str() == "trace_timestamp(state);")
                .count(),
            1,
            "HintSqrt must emit exactly one bare timestamp tick. lines: {:#?}",
            ctx.lines
        );
    }
}
