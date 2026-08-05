//! Deferral extension for rvr-openvm: IR nodes for CALL/OUTPUT and the
//! `DeferralRvrExtension` for lifting them.
#![cfg(feature = "rvr")]

use std::{
    ffi::c_void,
    mem::{align_of, size_of},
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

use openvm_circuit::arch::{
    deferral::{DeferralFn, InputMapVal},
    rvr::io::OpenVmIoState,
};
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    riscv::{RV64_NUM_REGISTERS, RV64_REGISTER_BYTES},
    LocalOpcode, VM_DIGEST_WIDTH,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{
    CfgEffect, ExtEmitCtx, ExtInstr, FixedTraceRows, InstrAt, LiftedInstr, Variable,
};
use rvr_openvm_lift::{
    air_index_to_c, decode_variable, fixed_trace_rows_for_chip,
    max_main_memory_pages_for_contiguous_range, opcode_air_idx, AirIndex, ExtensionError,
    RvrExtension, RvrExtensionCtx, RvrInstruction, RvrRuntimeExtension,
};

fn decode_reg(value: u32) -> Variable {
    decode_variable(value, RV64_REGISTER_BYTES as u32, RV64_NUM_REGISTERS as u32)
}

/// Size in bytes of a serialized deferral commitment.
pub const DEFERRAL_COMMIT_NUM_BYTES: usize = VM_DIGEST_WIDTH * core::mem::size_of::<u32>();
/// Size in bytes of a deferral output key: commitment followed by output length.
pub const DEFERRAL_OUTPUT_KEY_BYTES: usize =
    DEFERRAL_COMMIT_NUM_BYTES + core::mem::size_of::<u64>();
const DEFERRAL_OUTPUT_KEY_WORDS: usize = DEFERRAL_OUTPUT_KEY_BYTES / size_of::<u64>();
/// OUTPUT writes one sponge-rate row per guest word (`DIGEST_SIZE = 8` bytes).
const DEFERRAL_OUTPUT_ROW_BYTES: usize = size_of::<u64>();
const DEFERRAL_ACCUMULATOR_WORDS: usize = 2 * VM_DIGEST_WIDTH * size_of::<u32>() / size_of::<u64>();
const DEFERRAL_CALL_REPLAY_WORDS: usize = DEFERRAL_OUTPUT_KEY_WORDS + DEFERRAL_ACCUMULATOR_WORDS;
const DEFERRAL_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION: usize =
    max_main_memory_pages_for_contiguous_range(DEFERRAL_COMMIT_NUM_BYTES)
        + max_main_memory_pages_for_contiguous_range(DEFERRAL_OUTPUT_KEY_BYTES);

/// `(def_idx, output_raw) → output_commit` hasher registered by the host.
pub type DeferralHashFn = Box<dyn Fn(u32, &[u8]) -> [u8; DEFERRAL_COMMIT_NUM_BYTES] + Send + Sync>;

/// Poseidon2 compression over deferral accumulator field elements.
/// Values cross the crate boundary as canonical u32s.
pub type DeferralCompressFn = Box<
    dyn Fn([u32; VM_DIGEST_WIDTH], [u32; VM_DIGEST_WIDTH]) -> [u32; VM_DIGEST_WIDTH] + Send + Sync,
>;

pub struct DeferralCtx {
    pub fns: Vec<Arc<DeferralFn>>,
    pub hash: DeferralHashFn,
    pub compress: DeferralCompressFn,
}

impl DeferralCtx {
    pub fn new(
        fns: Vec<Arc<DeferralFn>>,
        hash: DeferralHashFn,
        compress: DeferralCompressFn,
    ) -> Self {
        Self {
            fns,
            hash,
            compress,
        }
    }
}

// ── IR Nodes ──────────────────────────────────────────────────────────────────

/// IR node for a deferral CALL instruction.
#[derive(Debug, Clone)]
pub struct DeferralCallInstr {
    pub rd_reg: Variable,
    pub rs_reg: Variable,
    pub def_idx: u32,
    pub poseidon2_chip_idx: Option<AirIndex>,
}

impl ExtInstr for DeferralCallInstr {
    fn opname(&self) -> &str {
        "def_call"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_var(self.rd_reg);
        let rs = ctx.read_var(self.rs_reg);
        let def_idx = format!("{}u", self.def_idx);
        let is_preflight = ctx.is_preflight();
        if is_preflight {
            // The opaque call performs four input-key reads, five output-key
            // writes, and two reads plus two writes of two-block AS4 digests.
            // Register reads are emitted above, for 2 + 17 = 19 total slots.
            ctx.reserve_preflight_timestamp_slots("17u");
            ctx.reserve_replay_values(&format!("{DEFERRAL_CALL_REPLAY_WORDS}u"));
            ctx.write_line(&format!(
                "uint64_t deferral_replay[{DEFERRAL_CALL_REPLAY_WORDS}u];"
            ));
        }
        let replay_out = if is_preflight {
            "deferral_replay"
        } else {
            "NULL"
        };
        ctx.emit_checked_call(
            "rvr_ext_deferral_call",
            &["state", &rd, &rs, &def_idx, replay_out],
        );
        if is_preflight {
            ctx.write_line(&format!(
                "for (uint32_t deferral_replay_idx = 0u; deferral_replay_idx < {DEFERRAL_CALL_REPLAY_WORDS}u; ++deferral_replay_idx) {{"
            ));
            ctx.append_replay_value("deferral_replay[deferral_replay_idx]");
            ctx.write_line("}");
        } else {
            ctx.count_fixed_replay_values(
                DEFERRAL_CALL_REPLAY_WORDS
                    .try_into()
                    .expect("deferral replay word count fits in u32"),
            );
        }
    }

    fn fixed_trace_rows(&self) -> Vec<FixedTraceRows> {
        fixed_trace_rows_for_chip(self.poseidon2_chip_idx, 2)
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }

    fn supports_preflight(&self) -> bool {
        true
    }
}

/// IR node for a deferral OUTPUT instruction.
#[derive(Debug, Clone)]
pub struct DeferralOutputInstr {
    pub rd_reg: Variable,
    pub rs_reg: Variable,
    pub def_idx: u32,
    pub output_chip_idx: Option<AirIndex>,
    pub poseidon2_chip_idx: Option<AirIndex>,
}

impl ExtInstr for DeferralOutputInstr {
    fn opname(&self) -> &str {
        "def_output"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_var(self.rd_reg);
        let rs = ctx.read_var(self.rs_reg);
        let is_preflight = ctx.is_preflight();
        let output_words = "deferral_output_words";
        if is_preflight {
            ctx.write_line(&format!(
                "if (unlikely({rs} > OPENVM_MEM_SIZE - {DEFERRAL_OUTPUT_KEY_BYTES}u)) {{"
            ));
            ctx.emit_trap();
            ctx.write_line("}");
            ctx.write_line(&format!(
                "uint64_t deferral_output_len_u64 = peek_mem_u64(state, {rs} + {DEFERRAL_COMMIT_NUM_BYTES}ull);"
            ));
            ctx.write_line("if (unlikely(deferral_output_len_u64 > UINT32_MAX)) {");
            ctx.emit_trap();
            ctx.write_line("}");
            ctx.write_line("uint32_t deferral_output_len = (uint32_t)deferral_output_len_u64;");
            // Output rows are one guest word. Reject a partial row before
            // reserving timestamp slots or mutating guest memory.
            ctx.write_line(&format!(
                "if (unlikely((deferral_output_len & {}u) != 0u)) {{",
                DEFERRAL_OUTPUT_ROW_BYTES - 1
            ));
            ctx.emit_trap();
            ctx.write_line("}");
            ctx.write_line(&format!(
                "uint32_t {output_words} = deferral_output_len / {}u;",
                size_of::<u64>()
            ));
            ctx.reserve_preflight_timestamp_slots(&format!("5u + {output_words}"));
            ctx.reserve_replay_values(&format!("1u + {output_words}"));
        }
        let output = air_index_to_c(self.output_chip_idx);
        let poseidon2 = air_index_to_c(self.poseidon2_chip_idx);
        let def_idx = format!("{}u", self.def_idx);
        ctx.write_line("uint32_t deferral_num_rows;");
        ctx.emit_checked_call(
            "rvr_ext_deferral_output",
            &["state", &rd, &rs, &def_idx, "&deferral_num_rows"],
        );
        ctx.trace_chip_if_nonzero(output, "deferral_num_rows - 1u");
        ctx.trace_chip(poseidon2, "deferral_num_rows");
        if is_preflight {
            ctx.write_line(&format!(
                "for (uint32_t deferral_replay_idx = 0u; deferral_replay_idx <= {output_words}; ++deferral_replay_idx) {{"
            ));
            ctx.append_replay_value(&format!(
                "deferral_replay_idx == 0u ? (uint64_t){output_words} : peek_mem_u64(state, {rd} + (uint64_t)(deferral_replay_idx - 1u) * 8ull)"
            ));
            ctx.write_line("}");
        } else {
            // Each non-header Deferral row contains four guest u64 words.
            // Count only after the checked host call has succeeded.
            ctx.count_replay_values("1ull + 4ull * ((uint64_t)deferral_num_rows - 1ull)");
        }
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }

    fn supports_preflight(&self) -> bool {
        true
    }
}

// ── Extension ─────────────────────────────────────────────────────────────────

/// The Deferral extension (CALL + OUTPUT opcodes).
pub struct DeferralRvrExtension {
    output_chip_idx: Option<AirIndex>,
    poseidon2_chip_idx: Option<AirIndex>,
}

impl DeferralRvrExtension {
    pub fn new(ctx: Option<&RvrExtensionCtx>) -> Result<Self, ExtensionError> {
        let call_chip_idx = opcode_air_idx(ctx, DeferralOpcode::CALL)?;
        let output_chip_idx = opcode_air_idx(ctx, DeferralOpcode::OUTPUT)?;
        // The Poseidon2 hasher is registered adjacent to the CALL chip and
        // assigned the next AIR index (call_air_idx + 1) due to reverse registration order.
        let poseidon2_chip_idx = call_chip_idx.map(AirIndex::next);

        Ok(Self {
            output_chip_idx,
            poseidon2_chip_idx,
        })
    }
}

impl RvrExtension for DeferralRvrExtension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode == DeferralOpcode::CALL.global_opcode_usize() {
            let rd_reg = decode_reg(insn.a);
            let rs_reg = decode_reg(insn.b);
            let def_idx = insn.c;
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Box::new(DeferralCallInstr {
                    rd_reg,
                    rs_reg,
                    def_idx,
                    poseidon2_chip_idx: self.poseidon2_chip_idx,
                }),
                source_loc: None,
            }));
        }

        if opcode == DeferralOpcode::OUTPUT.global_opcode_usize() {
            let rd_reg = decode_reg(insn.a);
            let rs_reg = decode_reg(insn.b);
            let def_idx = insn.c;
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Box::new(DeferralOutputInstr {
                    rd_reg,
                    rs_reg,
                    def_idx,
                    output_chip_idx: self.output_chip_idx,
                    poseidon2_chip_idx: self.poseidon2_chip_idx,
                }),
                source_loc: None,
            }));
        }

        None
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![(
            "rvr_ext_deferral.h",
            include_str!("../c/rvr_ext_deferral.h"),
        )]
    }

    fn c_sources(&self) -> Vec<(&'static str, &'static str)> {
        vec![(
            "rvr_ext_deferral.c",
            include_str!("../c/rvr_ext_deferral.c"),
        )]
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        DEFERRAL_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION
    }

    fn uses_deferral_address_space(&self) -> bool {
        true
    }
}

type DeferralCallLookupFn =
    unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *const u8, *mut u8, *mut u64) -> bool;
type DeferralOutputLookupFn =
    unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *const u8, *mut u8, u32) -> bool;

pub struct DeferralRuntimeHooks {
    deferral_ctx: DeferralCtx,
    call_lookup: DeferralCallLookupFn,
}

impl DeferralRuntimeHooks {
    /// # Safety
    ///
    /// `F` must be the field used by the VM state whose deferral memory is
    /// passed to these callbacks.
    pub unsafe fn new<F: PrimeField32>(
        fns: Vec<Arc<DeferralFn>>,
        hash: DeferralHashFn,
        compress: DeferralCompressFn,
    ) -> Self {
        Self {
            deferral_ctx: DeferralCtx::new(fns, hash, compress),
            call_lookup: host_deferral_call_lookup::<F>,
        }
    }
}

impl RvrRuntimeExtension for DeferralRuntimeHooks {
    unsafe fn register_host_callbacks(
        &self,
        lib: &libloading::Library,
    ) -> Result<(), ExtensionError> {
        let register_fn: RegisterFn = unsafe {
            let sym = lib
                .get::<RegisterFn>(b"register_deferral_callbacks")
                .map_err(|e| ExtensionError::HostCallbackRegistration(e.to_string()))?;
            *sym
        };
        // `ctx` aliases `self.deferral_ctx`; the C side must not outlive `self`.
        let callbacks = DeferralHostCallbacks {
            ctx: &self.deferral_ctx as *const DeferralCtx as *mut c_void,
            call_lookup: self.call_lookup,
            output_lookup: host_deferral_output_lookup,
        };
        unsafe { register_fn(&callbacks) };
        Ok(())
    }
}

// ── Deferral accumulator sync (DEFERRAL_AS) ────────────────────────────────
//
// CALL writes new `(input_acc, output_acc)` values to DEFERRAL_AS.

fn commit_bytes_to_field_values(bytes: &[u8; DEFERRAL_COMMIT_NUM_BYTES]) -> [u32; VM_DIGEST_WIDTH] {
    let mut out = [0u32; VM_DIGEST_WIDTH];
    for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = u32::from_le_bytes(chunk.try_into().unwrap());
    }
    out
}

/// # Safety
/// `io.deferral_memory` must point to a live DEFERRAL_AS buffer containing
/// initialized native `F` values and have exclusive access for the returned
/// slice's lifetime.
unsafe fn deferral_memory<'a, F: PrimeField32>(
    io: &'a mut OpenVmIoState<'_>,
) -> Option<&'a mut [F]> {
    if !io.deferral_memory_len_bytes.is_multiple_of(size_of::<F>()) {
        return None;
    }
    if io.deferral_memory_len_bytes == 0 {
        return Some(&mut []);
    }
    if io.deferral_memory.is_null() || !io.deferral_memory.addr().is_multiple_of(align_of::<F>()) {
        return None;
    }
    Some(unsafe {
        std::slice::from_raw_parts_mut(
            io.deferral_memory.cast(),
            io.deferral_memory_len_bytes / size_of::<F>(),
        )
    })
}

fn read_deferral_digest<F: PrimeField32>(
    memory: &[F],
    ptr: usize,
) -> Option<[u32; VM_DIGEST_WIDTH]> {
    let end = ptr.checked_add(VM_DIGEST_WIDTH)?;
    let values = memory.get(ptr..end)?;
    Some(std::array::from_fn(|i| values[i].as_canonical_u32()))
}

struct DeferralAccumulatorUpdate<F> {
    memory: *mut F,
    input_acc_ptr: usize,
    output_acc_ptr: usize,
    new_input_acc: [F; VM_DIGEST_WIDTH],
    new_output_acc: [F; VM_DIGEST_WIDTH],
    packed: [u64; DEFERRAL_ACCUMULATOR_WORDS],
}

/// Validates and computes one accumulator update without mutating VM state.
unsafe fn prepare_deferral_accumulator_update<F: PrimeField32>(
    deferral_ctx: &DeferralCtx,
    io: &mut OpenVmIoState<'_>,
    def_idx: u32,
    input_commit: &[u8; DEFERRAL_COMMIT_NUM_BYTES],
    output_commit: &[u8; DEFERRAL_COMMIT_NUM_BYTES],
) -> Option<DeferralAccumulatorUpdate<F>> {
    let input_acc_ptr = 2usize
        .checked_mul(def_idx as usize)
        .and_then(|value| value.checked_mul(VM_DIGEST_WIDTH))?;
    let output_acc_ptr = input_acc_ptr.checked_add(VM_DIGEST_WIDTH)?;
    let end = output_acc_ptr.checked_add(VM_DIGEST_WIDTH)?;
    let byte_start = input_acc_ptr.checked_mul(size_of::<F>())?;
    let byte_len = (end - input_acc_ptr).checked_mul(size_of::<F>())?;
    if !io.can_mark_preflight_deferral_write(byte_start, byte_len) {
        return None;
    }
    let memory = unsafe { deferral_memory::<F>(io) }?;
    memory.get(input_acc_ptr..end)?;
    let old_input_acc = read_deferral_digest(memory, input_acc_ptr)?;
    let old_output_acc = read_deferral_digest(memory, output_acc_ptr)?;
    let new_input_values =
        (deferral_ctx.compress)(old_input_acc, commit_bytes_to_field_values(input_commit));
    let new_output_values =
        (deferral_ctx.compress)(old_output_acc, commit_bytes_to_field_values(output_commit));
    let new_input_acc = new_input_values.map(F::from_u32);
    let new_output_acc = new_output_values.map(F::from_u32);
    let packed = std::array::from_fn(|word| {
        let value = if 2 * word < VM_DIGEST_WIDTH {
            &new_input_values
        } else {
            &new_output_values
        };
        let index = 2 * word % VM_DIGEST_WIDTH;
        u64::from(value[index]) | (u64::from(value[index + 1]) << 32)
    });
    Some(DeferralAccumulatorUpdate {
        memory: memory.as_mut_ptr(),
        input_acc_ptr,
        output_acc_ptr,
        new_input_acc,
        new_output_acc,
        packed,
    })
}

/// Commits a fully validated accumulator update. No fallible work may follow
/// this call in the host callback.
unsafe fn apply_deferral_accumulator_update<F: PrimeField32>(
    io: &mut OpenVmIoState<'_>,
    update: DeferralAccumulatorUpdate<F>,
) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            update.new_input_acc.as_ptr(),
            update.memory.add(update.input_acc_ptr),
            VM_DIGEST_WIDTH,
        );
        std::ptr::copy_nonoverlapping(
            update.new_output_acc.as_ptr(),
            update.memory.add(update.output_acc_ptr),
            VM_DIGEST_WIDTH,
        );
    }
    io.mark_preflight_deferral_write(
        update.input_acc_ptr * size_of::<F>(),
        2 * VM_DIGEST_WIDTH * size_of::<F>(),
    );
}

// ── Host callbacks ──────────────────────────────────────────────────────────

type RegisterFn = unsafe extern "C" fn(*const DeferralHostCallbacks);

/// Must match the C `DeferralHostCallbacks` layout in `rvr_ext_deferral.h`.
#[repr(C)]
pub struct DeferralHostCallbacks {
    pub ctx: *mut c_void,
    pub call_lookup: DeferralCallLookupFn,
    pub output_lookup: DeferralOutputLookupFn,
}

/// Deferral CALL lookup. Invalid advice or accumulator state returns `false`;
/// generated code converts that result into the ordinary VM trap path.
///
/// # Safety
///
/// `d_ctx` must point to a valid `DeferralCtx`. `io_ctx` must point to a valid
/// `OpenVmIoState` whose deferral memory contains native `F` values.
/// `input_commit_raw` must point to `DEFERRAL_COMMIT_NUM_BYTES` readable bytes.
/// `output_key_out` must point to `DEFERRAL_OUTPUT_KEY_BYTES` writable bytes.
/// `accumulators_out` may be null; otherwise it must point to
/// `DEFERRAL_ACCUMULATOR_WORDS` writable u64s.
pub unsafe extern "C" fn host_deferral_call_lookup<F: PrimeField32>(
    d_ctx: *mut c_void,
    io_ctx: *mut c_void,
    def_idx: u32,
    input_commit_raw: *const u8,
    output_key_out: *mut u8,
    accumulators_out: *mut u64,
) -> bool {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        host_deferral_call_lookup_inner::<F>(
            d_ctx,
            io_ctx,
            def_idx,
            input_commit_raw,
            output_key_out,
            accumulators_out,
        )
    }))
    .unwrap_or(false)
}

unsafe fn host_deferral_call_lookup_inner<F: PrimeField32>(
    d_ctx: *mut c_void,
    io_ctx: *mut c_void,
    def_idx: u32,
    input_commit_raw: *const u8,
    output_key_out: *mut u8,
    accumulators_out: *mut u64,
) -> bool {
    let deferral_ctx = unsafe { &*(d_ctx as *const DeferralCtx) };
    let io = unsafe { &mut *(io_ctx as *mut OpenVmIoState<'_>) };

    let mut input_commit = [0u8; DEFERRAL_COMMIT_NUM_BYTES];
    input_commit.copy_from_slice(unsafe {
        std::slice::from_raw_parts(input_commit_raw, DEFERRAL_COMMIT_NUM_BYTES)
    });
    let input_commit_key = input_commit.to_vec();

    let Some(deferral_state) = io.deferrals.get(def_idx as usize) else {
        return false;
    };
    let Some(input) = deferral_state.try_get_input(&input_commit_key).cloned() else {
        return false;
    };
    let (output_commit, output_len, generated_output) = match input {
        InputMapVal::Output(commit) => {
            let Some(raw) = deferral_state.try_get_output(&commit) else {
                return false;
            };
            let Ok(output_commit) = commit.as_slice().try_into() else {
                return false;
            };
            (output_commit, raw.len() as u64, None)
        }
        InputMapVal::Raw(input_raw) => {
            let Some(deferral_fn) = deferral_ctx.fns.get(def_idx as usize) else {
                return false;
            };
            let output_raw = deferral_fn.call_raw(&input_raw);
            let commit = (deferral_ctx.hash)(def_idx, &output_raw);
            let len = output_raw.len() as u64;
            (commit, len, Some(output_raw))
        }
    };
    if output_len > u32::MAX as u64 {
        return false;
    }

    let Some(update) = (unsafe {
        prepare_deferral_accumulator_update::<F>(
            deferral_ctx,
            io,
            def_idx,
            &input_commit,
            &output_commit,
        )
    }) else {
        return false;
    };
    let packed_accumulators = update.packed;
    if let Some(output_raw) = generated_output {
        let stored = io.deferrals[def_idx as usize].try_store_output(
            &input_commit_key,
            output_commit.to_vec(),
            output_raw,
        );
        debug_assert!(stored);
        if !stored {
            return false;
        }
    }
    unsafe { apply_deferral_accumulator_update(io, update) };

    let mut output_key = [0u8; DEFERRAL_OUTPUT_KEY_BYTES];
    output_key[..DEFERRAL_COMMIT_NUM_BYTES].copy_from_slice(&output_commit);
    output_key[DEFERRAL_COMMIT_NUM_BYTES..].copy_from_slice(&output_len.to_le_bytes());
    unsafe {
        std::ptr::copy_nonoverlapping(
            output_key.as_ptr(),
            output_key_out,
            DEFERRAL_OUTPUT_KEY_BYTES,
        );
        if !accumulators_out.is_null() {
            std::ptr::copy_nonoverlapping(
                packed_accumulators.as_ptr(),
                accumulators_out,
                packed_accumulators.len(),
            );
        }
    }
    true
}

/// Deferral OUTPUT lookup: `deferrals[def_idx].output_map[output_commit]`.
/// Returns `false` on a missing `def_idx`, commitment, or length mismatch.
///
/// # Safety
///
/// `d_ctx` must point to a valid `DeferralCtx`. `io_ctx` must point to a
/// valid `OpenVmIoState`.
/// `output_commit_raw` must point to `DEFERRAL_COMMIT_NUM_BYTES` readable bytes.
/// `output_raw_out` must point to at least `expected_len` writable bytes.
pub unsafe extern "C" fn host_deferral_output_lookup(
    _d_ctx: *mut c_void,
    io_ctx: *mut c_void,
    def_idx: u32,
    output_commit_raw: *const u8,
    output_raw_out: *mut u8,
    expected_len: u32,
) -> bool {
    catch_unwind(AssertUnwindSafe(|| unsafe {
        host_deferral_output_lookup_inner(
            io_ctx,
            def_idx,
            output_commit_raw,
            output_raw_out,
            expected_len,
        )
    }))
    .unwrap_or(false)
}

unsafe fn host_deferral_output_lookup_inner(
    io_ctx: *mut c_void,
    def_idx: u32,
    output_commit_raw: *const u8,
    output_raw_out: *mut u8,
    expected_len: u32,
) -> bool {
    let io = unsafe { &*(io_ctx as *const OpenVmIoState<'_>) };

    let output_commit: Vec<u8> = unsafe {
        std::slice::from_raw_parts(output_commit_raw, DEFERRAL_COMMIT_NUM_BYTES).to_vec()
    };
    let Some(state) = io.deferrals.get(def_idx as usize) else {
        return false;
    };
    let Some(raw) = state.try_get_output(&output_commit) else {
        return false;
    };
    if raw.len() != expected_len as usize {
        return false;
    }
    unsafe { std::ptr::copy_nonoverlapping(raw.as_ptr(), output_raw_out, raw.len()) };
    true
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, mem::size_of_val};

    use openvm_circuit::arch::{deferral::DeferralState, HintStream};
    use p3_baby_bear::BabyBear;
    use rand::{rngs::StdRng, SeedableRng};
    use rvr_openvm_ir::{MemWidth, PageAddressSpace};

    use super::*;

    struct TestEmitCtx {
        operations: Vec<String>,
        preflight: bool,
        trace_result: bool,
        next_tmp: usize,
    }

    impl TestEmitCtx {
        fn preflight() -> Self {
            Self {
                operations: Vec::new(),
                preflight: true,
                trace_result: false,
                next_tmp: 0,
            }
        }

        fn legacy() -> Self {
            Self {
                operations: Vec::new(),
                preflight: false,
                trace_result: false,
                next_tmp: 0,
            }
        }

        fn metered() -> Self {
            Self {
                trace_result: true,
                ..Self::legacy()
            }
        }
    }

    impl ExtEmitCtx for TestEmitCtx {
        fn is_preflight(&self) -> bool {
            self.preflight
        }

        fn read_var(&mut self, var: Variable) -> String {
            let value = format!("r{}", var.index());
            self.operations.push(format!("read({value})"));
            value
        }

        fn peek_var(&mut self, _var: Variable) -> String {
            unreachable!()
        }

        fn advance_timestamp(&mut self, _slots: u32) {
            unreachable!()
        }

        fn write_var(&mut self, _var: Variable, _val: &str) {
            unreachable!()
        }

        fn write_line(&mut self, line: &str) {
            self.operations.push(line.to_string());
        }

        fn emit_trap(&mut self) {
            self.operations.push("trap".to_string());
        }

        fn read_mem(&mut self, _base: &str, _offset: i16, _width: u8, _signed: bool) -> String {
            unreachable!()
        }

        fn write_mem(&mut self, _base: &str, _offset: i16, _val: &str, _width: u8) {
            unreachable!()
        }

        fn write_aligned_mem_block(&mut self, _addr: &str, _val: &str) {
            unreachable!()
        }

        fn reserve_preflight_timestamp_slots(&mut self, slots: &str) {
            self.operations.push(format!("reserve({slots})"));
        }

        fn reserve_replay_values(&mut self, count: &str) {
            self.operations.push(format!("reserve_replay({count})"));
        }

        fn count_replay_values(&mut self, count: &str) {
            if self.trace_result {
                self.operations.push(format!("count_replay({count})"));
            }
        }

        fn append_replay_value(&mut self, value: &str) {
            self.operations.push(format!("replay_value({value})"));
        }

        fn emit_call(&mut self, name: &str, args: &[&str]) {
            self.operations.push(format!("{name}({})", args.join(", ")));
        }

        fn emit_call_without_page_flush(&mut self, name: &str, args: &[&str]) {
            self.emit_call(name, args);
        }

        fn emit_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String {
            let value = format!("tmp{}", self.next_tmp);
            self.next_tmp += 1;
            self.operations
                .push(format!("{ret_ty} {value} = {name}({})", args.join(", ")));
            value
        }

        fn emit_call_with_trace_result(
            &mut self,
            ret_ty: &str,
            name: &str,
            args: &[&str],
        ) -> Option<String> {
            if self.trace_result {
                Some(self.emit_call_expr(ret_ty, name, args))
            } else {
                self.emit_call(name, args);
                None
            }
        }

        fn trace_chip(&mut self, chip_idx: u32, count_expr: &str) {
            self.operations
                .push(format!("trace({chip_idx}, {count_expr})"));
        }

        fn trace_chip_if_nonzero(&mut self, chip_idx: u32, count_expr: &str) {
            self.operations
                .push(format!("trace_nonzero({chip_idx}, {count_expr})"));
        }

        fn trace_page_access(
            &mut self,
            _addr: &str,
            _width: MemWidth,
            _addr_space: PageAddressSpace,
        ) {
            unreachable!()
        }

        fn trace_page_access_u64_range(
            &mut self,
            _base_addr: &str,
            _num_dwords: &str,
            _addr_space: PageAddressSpace,
        ) {
            unreachable!()
        }
    }

    #[test]
    fn fixed_page_bound_covers_call_memory_ranges() {
        assert_eq!(DEFERRAL_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION, 4);
    }

    fn commit_from_values(values: [u32; VM_DIGEST_WIDTH]) -> [u8; DEFERRAL_COMMIT_NUM_BYTES] {
        let mut commit = [0u8; DEFERRAL_COMMIT_NUM_BYTES];
        for (chunk, value) in commit.chunks_exact_mut(size_of::<u32>()).zip(values) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        commit
    }

    fn pack_value_pairs(values: [u32; VM_DIGEST_WIDTH]) -> [u64; VM_DIGEST_WIDTH / 2] {
        std::array::from_fn(|i| u64::from(values[2 * i]) | (u64::from(values[2 * i + 1]) << 32))
    }

    #[test]
    fn call_callback_dirty_capacity_failure_is_atomic() {
        // Each deferral owns 64 bytes of AS4 accumulator state. Deferral 4096
        // starts on dirty page 64, just beyond a one-word bitmap.
        let def_idx = 4096;
        let input_commit = commit_from_values(std::array::from_fn(|i| i as u32 + 1));
        let input_commit_key = input_commit.to_vec();
        let input_raw = vec![7u8; 8];
        let output_raw = vec![9u8; 8];
        let output_commit = commit_from_values(std::array::from_fn(|i| i as u32 + 21));
        let mut deferrals = vec![DeferralState::default(); def_idx + 1];
        deferrals[def_idx].store_input(input_commit_key.clone(), input_raw.clone());
        let preserved_commit = vec![0x44; DEFERRAL_COMMIT_NUM_BYTES];
        let preserved_output_commit = vec![0x55; DEFERRAL_COMMIT_NUM_BYTES];
        let preserved_output = vec![0x66; 8];
        deferrals[def_idx].store_input(preserved_commit.clone(), vec![0x33; 8]);
        deferrals[def_idx].store_output(
            &preserved_commit,
            preserved_output_commit.clone(),
            preserved_output.clone(),
        );
        let deferral_fn = Arc::new(DeferralFn::new({
            let output_raw = output_raw.clone();
            move |_| output_raw.clone()
        }));
        let deferral_ctx = DeferralCtx::new(
            vec![deferral_fn; def_idx + 1],
            Box::new(move |_, _| output_commit),
            Box::new(|left, right| std::array::from_fn(|i| left[i] + right[i])),
        );

        let accumulator_elements = 2 * (def_idx + 1) * VM_DIGEST_WIDTH;
        let mut deferral_memory = vec![BabyBear::new(17); accumulator_elements];
        let original_memory = deferral_memory.clone();
        let mut dirty_pages = [0x5a5a_5a5a_5a5a_5a5au64];
        let original_dirty_pages = dirty_pages;
        let mut replay = [0xa5a5_a5a5_a5a5_a5a5u64; DEFERRAL_CALL_REPLAY_WORDS];
        let original_replay = replay;
        let mut input_stream = VecDeque::new();
        let mut hint_stream = HintStream::default();
        let mut rng = StdRng::seed_from_u64(0);
        let mut memory = [0u8; 1];
        let mut public_values = [];
        let mut io = OpenVmIoState {
            input_stream: &mut input_stream,
            hint_stream: &mut hint_stream,
            rng: &mut rng,
            memory_ptr: memory.as_mut_ptr(),
            public_values: &mut public_values,
            deferral_memory: deferral_memory.as_mut_ptr().cast(),
            deferral_memory_len_bytes: size_of_val(deferral_memory.as_slice()),
            preflight_deferral_dirty_pages: Some(&mut dirty_pages),
            deferrals: &mut deferrals,
        };

        let succeeded = unsafe {
            host_deferral_call_lookup::<BabyBear>(
                (&deferral_ctx as *const DeferralCtx).cast_mut().cast(),
                (&mut io as *mut OpenVmIoState<'_>).cast(),
                def_idx as u32,
                input_commit.as_ptr(),
                replay.as_mut_ptr().cast(),
                replay.as_mut_ptr().add(DEFERRAL_OUTPUT_KEY_WORDS),
            )
        };
        assert!(!succeeded);
        assert_eq!(deferral_memory, original_memory);
        assert_eq!(dirty_pages, original_dirty_pages);
        assert_eq!(replay, original_replay);
        match deferrals[def_idx].get_input(&input_commit_key) {
            InputMapVal::Raw(raw) => assert_eq!(raw, &input_raw),
            InputMapVal::Output(_) => panic!("failed callback mutated the input map"),
        }
        assert!(deferrals[def_idx]
            .try_get_output(&output_commit.to_vec())
            .is_none());
        assert_eq!(
            deferrals[def_idx].try_get_output(&preserved_output_commit),
            Some(&preserved_output)
        );
    }

    #[test]
    fn call_callback_commits_state_and_exact_replay_order() {
        let input_values = std::array::from_fn(|i| i as u32 + 1);
        let output_values = std::array::from_fn(|i| i as u32 + 21);
        let input_commit = commit_from_values(input_values);
        let input_commit_key = input_commit.to_vec();
        let output_commit = commit_from_values(output_values);
        let output_commit_key = output_commit.to_vec();
        let output_raw = vec![9u8; 8];
        let mut deferral = DeferralState::default();
        deferral.store_input(input_commit_key.clone(), vec![7u8; 8]);
        let deferral_ctx = DeferralCtx::new(
            vec![Arc::new(DeferralFn::new({
                let output_raw = output_raw.clone();
                move |_| output_raw.clone()
            }))],
            Box::new(move |_, _| output_commit),
            Box::new(|left, right| std::array::from_fn(|i| left[i] + right[i])),
        );

        let old_input_values: [u32; VM_DIGEST_WIDTH] = std::array::from_fn(|i| i as u32 + 100);
        let old_output_values: [u32; VM_DIGEST_WIDTH] = std::array::from_fn(|i| i as u32 + 200);
        let mut deferral_memory = old_input_values
            .into_iter()
            .chain(old_output_values)
            .map(BabyBear::new)
            .collect::<Vec<_>>();
        let mut dirty_pages = [0u64; 1];
        let mut replay = [u64::MAX; DEFERRAL_CALL_REPLAY_WORDS];
        let mut deferrals = vec![deferral];
        let mut input_stream = VecDeque::new();
        let mut hint_stream = HintStream::default();
        let mut rng = StdRng::seed_from_u64(0);
        let mut memory = [0u8; 1];
        let mut public_values = [];
        let mut io = OpenVmIoState {
            input_stream: &mut input_stream,
            hint_stream: &mut hint_stream,
            rng: &mut rng,
            memory_ptr: memory.as_mut_ptr(),
            public_values: &mut public_values,
            deferral_memory: deferral_memory.as_mut_ptr().cast(),
            deferral_memory_len_bytes: size_of_val(deferral_memory.as_slice()),
            preflight_deferral_dirty_pages: Some(&mut dirty_pages),
            deferrals: &mut deferrals,
        };

        let succeeded = unsafe {
            host_deferral_call_lookup::<BabyBear>(
                (&deferral_ctx as *const DeferralCtx).cast_mut().cast(),
                (&mut io as *mut OpenVmIoState<'_>).cast(),
                0,
                input_commit.as_ptr(),
                replay.as_mut_ptr().cast(),
                replay.as_mut_ptr().add(DEFERRAL_OUTPUT_KEY_WORDS),
            )
        };
        assert!(succeeded);

        let new_input_values = std::array::from_fn(|i| old_input_values[i] + input_values[i]);
        let new_output_values = std::array::from_fn(|i| old_output_values[i] + output_values[i]);
        let output_words = pack_value_pairs(output_values);
        let new_input_words = pack_value_pairs(new_input_values);
        let new_output_words = pack_value_pairs(new_output_values);
        let expected_replay = [
            output_words[0],
            output_words[1],
            output_words[2],
            output_words[3],
            output_raw.len() as u64,
            new_input_words[0],
            new_input_words[1],
            new_input_words[2],
            new_input_words[3],
            new_output_words[0],
            new_output_words[1],
            new_output_words[2],
            new_output_words[3],
        ];
        assert_eq!(replay, expected_replay);
        assert_eq!(
            deferral_memory,
            new_input_values
                .into_iter()
                .chain(new_output_values)
                .map(BabyBear::new)
                .collect::<Vec<_>>()
        );
        assert_eq!(dirty_pages, [1]);
        match deferrals[0].get_input(&input_commit_key) {
            InputMapVal::Output(commit) => assert_eq!(commit, &output_commit_key),
            InputMapVal::Raw(_) => panic!("successful callback did not update the input map"),
        }
        assert_eq!(
            deferrals[0].try_get_output(&output_commit_key),
            Some(&output_raw)
        );
    }

    #[test]
    fn host_callbacks_report_invalid_advice_without_unwinding_across_ffi() {
        let mut deferral = DeferralState::default();
        let commit = vec![0u8; DEFERRAL_COMMIT_NUM_BYTES];
        deferral.store_input(commit.clone(), vec![1]);
        let deferral_ctx = DeferralCtx::new(
            vec![Arc::new(DeferralFn::new(|_| panic!("advice failure")))],
            Box::new(|_, _| [0; DEFERRAL_COMMIT_NUM_BYTES]),
            Box::new(|_, _| [0; VM_DIGEST_WIDTH]),
        );
        let mut input_stream = VecDeque::new();
        let mut hint_stream = HintStream::default();
        let mut rng = StdRng::seed_from_u64(0);
        let mut memory = [0u8; 1];
        let mut public_values = [];
        let mut deferral_memory = [BabyBear::new(0); 2 * VM_DIGEST_WIDTH];
        let mut deferrals = vec![deferral];
        let mut io = OpenVmIoState {
            input_stream: &mut input_stream,
            hint_stream: &mut hint_stream,
            rng: &mut rng,
            memory_ptr: memory.as_mut_ptr(),
            public_values: &mut public_values,
            deferral_memory: deferral_memory.as_mut_ptr().cast(),
            deferral_memory_len_bytes: size_of_val(&deferral_memory),
            preflight_deferral_dirty_pages: None,
            deferrals: &mut deferrals,
        };
        let d_ctx = &deferral_ctx as *const DeferralCtx as *mut c_void;
        let io_ctx = &mut io as *mut OpenVmIoState<'_> as *mut c_void;
        let input_commit = [0u8; DEFERRAL_COMMIT_NUM_BYTES];
        let mut output_key = [0u8; DEFERRAL_OUTPUT_KEY_BYTES];
        let mut accumulator_words = [0u64; DEFERRAL_ACCUMULATOR_WORDS];
        let mut output = [0u8; 1];

        unsafe {
            assert!(!host_deferral_call_lookup::<BabyBear>(
                d_ctx,
                io_ctx,
                0,
                input_commit.as_ptr(),
                output_key.as_mut_ptr(),
                accumulator_words.as_mut_ptr(),
            ));
            assert!(!host_deferral_call_lookup::<BabyBear>(
                d_ctx,
                io_ctx,
                1,
                input_commit.as_ptr(),
                output_key.as_mut_ptr(),
                accumulator_words.as_mut_ptr(),
            ));
            assert!(!host_deferral_output_lookup(
                d_ctx,
                io_ctx,
                1,
                input_commit.as_ptr(),
                output.as_mut_ptr(),
                1,
            ));
        }
    }

    #[test]
    fn output_source_drains_before_and_after_variable_writes() {
        let source = include_str!("../c/rvr_ext_deferral.c");
        assert_eq!(source.matches("g_deferral.call_lookup(").count(), 1);
        assert_eq!(source.matches("g_deferral.output_lookup(").count(), 1);
        let length_rejection = source
            .find("if (unlikely(output_len_u64 > UINT32_MAX)) return false;")
            .expect("RVR OUTPUT must reject a nonzero length high word");
        let output_lookup = source.find("g_deferral.output_lookup(").unwrap();
        assert!(length_rejection < output_lookup);
        assert_eq!(
            source
                .matches("flush_main_memory_page_buffer(state);")
                .count(),
            2
        );
        assert!(source.contains("chunk_start += OUTPUT_ROWS_PER_PAGE_BUFFER"));
    }

    #[test]
    fn call_preflight_reserves_exact_schedule_and_emits_replay_values_once() {
        let instruction = DeferralCallInstr {
            rd_reg: Variable::new(1),
            rs_reg: Variable::new(2),
            def_idx: 3,
            poseidon2_chip_idx: None,
        };
        assert!(instruction.supports_preflight());

        let mut preflight = TestEmitCtx::preflight();
        instruction.emit_c(&mut preflight);
        assert_eq!(
            preflight.operations,
            [
                "read(r1)",
                "read(r2)",
                "reserve(17u)",
                "reserve_replay(13u)",
                "uint64_t deferral_replay[13u];",
                "bool tmp0 = rvr_ext_deferral_call(state, r1, r2, 3u, deferral_replay)",
                "if (unlikely(!tmp0)) {",
                "trap",
                "}",
                "for (uint32_t deferral_replay_idx = 0u; deferral_replay_idx < 13u; ++deferral_replay_idx) {",
                "replay_value(deferral_replay[deferral_replay_idx])",
                "}",
            ]
        );
        assert_eq!(
            preflight
                .operations
                .iter()
                .filter(|operation| operation.contains("rvr_ext_deferral_call("))
                .count(),
            1
        );

        let mut legacy = TestEmitCtx::legacy();
        instruction.emit_c(&mut legacy);
        assert_eq!(
            legacy.operations,
            [
                "read(r1)",
                "read(r2)",
                "bool tmp0 = rvr_ext_deferral_call(state, r1, r2, 3u, NULL)",
                "if (unlikely(!tmp0)) {",
                "trap",
                "}",
            ]
        );
    }

    #[test]
    fn output_preflight_reserves_dynamic_schedule_and_calls_closure_once() {
        let instruction = DeferralOutputInstr {
            rd_reg: Variable::new(1),
            rs_reg: Variable::new(2),
            def_idx: 3,
            output_chip_idx: None,
            poseidon2_chip_idx: None,
        };
        assert!(instruction.supports_preflight());

        let mut preflight = TestEmitCtx::preflight();
        instruction.emit_c(&mut preflight);
        assert_eq!(
            preflight.operations,
            [
                "read(r1)",
                "read(r2)",
                "if (unlikely(r2 > OPENVM_MEM_SIZE - 40u)) {",
                "trap",
                "}",
                "uint64_t deferral_output_len_u64 = peek_mem_u64(state, r2 + 32ull);",
                "if (unlikely(deferral_output_len_u64 > UINT32_MAX)) {",
                "trap",
                "}",
                "uint32_t deferral_output_len = (uint32_t)deferral_output_len_u64;",
                "if (unlikely((deferral_output_len & 7u) != 0u)) {",
                "trap",
                "}",
                "uint32_t deferral_output_words = deferral_output_len / 8u;",
                "reserve(5u + deferral_output_words)",
                "reserve_replay(1u + deferral_output_words)",
                "uint32_t deferral_num_rows;",
                "bool tmp0 = rvr_ext_deferral_output(state, r1, r2, 3u, &deferral_num_rows)",
                "if (unlikely(!tmp0)) {",
                "trap",
                "}",
                "trace_nonzero(4294967295, deferral_num_rows - 1u)",
                "trace(4294967295, deferral_num_rows)",
                "for (uint32_t deferral_replay_idx = 0u; deferral_replay_idx <= deferral_output_words; ++deferral_replay_idx) {",
                "replay_value(deferral_replay_idx == 0u ? (uint64_t)deferral_output_words : peek_mem_u64(state, r1 + (uint64_t)(deferral_replay_idx - 1u) * 8ull))",
                "}",
            ]
        );
        assert_eq!(
            preflight
                .operations
                .iter()
                .filter(|operation| operation.contains("rvr_ext_deferral_output("))
                .count(),
            1
        );

        let mut legacy = TestEmitCtx::legacy();
        instruction.emit_c(&mut legacy);
        assert_eq!(
            legacy.operations,
            [
                "read(r1)",
                "read(r2)",
                "uint32_t deferral_num_rows;",
                "bool tmp0 = rvr_ext_deferral_output(state, r1, r2, 3u, &deferral_num_rows)",
                "if (unlikely(!tmp0)) {",
                "trap",
                "}",
                "trace_nonzero(4294967295, deferral_num_rows - 1u)",
                "trace(4294967295, deferral_num_rows)",
            ]
        );

        let mut metered = TestEmitCtx::metered();
        instruction.emit_c(&mut metered);
        assert_eq!(
            metered.operations,
            [
                "read(r1)",
                "read(r2)",
                "uint32_t deferral_num_rows;",
                "bool tmp0 = rvr_ext_deferral_output(state, r1, r2, 3u, &deferral_num_rows)",
                "if (unlikely(!tmp0)) {",
                "trap",
                "}",
                "trace_nonzero(4294967295, deferral_num_rows - 1u)",
                "trace(4294967295, deferral_num_rows)",
                "count_replay(1ull + 4ull * ((uint64_t)deferral_num_rows - 1ull))",
            ]
        );
        let checked_call = metered
            .operations
            .iter()
            .position(|operation| operation.contains("rvr_ext_deferral_output"))
            .unwrap();
        let replay_value_count = metered
            .operations
            .iter()
            .position(|operation| operation.starts_with("count_replay"))
            .unwrap();
        assert!(checked_call < replay_value_count);
    }
}
