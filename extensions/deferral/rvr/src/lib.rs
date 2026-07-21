//! Deferral extension for rvr-openvm: IR nodes for CALL/OUTPUT and the
//! `DeferralRvrExtension` for lifting them.
#![cfg(feature = "rvr")]

use std::{
    ffi::c_void,
    mem::{align_of, size_of},
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
    CfgEffect, ExtEmitCtx, ExtInstr, FixedTraceRows, InstrAt, LiftedInstr, ValueSlot,
};
use rvr_openvm_lift::{
    air_index_to_c, decode_value_slot, fixed_trace_rows_for_chip, opcode_air_idx, AirIndex,
    ExtensionError, RvrExtension, RvrExtensionCtx, RvrInstruction, RvrRuntimeExtension,
};

fn decode_reg(value: u32) -> ValueSlot {
    decode_value_slot(value, RV64_REGISTER_BYTES as u32, RV64_NUM_REGISTERS as u32)
}

/// Size in bytes of a serialized deferral commitment.
pub const DEFERRAL_COMMIT_NUM_BYTES: usize = VM_DIGEST_WIDTH * core::mem::size_of::<u32>();
/// Size in bytes of a deferral output key: commitment followed by output length.
pub const DEFERRAL_OUTPUT_KEY_BYTES: usize =
    DEFERRAL_COMMIT_NUM_BYTES + core::mem::size_of::<u64>();

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
    pub rd_reg: ValueSlot,
    pub rs_reg: ValueSlot,
    pub def_idx: u32,
    pub poseidon2_chip_idx: Option<AirIndex>,
}

impl ExtInstr for DeferralCallInstr {
    fn opname(&self) -> &str {
        "def_call"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_slot(self.rd_reg);
        let rs = ctx.read_slot(self.rs_reg);
        let def_idx = format!("{}u", self.def_idx);
        ctx.emit_call("rvr_ext_deferral_call", &["state", &rd, &rs, &def_idx]);
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
}

/// IR node for a deferral OUTPUT instruction.
#[derive(Debug, Clone)]
pub struct DeferralOutputInstr {
    pub rd_reg: ValueSlot,
    pub rs_reg: ValueSlot,
    pub def_idx: u32,
    pub output_chip_idx: Option<AirIndex>,
    pub poseidon2_chip_idx: Option<AirIndex>,
}

impl ExtInstr for DeferralOutputInstr {
    fn opname(&self) -> &str {
        "def_output"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_slot(self.rd_reg);
        let rs = ctx.read_slot(self.rs_reg);
        let output = air_index_to_c(self.output_chip_idx);
        let poseidon2 = air_index_to_c(self.poseidon2_chip_idx);
        let def_idx = format!("{}u", self.def_idx);
        let num_rows = ctx.emit_call_with_trace_result(
            "uint32_t",
            "rvr_ext_deferral_output",
            &["state", &rd, &rs, &def_idx],
        );
        if let Some(num_rows) = num_rows {
            ctx.trace_chip_if_nonzero(output, &format!("{num_rows} - 1u"));
            ctx.trace_chip(poseidon2, &num_rows);
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
}

type DeferralCallLookupFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *const u8, *mut u8);
type DeferralOutputLookupFn =
    unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *const u8, *mut u8, u32);

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
unsafe fn deferral_memory<'a, F: PrimeField32>(io: &'a mut OpenVmIoState<'_>) -> &'a mut [F] {
    assert_eq!(io.deferral_memory_len_bytes % size_of::<F>(), 0);
    if io.deferral_memory_len_bytes == 0 {
        return &mut [];
    }
    assert!(!io.deferral_memory.is_null());
    assert_eq!(io.deferral_memory.addr() % align_of::<F>(), 0);
    unsafe {
        std::slice::from_raw_parts_mut(
            io.deferral_memory.cast(),
            io.deferral_memory_len_bytes / size_of::<F>(),
        )
    }
}

fn read_deferral_digest<F: PrimeField32>(
    memory: &[F],
    ptr: usize,
) -> Option<[u32; VM_DIGEST_WIDTH]> {
    let end = ptr.checked_add(VM_DIGEST_WIDTH)?;
    let values = memory.get(ptr..end)?;
    Some(std::array::from_fn(|i| values[i].as_canonical_u32()))
}

fn write_deferral_digest<F: PrimeField32>(
    memory: &mut [F],
    ptr: usize,
    values: [u32; VM_DIGEST_WIDTH],
) -> bool {
    let Some(end) = ptr.checked_add(VM_DIGEST_WIDTH) else {
        return false;
    };
    let Some(dst) = memory.get_mut(ptr..end) else {
        return false;
    };
    dst.iter_mut()
        .zip(values)
        .for_each(|(dst, value)| *dst = F::from_u32(value));
    true
}

/// Updates the input/output accumulator slots for one deferral CALL.
/// Slot offsets are in F-element units.
unsafe fn update_deferral_accumulators<F: PrimeField32>(
    deferral_ctx: &DeferralCtx,
    io: &mut OpenVmIoState<'_>,
    def_idx: u32,
    input_commit: &[u8; DEFERRAL_COMMIT_NUM_BYTES],
    output_commit: &[u8; DEFERRAL_COMMIT_NUM_BYTES],
) -> bool {
    let memory = unsafe { deferral_memory::<F>(io) };
    let input_acc_ptr = 2 * def_idx as usize * VM_DIGEST_WIDTH;
    let output_acc_ptr = input_acc_ptr + VM_DIGEST_WIDTH;
    let Some(old_input_acc) = read_deferral_digest(memory, input_acc_ptr) else {
        return false;
    };
    let Some(old_output_acc) = read_deferral_digest(memory, output_acc_ptr) else {
        return false;
    };
    let new_input_acc =
        (deferral_ctx.compress)(old_input_acc, commit_bytes_to_field_values(input_commit));
    let new_output_acc =
        (deferral_ctx.compress)(old_output_acc, commit_bytes_to_field_values(output_commit));
    write_deferral_digest(memory, input_acc_ptr, new_input_acc)
        && write_deferral_digest(memory, output_acc_ptr, new_output_acc)
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

/// Deferral CALL lookup. Panics if `def_idx` or the accumulator update is
/// invalid (a hard error in the lifted program).
///
/// # Safety
///
/// `d_ctx` must point to a valid `DeferralCtx`. `io_ctx` must point to a valid
/// `OpenVmIoState` whose deferral memory contains native `F` values.
/// `input_commit_raw` must point to `DEFERRAL_COMMIT_NUM_BYTES` readable bytes.
/// `output_key_out` must point to `DEFERRAL_OUTPUT_KEY_BYTES` writable bytes.
pub unsafe extern "C" fn host_deferral_call_lookup<F: PrimeField32>(
    d_ctx: *mut c_void,
    io_ctx: *mut c_void,
    def_idx: u32,
    input_commit_raw: *const u8,
    output_key_out: *mut u8,
) {
    let deferral_ctx = unsafe { &*(d_ctx as *const DeferralCtx) };
    let io = unsafe { &mut *(io_ctx as *mut OpenVmIoState<'_>) };

    let mut input_commit = [0u8; DEFERRAL_COMMIT_NUM_BYTES];
    input_commit.copy_from_slice(unsafe {
        std::slice::from_raw_parts(input_commit_raw, DEFERRAL_COMMIT_NUM_BYTES)
    });
    let input_commit_key = input_commit.to_vec();

    let deferral_state = io
        .deferrals
        .get_mut(def_idx as usize)
        .unwrap_or_else(|| panic!("deferral CALL lookup failed: def_idx={def_idx} out of range"));

    let (output_commit, output_len) = match deferral_state.get_input(&input_commit_key).clone() {
        InputMapVal::Output(commit) => {
            let len = deferral_state.get_output(&commit).len() as u64;
            let output_commit: [u8; DEFERRAL_COMMIT_NUM_BYTES] =
                commit.as_slice().try_into().unwrap();
            (output_commit, len)
        }
        InputMapVal::Raw(input_raw) => {
            let deferral_fn = deferral_ctx
                .fns
                .get(def_idx as usize)
                .expect("deferral CALL: def_idx has no closure registered");
            let output_raw = deferral_fn.call_raw(&input_raw);
            let commit = (deferral_ctx.hash)(def_idx, &output_raw);
            let len = output_raw.len() as u64;
            deferral_state.store_output(&input_commit_key, commit.to_vec(), output_raw);
            (commit, len)
        }
    };

    assert!(
        unsafe {
            update_deferral_accumulators::<F>(
                deferral_ctx,
                io,
                def_idx,
                &input_commit,
                &output_commit,
            )
        },
        "deferral CALL lookup failed: accumulator update for def_idx={def_idx}"
    );

    let mut output_key = [0u8; DEFERRAL_OUTPUT_KEY_BYTES];
    output_key[..DEFERRAL_COMMIT_NUM_BYTES].copy_from_slice(&output_commit);
    output_key[DEFERRAL_COMMIT_NUM_BYTES..].copy_from_slice(&output_len.to_le_bytes());
    unsafe {
        std::ptr::copy_nonoverlapping(
            output_key.as_ptr(),
            output_key_out,
            DEFERRAL_OUTPUT_KEY_BYTES,
        );
    }
}

/// Deferral OUTPUT lookup: `deferrals[def_idx].output_map[output_commit]`.
/// Panics on a missing `def_idx` or a length mismatch.
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
) {
    let io = unsafe { &*(io_ctx as *const OpenVmIoState<'_>) };

    let output_commit: Vec<u8> = unsafe {
        std::slice::from_raw_parts(output_commit_raw, DEFERRAL_COMMIT_NUM_BYTES).to_vec()
    };
    let state = io
        .deferrals
        .get(def_idx as usize)
        .unwrap_or_else(|| panic!("deferral OUTPUT lookup failed: def_idx={def_idx} out of range"));
    let raw = state.get_output(&output_commit);
    // TODO: change these panics to something better to handle across the FFI boundary.
    assert_eq!(raw.len(), expected_len as usize);
    unsafe { std::ptr::copy_nonoverlapping(raw.as_ptr(), output_raw_out, raw.len()) };
}
