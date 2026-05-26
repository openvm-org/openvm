//! Deferral extension for rvr-openvm: IR nodes for CALL/OUTPUT and the
//! `DeferralRvrExtension` for lifting them via double FFI.
#![cfg(feature = "rvr")]

use std::{ffi::c_void, sync::Arc};

use openvm_circuit::arch::{
    deferral::{DeferralFn, InputMapVal},
    rvr::io::OpenVmIoState,
};
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ext_ffi_common::{
    DEFERRAL_COMMIT_NUM_BYTES, DEFERRAL_DIGEST_SIZE, DEFERRAL_OUTPUT_KEY_BYTES,
};
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{
    air_index_to_c, decode_reg, opcode_air_idx, AirIndex, ExtensionError, RvrExtension,
    RvrExtensionCtx,
};

/// `(def_idx, output_raw) → output_commit` hasher registered by the host.
pub type DeferralHashFn = Arc<dyn Fn(u32, &[u8]) -> [u8; DEFERRAL_COMMIT_NUM_BYTES] + Send + Sync>;

/// Poseidon2 compression over deferral accumulator field elements.
/// Values cross the crate boundary as canonical u32s.
pub type DeferralCompressFn = Arc<
    dyn Fn([u32; DEFERRAL_DIGEST_SIZE], [u32; DEFERRAL_DIGEST_SIZE]) -> [u32; DEFERRAL_DIGEST_SIZE]
        + Send
        + Sync,
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
    pub rd_reg: Reg,
    pub rs_reg: Reg,
    pub def_idx: u32,
    pub poseidon2_chip_idx: Option<AirIndex>,
}

impl ExtInstr for DeferralCallInstr {
    fn opname(&self) -> &str {
        "def_call"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_reg(self.rd_reg);
        let rs = ctx.read_reg(self.rs_reg);
        let poseidon2 = air_index_to_c(self.poseidon2_chip_idx);
        ctx.write_line(&format!(
            "rvr_ext_deferral_call(state, {rd}, {rs}, {}u, {poseidon2}u);",
            self.def_idx
        ));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// IR node for a deferral OUTPUT instruction.
#[derive(Debug, Clone)]
pub struct DeferralOutputInstr {
    pub rd_reg: Reg,
    pub rs_reg: Reg,
    pub def_idx: u32,
    pub output_chip_idx: Option<AirIndex>,
    pub poseidon2_chip_idx: Option<AirIndex>,
}

impl ExtInstr for DeferralOutputInstr {
    fn opname(&self) -> &str {
        "def_output"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_reg(self.rd_reg);
        let rs = ctx.read_reg(self.rs_reg);
        let output = air_index_to_c(self.output_chip_idx);
        let poseidon2 = air_index_to_c(self.poseidon2_chip_idx);
        ctx.write_line(&format!(
            "rvr_ext_deferral_output(state, {rd}, {rs}, {}u, {output}u, {poseidon2}u);",
            self.def_idx
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

/// The Deferral extension (CALL + OUTPUT opcodes).
pub struct DeferralRvrExtension {
    #[allow(dead_code)]
    call_chip_idx: Option<AirIndex>,
    output_chip_idx: Option<AirIndex>,
    poseidon2_chip_idx: Option<AirIndex>,
    deferral_ctx: DeferralCtx,
}

impl DeferralRvrExtension {
    pub fn new(
        ctx: Option<&RvrExtensionCtx>,
        fns: Vec<Arc<DeferralFn>>,
        hash: DeferralHashFn,
        compress: DeferralCompressFn,
    ) -> Result<Self, ExtensionError> {
        let call_chip_idx = opcode_air_idx(ctx, DeferralOpcode::CALL)?;
        let output_chip_idx = opcode_air_idx(ctx, DeferralOpcode::OUTPUT)?;
        // Poseidon2 periphery chip: in extend_circuit, the hasher is added
        // right before the CALL chip. Due to reverse ordering of AIR indices,
        // poseidon2_air_idx = call_air_idx + 1.
        let poseidon2_chip_idx = call_chip_idx.map(AirIndex::next);

        Ok(Self {
            call_chip_idx,
            output_chip_idx,
            poseidon2_chip_idx,
            deferral_ctx: DeferralCtx::new(fns, hash, compress),
        })
    }
}

impl<F: PrimeField32> RvrExtension<F> for DeferralRvrExtension {
    fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode == DeferralOpcode::CALL.global_opcode_usize() {
            let rd_reg = decode_reg(insn.a);
            let rs_reg = decode_reg(insn.b);
            let def_idx = insn.c.as_canonical_u32();
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(DeferralCallInstr {
                    rd_reg,
                    rs_reg,
                    def_idx,
                    poseidon2_chip_idx: self.poseidon2_chip_idx,
                })),
                source_loc: None,
            }));
        }

        if opcode == DeferralOpcode::OUTPUT.global_opcode_usize() {
            let rd_reg = decode_reg(insn.a);
            let rs_reg = decode_reg(insn.b);
            let def_idx = insn.c.as_canonical_u32();
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(DeferralOutputInstr {
                    rd_reg,
                    rs_reg,
                    def_idx,
                    output_chip_idx: self.output_chip_idx,
                    poseidon2_chip_idx: self.poseidon2_chip_idx,
                })),
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

    fn staticlib_file(&self) -> (&'static str, &'static [u8]) {
        (
            "librvr_openvm_ext_deferral_ffi.a",
            include_bytes!(env!("RVR_DEFERRAL_FFI_STATICLIB")),
        )
    }

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
            call_lookup: host_deferral_call_lookup::<F>,
            output_lookup: host_deferral_output_lookup::<F>,
        };
        unsafe { register_fn(&callbacks) };
        Ok(())
    }
}

// ── Deferral accumulator sync (DEFERRAL_AS) ────────────────────────────────
//
// CALL writes new `(input_acc, output_acc)` values to DEFERRAL_AS.

fn commit_bytes_to_field_values(
    bytes: &[u8; DEFERRAL_COMMIT_NUM_BYTES],
) -> [u32; DEFERRAL_DIGEST_SIZE] {
    let mut out = [0u32; DEFERRAL_DIGEST_SIZE];
    for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = u32::from_le_bytes(chunk.try_into().unwrap());
    }
    out
}

fn has_deferral_digest_range<F: PrimeField32>(io: &OpenVmIoState<'_, F>, ptr: usize) -> bool {
    !io.deferral_memory.is_null()
        && ptr
            .checked_add(DEFERRAL_DIGEST_SIZE)
            .is_some_and(|end| end <= io.deferral_memory_len)
}

/// Reads one accumulator digest from DEFERRAL_AS as canonical u32 values.
///
/// # Safety
/// Caller must ensure `io.deferral_memory` is a valid `*mut F` pointing into
/// a live DEFERRAL_AS buffer with at least `io.deferral_memory_len` F slots.
unsafe fn read_deferral_digest<F: PrimeField32>(
    io: &OpenVmIoState<'_, F>,
    ptr: usize,
) -> Option<[u32; DEFERRAL_DIGEST_SIZE]> {
    if !has_deferral_digest_range(io, ptr) {
        return None;
    }
    let mut out = [0u32; DEFERRAL_DIGEST_SIZE];
    for (i, dst) in out.iter_mut().enumerate() {
        *dst = (*io.deferral_memory.add(ptr + i)).as_canonical_u32();
    }
    Some(out)
}

/// Writes one accumulator digest to DEFERRAL_AS.
///
/// # Safety
/// See `read_deferral_digest`. Each canonical u32 is converted back to `F`
/// before writing to AS=4.
unsafe fn write_deferral_digest<F: PrimeField32>(
    io: &mut OpenVmIoState<'_, F>,
    ptr: usize,
    values: [u32; DEFERRAL_DIGEST_SIZE],
) -> bool {
    if !has_deferral_digest_range(io, ptr) {
        return false;
    }
    for (i, value) in values.into_iter().enumerate() {
        *io.deferral_memory.add(ptr + i) = F::from_u32(value);
    }
    true
}

/// Updates the input/output accumulator slots for one deferral CALL.
/// Slot offsets are in F-element units.
unsafe fn update_deferral_accumulators<F: PrimeField32>(
    deferral_ctx: &DeferralCtx,
    io: &mut OpenVmIoState<'_, F>,
    def_idx: u32,
    input_commit: &[u8; DEFERRAL_COMMIT_NUM_BYTES],
    output_commit: &[u8; DEFERRAL_COMMIT_NUM_BYTES],
) -> bool {
    let input_acc_ptr = 2 * def_idx as usize * DEFERRAL_DIGEST_SIZE;
    let output_acc_ptr = input_acc_ptr + DEFERRAL_DIGEST_SIZE;
    let Some(old_input_acc) = read_deferral_digest(io, input_acc_ptr) else {
        return false;
    };
    let Some(old_output_acc) = read_deferral_digest(io, output_acc_ptr) else {
        return false;
    };
    let new_input_acc =
        (deferral_ctx.compress)(old_input_acc, commit_bytes_to_field_values(input_commit));
    let new_output_acc =
        (deferral_ctx.compress)(old_output_acc, commit_bytes_to_field_values(output_commit));
    write_deferral_digest(io, input_acc_ptr, new_input_acc)
        && write_deferral_digest(io, output_acc_ptr, new_output_acc)
}

// ── Host callbacks ──────────────────────────────────────────────────────────

type RegisterFn = unsafe extern "C" fn(*const DeferralHostCallbacks);

/// Must match the C `DeferralHostCallbacks` layout in `rvr_ext_deferral.c`.
#[repr(C)]
pub struct DeferralHostCallbacks {
    pub ctx: *mut c_void,
    pub call_lookup: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *const u8, *mut u8) -> i32,
    pub output_lookup:
        unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *const u8, *mut u8, u32) -> i32,
}

/// Deferral CALL lookup. Returns 1 on hit, 0 on miss.
///
/// # Safety
///
/// `d_ctx` must point to a valid `DeferralCtx`. `io_ctx` must point to a valid
/// `OpenVmIoState<'_, F>`.
/// `input_commit_raw` must point to `DEFERRAL_COMMIT_NUM_BYTES` readable bytes.
/// `output_key_out` must point to `DEFERRAL_OUTPUT_KEY_BYTES` writable bytes.
pub unsafe extern "C" fn host_deferral_call_lookup<F: PrimeField32>(
    d_ctx: *mut c_void,
    io_ctx: *mut c_void,
    def_idx: u32,
    input_commit_raw: *const u8,
    output_key_out: *mut u8,
) -> i32 {
    let deferral_ctx = unsafe { &*(d_ctx as *const DeferralCtx) };
    let io = unsafe { &mut *(io_ctx as *mut OpenVmIoState<'_, F>) };

    let mut input_commit = [0u8; DEFERRAL_COMMIT_NUM_BYTES];
    input_commit.copy_from_slice(unsafe {
        std::slice::from_raw_parts(input_commit_raw, DEFERRAL_COMMIT_NUM_BYTES)
    });
    let input_commit_key = input_commit.to_vec();

    let Some(deferral_state) = io.deferrals.get_mut(def_idx as usize) else {
        return 0;
    };

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

    if !unsafe {
        update_deferral_accumulators(deferral_ctx, io, def_idx, &input_commit, &output_commit)
    } {
        return 0;
    }

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
    1
}

/// Deferral OUTPUT lookup: `deferrals[def_idx].output_map[output_commit]`.
/// Returns 1 on hit, 0 on miss.
///
/// # Safety
///
/// `d_ctx` must point to a valid `DeferralCtx`. `io_ctx` must point to a
/// valid `OpenVmIoState<'_, F>`.
/// `output_commit_raw` must point to `DEFERRAL_COMMIT_NUM_BYTES` readable bytes.
/// `output_raw_out` must point to at least `expected_len` writable bytes.
pub unsafe extern "C" fn host_deferral_output_lookup<F: PrimeField32>(
    _d_ctx: *mut c_void,
    io_ctx: *mut c_void,
    def_idx: u32,
    output_commit_raw: *const u8,
    output_raw_out: *mut u8,
    expected_len: u32,
) -> i32 {
    let io = unsafe { &*(io_ctx as *const OpenVmIoState<'_, F>) };

    let output_commit: Vec<u8> = unsafe {
        std::slice::from_raw_parts(output_commit_raw, DEFERRAL_COMMIT_NUM_BYTES).to_vec()
    };
    let Some(state) = io.deferrals.get(def_idx as usize) else {
        return 0;
    };
    let raw = state.get_output(&output_commit);
    // TODO: change these panics to something better to handle across the FFI boundary.
    assert_eq!(raw.len(), expected_len as usize);
    unsafe { std::ptr::copy_nonoverlapping(raw.as_ptr(), output_raw_out, raw.len()) };
    1
}
