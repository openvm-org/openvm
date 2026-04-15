//! OpenVM IO runtime: Rust-side IO state with FFI callbacks.
//!
//! All execution IO state (input streams and hint streams) lives in
//! Rust. The generated C code calls back into Rust via function pointers
//! registered through `register_openvm_callbacks`.
//!
//! Metering adjustments for IO instructions are handled entirely in the
//! generated C code (via `trace_io_*` functions in tracer headers). These
//! callbacks are pure IO logic.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::io::Write;

use openvm_stark_backend::p3_field::PrimeField32;
use rand::rngs::StdRng;
use rand::Rng;
use rvr_openvm_ext_ffi_common::{DEFERRAL_COMMIT_NUM_BYTES, DEFERRAL_OUTPUT_KEY_BYTES};

// ── Deferral lookup data ───────────────────────────────────────────────────

/// Pre-computed deferral call lookup table: (def_idx, input_commit) → output_key.
pub type DeferralCallMap =
    HashMap<(u32, [u8; DEFERRAL_COMMIT_NUM_BYTES]), [u8; DEFERRAL_OUTPUT_KEY_BYTES]>;

/// Pre-computed deferral output lookup table: output_commit → output_raw.
pub type DeferralOutputMap = HashMap<[u8; DEFERRAL_COMMIT_NUM_BYTES], Vec<u8>>;

/// Pre-computed deferral lookup data, populated before execution.
#[derive(Default)]
pub struct DeferralData {
    pub call_entries: DeferralCallMap,
    pub output_entries: DeferralOutputMap,
}

/// All IO execution state, owned by Rust.
pub struct OpenVmIoState {
    pub input_stream: VecDeque<Vec<u8>>,
    pub hint_stream: Vec<u8>,
    pub hint_pos: usize,
    pub public_values: Vec<u8>,
    /// Guest memory pointer (constant during execution).
    pub memory: *mut u8,
    /// Persistent RNG matching openvm's `StdRng::seed_from_u64(0)`.
    pub rng: StdRng,
    /// Pre-computed deferral lookup data (empty when no deferral extension).
    pub deferral: DeferralData,
}

/// Function-pointer struct passed to C via `register_openvm_callbacks`.
/// Must match the C `OpenVmHostCallbacks` layout exactly.
#[repr(C)]
pub struct OpenVmHostCallbacks {
    pub ctx: *mut c_void,
    pub hint_input: extern "C" fn(*mut c_void),
    pub print_str: extern "C" fn(*mut c_void, u32, u32),
    pub hint_random: extern "C" fn(*mut c_void, u32),
    pub hint_storew: extern "C" fn(*mut c_void, u32),
    pub hint_buffer: extern "C" fn(*mut c_void, u32, u32),
    pub reveal: extern "C" fn(*mut c_void, u32, u32, u32),
    pub hint_stream_set: unsafe extern "C" fn(*mut c_void, *const u8, u32),
    pub deferral_call_lookup: unsafe extern "C" fn(*mut c_void, u32, *const u8, *mut u8) -> i32,
    pub deferral_output_lookup: unsafe extern "C" fn(*mut c_void, *const u8, *mut u8, u32) -> i32,
}

// ── Callback implementations ────────────────────────────────────────────────

/// HintInput: pop next vector from input_stream, build hint buffer with
/// 4-byte LE length prefix, padded to 4-byte alignment.
pub extern "C" fn host_hint_input(ctx: *mut c_void) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState) };

    if let Some(vec) = io.input_stream.pop_front() {
        let vec_len = vec.len() as u32;
        let padded_len = (vec.len() + 3) & !3;
        let total = 4 + padded_len;
        let mut buf = vec![0u8; total];
        buf[0..4].copy_from_slice(&vec_len.to_le_bytes());
        buf[4..4 + vec.len()].copy_from_slice(&vec);
        io.hint_stream = buf;
    } else {
        io.hint_stream = Vec::new();
    }
    io.hint_pos = 0;
}

/// PrintStr: read UTF-8 from guest memory and print to stdout.
pub extern "C" fn host_print_str(ctx: *mut c_void, ptr: u32, len: u32) {
    let io = unsafe { &*(ctx as *const OpenVmIoState) };
    if len > 0 && !io.memory.is_null() {
        let slice =
            unsafe { std::slice::from_raw_parts(io.memory.add(ptr as usize), len as usize) };
        let _ = std::io::stdout().write_all(slice);
        let _ = std::io::stdout().flush();
    }
}

/// HintRandom: fill hint buffer with random bytes using persistent StdRng
/// (matches openvm's `Rv32HintRandomSubEx`).
pub extern "C" fn host_hint_random(ctx: *mut c_void, num_words: u32) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState) };
    let nbytes = num_words as usize * 4;
    let mut buf = vec![0u8; nbytes];

    for byte in buf.iter_mut() {
        *byte = io.rng.random::<u8>();
    }

    io.hint_stream = buf;
    io.hint_pos = 0;
}

/// HINT_STOREW: copy 4 bytes from hint buffer to guest memory.
pub extern "C" fn host_hint_storew(ctx: *mut c_void, dest_addr: u32) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState) };
    if io.hint_pos + 4 <= io.hint_stream.len() && !io.memory.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(
                io.hint_stream.as_ptr().add(io.hint_pos),
                io.memory.add(dest_addr as usize),
                4,
            );
        }
        io.hint_pos += 4;
    }
}

/// HINT_BUFFER: copy num_words * 4 bytes from hint buffer to guest memory.
pub extern "C" fn host_hint_buffer(ctx: *mut c_void, dest_addr: u32, num_words: u32) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState) };
    let nbytes = num_words as usize * 4;
    if io.hint_pos + nbytes <= io.hint_stream.len() && !io.memory.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(
                io.hint_stream.as_ptr().add(io.hint_pos),
                io.memory.add(dest_addr as usize),
                nbytes,
            );
        }
        io.hint_pos += nbytes;
    }
}

/// REVEAL: capture public output bytes in host state. Cost corrections handled in C.
pub extern "C" fn host_reveal(ctx: *mut c_void, src_val: u32, ptr: u32, offset: u32) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState) };
    let start = ptr as usize + offset as usize;
    let end = start + 4;
    if io.public_values.len() < end {
        io.public_values.resize(end, 0);
    }
    io.public_values[start..end].copy_from_slice(&src_val.to_le_bytes());
}

/// HINT_STREAM_SET: replace the hint stream contents (used by extension phantoms).
///
/// # Safety
///
/// `ctx` must be a valid `OpenVmIoState` pointer. `data` must point to `len` bytes (or be null).
pub unsafe extern "C" fn host_hint_stream_set(ctx: *mut c_void, data: *const u8, len: u32) {
    let io = &mut *(ctx as *mut OpenVmIoState);
    if len > 0 && !data.is_null() {
        io.hint_stream = std::slice::from_raw_parts(data, len as usize).to_vec();
    } else {
        io.hint_stream = Vec::new();
    }
    io.hint_pos = 0;
}

// ── Deferral callbacks ─────────────────────────────────────────────────────

/// Deferral CALL lookup: find output_key for (def_idx, input_commit).
///
/// Returns 1 on success (output_key_out written), 0 on miss.
///
/// # Safety
///
/// `input_commit` must point to `DEFERRAL_COMMIT_NUM_BYTES` readable bytes.
/// `output_key_out` must point to `DEFERRAL_OUTPUT_KEY_BYTES` writable bytes.
pub unsafe extern "C" fn host_deferral_call_lookup(
    ctx: *mut c_void,
    def_idx: u32,
    input_commit: *const u8,
    output_key_out: *mut u8,
) -> i32 {
    let io = &*(ctx as *const OpenVmIoState);
    let ic: [u8; DEFERRAL_COMMIT_NUM_BYTES] =
        std::slice::from_raw_parts(input_commit, DEFERRAL_COMMIT_NUM_BYTES)
            .try_into()
            .unwrap();
    if let Some(ok) = io.deferral.call_entries.get(&(def_idx, ic)) {
        std::ptr::copy_nonoverlapping(ok.as_ptr(), output_key_out, DEFERRAL_OUTPUT_KEY_BYTES);
        1
    } else {
        0
    }
}

/// Deferral OUTPUT lookup: find raw output for output_commit.
///
/// Returns 1 on success (output_raw_out written), 0 on miss.
///
/// # Safety
///
/// `output_commit` must point to `DEFERRAL_COMMIT_NUM_BYTES` readable bytes.
/// `output_raw_out` must point to at least `expected_len` writable bytes.
pub unsafe extern "C" fn host_deferral_output_lookup(
    ctx: *mut c_void,
    output_commit: *const u8,
    output_raw_out: *mut u8,
    expected_len: u32,
) -> i32 {
    let io = &*(ctx as *const OpenVmIoState);
    let oc: [u8; DEFERRAL_COMMIT_NUM_BYTES] =
        std::slice::from_raw_parts(output_commit, DEFERRAL_COMMIT_NUM_BYTES)
            .try_into()
            .unwrap();
    if let Some(raw) = io.deferral.output_entries.get(&oc) {
        debug_assert_eq!(raw.len(), expected_len as usize);
        std::ptr::copy_nonoverlapping(raw.as_ptr(), output_raw_out, raw.len());
        1
    } else {
        0
    }
}

// ── Conversion helpers ──────────────────────────────────────────────────────

/// Convert an OpenVM field-element input stream to byte-packed vectors.
pub fn convert_input_stream<F: PrimeField32>(stream: &VecDeque<Vec<F>>) -> VecDeque<Vec<u8>> {
    stream
        .iter()
        .map(|vec| vec.iter().map(|f| f.as_canonical_u32() as u8).collect())
        .collect()
}
