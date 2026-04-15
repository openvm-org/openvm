//! FFI declarations for the C tracing wrapper functions.
//!
//! Extension staticlibs call these non-inline wrappers (defined in
//! `rvr_ext_wrappers.c`) for traced memory access, instruction dispatch,
//! chip cost tracking, and block metering. Register access is handled in
//! the generated C code; resolved register values are passed to extension
//! FFI entry points as parameters.
//!
//! The `state` parameter is always an opaque `*mut c_void` pointing
//! to the C `RvState` struct.

use std::ffi::c_void;

/// Word size for memory access (4 bytes).
pub const WORD_SIZE: usize = 4;

// ── Deferral constants ───────────────────────────────────────────────────

/// BabyBear Poseidon2 digest size in field elements.
pub const DEFERRAL_DIGEST_SIZE: usize = 8;

/// Commit size in bytes (DIGEST_SIZE field elements × 4 bytes each).
pub const DEFERRAL_COMMIT_NUM_BYTES: usize = DEFERRAL_DIGEST_SIZE * WORD_SIZE;

/// Output key size in bytes (commit + u64 length).
pub const DEFERRAL_OUTPUT_KEY_BYTES: usize = DEFERRAL_COMMIT_NUM_BYTES + 8;

// ── OpenVM address space identifiers (mirror C `openvm_state.h`) ──
/// Register file address space.
pub const AS_REGISTER: u32 = 1;
/// Main guest memory address space.
pub const AS_MEMORY: u32 = 2;
/// Public values address space.
pub const AS_PUBLIC_VALUES: u32 = 3;
/// Deferral output address space.
pub const DEFERRAL_AS: u32 = 4;

extern "C" {
    // ── Memory access (single word, data) ─────────────────────────────
    pub fn rd_mem_u32_wrapper(state: *mut c_void, addr: u32) -> u32;
    pub fn wr_mem_u32_wrapper(state: *mut c_void, addr: u32, val: u32);

    // ── Memory access (single word, trace-only) ───────────────────────
    pub fn trace_rd_mem_u32_wrapper(state: *mut c_void, addr: u32, val: u32);
    pub fn trace_wr_mem_u32_wrapper(state: *mut c_void, addr: u32, new_val: u32);
    pub fn trace_mem_access_u32_wrapper(state: *mut c_void, addr: u32, addr_space: u32);

    // ── Memory access (word-aligned ranges, data) ─────────────────────
    pub fn rd_mem_u32_range_wrapper(
        state: *mut c_void,
        base_addr: u32,
        out: *mut u32,
        num_words: u32,
    );
    pub fn wr_mem_u32_range_wrapper(
        state: *mut c_void,
        base_addr: u32,
        vals: *const u32,
        num_words: u32,
    );

    // ── Memory access (word-aligned ranges, trace-only) ───────────────
    pub fn trace_rd_mem_u32_range_wrapper(
        state: *mut c_void,
        base_addr: u32,
        vals: *const u32,
        num_words: u32,
    );
    pub fn trace_wr_mem_u32_range_wrapper(
        state: *mut c_void,
        base_addr: u32,
        vals: *const u32,
        num_words: u32,
    );
    pub fn trace_mem_access_u32_range_wrapper(
        state: *mut c_void,
        base_addr: u32,
        num_words: u32,
        addr_space: u32,
    );

    // ── Instruction dispatch / chip cost ──────────────────────────────
    pub fn trace_pc_wrapper(state: *mut c_void, pc: u32);
    pub fn trace_chip_wrapper(state: *mut c_void, chip_idx: u32, count: u32);

    // ── Block metering ────────────────────────────────────────────────
    pub fn trace_block_wrapper(state: *mut c_void, pc: u32, block_insn_count: u32);

    // ── Hint stream (for extension phantom instructions) ──────────────
    /// Replace the hint stream contents. Forwarded through `openvm_io.c`
    /// to `OpenVmIoState.hint_stream` via the host callback mechanism.
    pub fn ext_hint_stream_set(data: *const u8, len: u32);

    // ── Deferral lookup (forwarded to OpenVmIoState.deferral) ────────
    /// Look up deferral CALL output_key by (def_idx, input_commit).
    /// Returns 1 on success (output_key_out written), 0 on miss.
    pub fn ext_deferral_call_lookup(
        def_idx: u32,
        input_commit: *const u8,
        output_key_out: *mut u8,
    ) -> i32;
    /// Look up deferral OUTPUT raw data by output_commit.
    /// Returns 1 on success (output_raw_out written), 0 on miss.
    pub fn ext_deferral_output_lookup(
        output_commit: *const u8,
        output_raw_out: *mut u8,
        expected_len: u32,
    ) -> i32;

}

// ── Zero-copy lane reinterpretation ─────────────────────────────────────

// We assume a little-endian host (x86_64, aarch64). All u64↔u32 reinterpret
// helpers below depend on that.
const _: () = assert!(
    cfg!(target_endian = "little"),
    "rvr-openvm-ext-ffi-common assumes a little-endian host"
);

/// Reinterpret a `[u64]` slice as a `[u32]` slice (twice as many elements).
/// Zero-copy on LE: the low u32 of each u64 comes first, then the high u32.
#[inline(always)]
pub fn u64s_as_u32s(lanes: &[u64]) -> &[u32] {
    let len = lanes.len() * 2;
    // SAFETY: u64 alignment (8) >= u32 alignment (4); total bytes match;
    // POD types; LE host (asserted above) makes the layout (lo, hi).
    unsafe { std::slice::from_raw_parts(lanes.as_ptr().cast::<u32>(), len) }
}

/// Mutable counterpart of [`u64s_as_u32s`]. Mutating through the returned
/// slice mutates the underlying u64 lanes.
#[inline(always)]
pub fn u64s_as_u32s_mut(lanes: &mut [u64]) -> &mut [u32] {
    let len = lanes.len() * 2;
    // SAFETY: see u64s_as_u32s.
    unsafe { std::slice::from_raw_parts_mut(lanes.as_mut_ptr().cast::<u32>(), len) }
}

/// Reinterpret a `[u32]` slice as a `[u8]` slice (4× as many elements).
/// Zero-copy: each u32 contributes its LE bytes in order.
#[inline(always)]
pub fn u32s_as_u8s(words: &[u32]) -> &[u8] {
    let len = words.len() * WORD_SIZE;
    // SAFETY: u32 alignment (4) >= u8 alignment (1); total bytes match;
    // POD; LE host (asserted above).
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), len) }
}

/// Reinterpret a `[u32]` slice as a `[u64]` slice (half as many elements).
/// `words.len()` must be even and the slice must be u64-aligned (a stack
/// `[u32; 2N]` is **not** guaranteed to be 8-aligned; use this only on slices
/// that originated as `[u64]`).
#[inline(always)]
pub fn u32s_as_u64s(words: &[u32]) -> &[u64] {
    debug_assert!(words.len().is_multiple_of(2));
    debug_assert!(
        (words.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u64>()),
        "u32s_as_u64s requires 8-byte aligned input"
    );
    let len = words.len() / 2;
    // SAFETY: alignment debug-asserted; total bytes match; POD; LE host.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u64>(), len) }
}

// ── Ergonomic batched memory helpers ─────────────────────────────────────

/// Read `out.len()` consecutive u32 words from guest memory into `out`,
/// then trace each as a memory read.
///
/// `out` must be non-empty; the C trace functions assume `num_words >= 1`
/// to avoid an underflow check on the hot path. Callers handling
/// guest-supplied dynamic sizes (e.g. xorin with `len = 0`) must guard
/// the empty case themselves.
///
/// # Safety
/// `state` must be a valid `RvState` pointer; the byte range
/// `[base_addr, base_addr + out.len()*WORD_SIZE)` must lie within guest memory.
pub unsafe fn rd_mem_words_traced(state: *mut c_void, base_addr: u32, out: &mut [u32]) {
    debug_assert!(
        !out.is_empty(),
        "rd_mem_words_traced requires a non-empty range"
    );
    let n = out.len() as u32;
    rd_mem_u32_range_wrapper(state, base_addr, out.as_mut_ptr(), n);
    trace_rd_mem_u32_range_wrapper(state, base_addr, out.as_ptr(), n);
}

/// Trace each value in `vals` as a write at `base_addr + i*WORD_SIZE`,
/// then write them to guest memory. Trace-before-write so future tracers
/// can read the old value before it is overwritten.
///
/// `vals` must be non-empty; see [`rd_mem_words_traced`] for rationale.
///
/// # Safety
/// `state` must be a valid `RvState` pointer; the byte range
/// `[base_addr, base_addr + vals.len()*WORD_SIZE)` must lie within guest memory.
pub unsafe fn wr_mem_words_traced(state: *mut c_void, base_addr: u32, vals: &[u32]) {
    debug_assert!(
        !vals.is_empty(),
        "wr_mem_words_traced requires a non-empty range"
    );
    let n = vals.len() as u32;
    trace_wr_mem_u32_range_wrapper(state, base_addr, vals.as_ptr(), n);
    wr_mem_u32_range_wrapper(state, base_addr, vals.as_ptr(), n);
}

/// Trace `num_words` consecutive memory-access notifications (no value)
/// at `addr_space`, one per WORD_SIZE step from `base_addr`.
///
/// `num_words` must be `>= 1`; see [`rd_mem_words_traced`] for rationale.
///
/// # Safety
/// `state` must be a valid `RvState` pointer.
pub unsafe fn trace_mem_access_range(
    state: *mut c_void,
    base_addr: u32,
    num_words: u32,
    addr_space: u32,
) {
    debug_assert!(
        num_words >= 1,
        "trace_mem_access_range requires num_words >= 1"
    );
    trace_mem_access_u32_range_wrapper(state, base_addr, num_words, addr_space);
}
