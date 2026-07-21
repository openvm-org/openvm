//! FFI declarations for the C tracing wrapper functions.
//!
//! Extension staticlibs call these non-inline wrappers (defined in
//! `rvr_ext_wrappers.c`) for traced memory access and chip cost tracking.
//! Register access stays in generated C; resolved values are passed to extension FFI entry points.
//!
//! The `state` parameter is always an opaque `*mut c_void` pointing
//! to the C `RvState` struct.

use std::ffi::c_void;

extern "C" {
    // ── Memory access (single u64 word, data) ─────────────────────────
    pub fn rd_mem_u64_wrapper(state: *mut c_void, addr: u64) -> u64;

    // ── Memory access (u64-word ranges, data) ───────────────────────
    pub fn rd_mem_u64_range_wrapper(
        state: *mut c_void,
        base_addr: u64,
        out: *mut u64,
        num_words: u32,
    );
    pub fn wr_mem_u64_range_wrapper(
        state: *mut c_void,
        base_addr: u64,
        vals: *const u64,
        num_words: u32,
    );
    pub fn trace_mem_access_u64_range_wrapper(
        state: *mut c_void,
        base_addr: u64,
        num_words: u32,
        addr_space: u32,
    );

    // ── Memory access (u64-word ranges, trace-only) ───────────────────
    pub fn trace_rd_mem_u64_range_wrapper(
        state: *mut c_void,
        base_addr: u64,
        vals: *const u64,
        num_words: u32,
    );
    pub fn trace_wr_mem_u64_range_wrapper(
        state: *mut c_void,
        base_addr: u64,
        vals: *const u64,
        num_words: u32,
    );

    // ── Chip cost ─────────────────────────────────────────────────────
    pub fn trace_chip_wrapper(state: *mut c_void, chip_idx: u32, count: u32);

    // ── Hint stream (for extension phantom instructions) ──────────────
    /// Replace the hint stream contents. Forwarded through `openvm_io.c`
    /// to `OpenVmIoState.hint_stream` via the host callback mechanism.
    pub fn ext_hint_stream_set(data: *const u8, len: u64);
}

// ── Zero-copy lane reinterpretation ─────────────────────────────────────

// We assume a little-endian host (x86_64, aarch64). The u64→u32 reinterpret
// helper below depends on that.
const _: () = assert!(
    cfg!(target_endian = "little"),
    "rvr-openvm-ext-ffi-common assumes a little-endian host"
);

/// Reinterpret a `[u64]` slice as a `[u32]` slice (twice as many elements),
/// mutably. Zero-copy on LE: the low u32 of each u64 comes first, then the
/// high u32. Mutating through the returned slice mutates the underlying u64
/// lanes.
#[inline(always)]
pub fn u64s_as_u32s_mut(lanes: &mut [u64]) -> &mut [u32] {
    let len = lanes.len() * 2;
    // SAFETY: u64 alignment (8) >= u32 alignment (4); total bytes match;
    // POD types; LE host (asserted above) makes the layout (lo, hi).
    unsafe { std::slice::from_raw_parts_mut(lanes.as_mut_ptr().cast::<u32>(), len) }
}

// ── Ergonomic batched memory helpers ─────────────────────────────────────

/// Read `out.len()` consecutive u64 words from guest memory into `out`,
/// then trace each as a memory read.
///
/// `out` must be non-empty; the C trace functions assume `num_words >= 1`
/// to avoid an underflow check on the hot path. Callers handling
/// guest-supplied dynamic sizes (e.g. xorin with `len = 0`) must guard
/// the empty case themselves.
///
/// # Safety
/// `state` must be a valid `RvState` pointer; `out.len()` must fit in `u32`, and
/// the byte range `[base_addr, base_addr + out.len() * WORD_SIZE)` must lie within guest memory.
pub unsafe fn rd_mem_words_traced(state: *mut c_void, base_addr: u64, out: &mut [u64]) {
    debug_assert!(
        !out.is_empty(),
        "rd_mem_words_traced requires a non-empty range"
    );
    debug_assert!(u32::try_from(out.len()).is_ok());
    let n = out.len() as u32;
    rd_mem_u64_range_wrapper(state, base_addr, out.as_mut_ptr(), n);
    trace_rd_mem_u64_range_wrapper(state, base_addr, out.as_ptr(), n);
}

/// Trace each value in `vals` as a write to guest memory, then write them.
/// Trace-before-write so future tracers can read the old value before it is
/// overwritten.
///
/// `vals` must be non-empty; see [`rd_mem_words_traced`] for rationale.
///
/// # Safety
/// `state` must be a valid `RvState` pointer; `vals.len()` must fit in `u32`, and
/// the byte range `[base_addr, base_addr + vals.len() * WORD_SIZE)` must lie within guest memory.
pub unsafe fn wr_mem_words_traced(state: *mut c_void, base_addr: u64, vals: &[u64]) {
    debug_assert!(
        !vals.is_empty(),
        "wr_mem_words_traced requires a non-empty range"
    );
    debug_assert!(u32::try_from(vals.len()).is_ok());
    let n = vals.len() as u32;
    trace_wr_mem_u64_range_wrapper(state, base_addr, vals.as_ptr(), n);
    wr_mem_u64_range_wrapper(state, base_addr, vals.as_ptr(), n);
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
    base_addr: u64,
    num_words: u32,
    addr_space: u32,
) {
    debug_assert!(
        num_words >= 1,
        "trace_mem_access_range requires num_words >= 1"
    );
    trace_mem_access_u64_range_wrapper(state, base_addr, num_words, addr_space);
}
