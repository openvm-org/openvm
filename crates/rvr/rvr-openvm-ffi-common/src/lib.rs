//! Memory and host-I/O helpers shared by Rust RVR extensions.
//!
//! Extension staticlibs call the C functions defined in `rvr_ext_wrappers.c`
//! for execution-mode-specific memory access.
//! Register access stays in generated C; resolved values are passed to extension FFI entry points.
//!
//! The `state` parameter is always an opaque `*mut c_void` pointing
//! to the C `RvState` struct.

use std::ffi::c_void;

extern "C" {
    fn read_mem_u64_execution_input_wrapper(state: *mut c_void, addr: u64) -> u64;
    fn read_mem_u64_range_execution_input_wrapper(
        state: *mut c_void,
        base_addr: u64,
        out: *mut u64,
        num_words: u32,
    );
    fn read_mem_u64_range_wrapper(
        state: *mut c_void,
        base_addr: u64,
        out: *mut u64,
        num_words: u32,
    );
    fn write_mem_u64_range_wrapper(
        state: *mut c_void,
        base_addr: u64,
        vals: *const u64,
        num_words: u32,
    );
    fn record_page_access_u64_range_wrapper(
        state: *mut c_void,
        base_addr: u64,
        num_words: u32,
        addr_space: u32,
    );

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

/// View u64 words as their little-endian bytes without copying.
#[inline(always)]
pub fn u64s_as_bytes(words: &[u64]) -> &[u8] {
    let len = core::mem::size_of_val(words);
    // SAFETY: u8 has alignment 1, the byte length matches the source slice,
    // and the supported little-endian host layout matches guest word order.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), len) }
}

/// View mutable u64 words as their little-endian bytes without copying.
#[inline(always)]
pub fn u64s_as_bytes_mut(words: &mut [u64]) -> &mut [u8] {
    let len = core::mem::size_of_val(words);
    // SAFETY: the same layout argument as [`u64s_as_bytes`] applies, and the
    // mutable borrow keeps the returned byte slice exclusive.
    unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), len) }
}

// ── Ergonomic batched memory helpers ─────────────────────────────────────

#[inline(always)]
fn memory_word_count(len: usize) -> u32 {
    debug_assert!(len > 0, "memory word range must be non-empty");
    debug_assert!(u32::try_from(len).is_ok());
    len as u32
}

/// Read `out.len()` consecutive u64 words as VM memory accesses.
///
/// `out` must be non-empty; the mode-specific functions assume `num_words >= 1`
/// to avoid an underflow check on the hot path. Callers handling
/// guest-supplied dynamic sizes (e.g. xorin with `len = 0`) must guard
/// the empty case themselves.
///
/// # Safety
///
/// `state` must point to a valid `RvState`. `out.len()` must fit in `u32`, and
/// `[base_addr, base_addr + out.len() * WORD_SIZE)` must fit in guest memory.
pub unsafe fn read_mem_words(state: *mut c_void, base_addr: u64, out: &mut [u64]) {
    let num_words = memory_word_count(out.len());
    read_mem_u64_range_wrapper(state, base_addr, out.as_mut_ptr(), num_words);
}

/// Write `vals` as VM memory accesses.
///
/// `vals` must be non-empty; see [`read_mem_words`] for rationale.
///
/// # Safety
///
/// `state` must point to a valid `RvState`. `vals.len()` must fit in `u32`, and
/// `[base_addr, base_addr + vals.len() * WORD_SIZE)` must fit in guest memory.
pub unsafe fn write_mem_words(state: *mut c_void, base_addr: u64, vals: &[u64]) {
    let num_words = memory_word_count(vals.len());
    write_mem_u64_range_wrapper(state, base_addr, vals.as_ptr(), num_words);
}

/// Read one u64 whose value affects execution but is not a VM memory access.
///
/// # Safety
/// `state` must point to a valid `RvState`, and `addr` must address one u64 in
/// guest memory.
pub unsafe fn read_mem_u64_execution_input(state: *mut c_void, addr: u64) -> u64 {
    read_mem_u64_execution_input_wrapper(state, addr)
}

/// Read guest words whose values affect execution but are not VM memory
/// accesses.
///
/// # Safety
/// `state` must point to a valid `RvState`. `out` must be non-empty, and its
/// guest-memory range must be valid.
pub unsafe fn read_mem_words_execution_input(state: *mut c_void, base_addr: u64, out: &mut [u64]) {
    let num_words = memory_word_count(out.len());
    read_mem_u64_range_execution_input_wrapper(state, base_addr, out.as_mut_ptr(), num_words);
}

/// Record the pages touched by `num_words` consecutive words in `addr_space`.
/// This is metering data only; it does not record the accessed values.
///
/// `num_words` must be `>= 1`; see [`read_mem_words`] for rationale.
///
/// # Safety
/// `state` must be a valid `RvState` pointer.
pub unsafe fn record_page_access_range(
    state: *mut c_void,
    base_addr: u64,
    num_words: u32,
    addr_space: u32,
) {
    debug_assert!(
        num_words >= 1,
        "record_page_access_range requires num_words >= 1"
    );
    record_page_access_u64_range_wrapper(state, base_addr, num_words, addr_space);
}
