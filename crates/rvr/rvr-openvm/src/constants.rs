//! Compile-time constants emitted into generated native tracer headers.
//!
//! The buffer-capacity constants here are the single source of truth for
//! both the C tracer struct layout and the Rust-side `SegmentationState`
//! buffer allocations — they must match byte-for-byte.

use openvm_platform::memory::MEM_SIZE;
use openvm_rv32im_guest::MAX_HINT_BUFFER_WORDS;
use rvr_openvm_ext_ffi_common::{
    AS_MEMORY, AS_PUBLIC_VALUES, AS_REGISTER, CHUNK, DEFAULT_PAGE_BITS,
    DEFAULT_SEGMENT_CHECK_INSNS, DEFERRAL_AS, WORD_SIZE,
};

/// Worst-case AS_MEMORY pages a single instruction can touch.
///
/// Bound is set by `HINT_BUFFER`, which writes up to
/// `MAX_HINT_BUFFER_WORDS * WORD_SIZE` contiguous bytes. One AS_MEMORY page
/// covers `CHUNK * 2^PAGE_BITS` bytes. The `+1` covers worst-case
/// misalignment of the range across page boundaries.
pub const MAX_MEM_PAGES_PER_INSN: usize = {
    let page_bytes = CHUNK * (1 << DEFAULT_PAGE_BITS);
    let max_bytes = MAX_HINT_BUFFER_WORDS * WORD_SIZE;
    max_bytes.div_ceil(page_bytes) + 1
};

/// Maximum AS_MEMORY page buffer entries per segment check interval.
///
/// **No bounds checks in C — capacity must be sufficient.**
///
/// Flushed at most every 2 × `DEFAULT_SEGMENT_CHECK_INSNS` instructions
/// (block-granular check can overshoot by up to one block, which is at
/// most `DEFAULT_SEGMENT_CHECK_INSNS` instructions).
/// Worst-case unique pages per instruction: ~10 (ECC setup / HINT_BUFFER, HINT_BUFFER is taken as
/// worst-case). 2000 insns × 10 pages = 20 000 — well under 65 536.
pub const MEM_PAGE_BUF_CAP: usize = 1 << 16;

/// Maximum AS_PUBLIC_VALUES page buffer entries per segment check interval.
/// No bounds checks in C. At most 1 page per instruction (reveal/publish).
pub const PV_PAGE_BUF_CAP: usize = 1 << 12;

/// Maximum AS_DEFERRAL page buffer entries per segment check interval.
/// No bounds checks in C.
// TODO: justify this bound (audit max deferral pages per instruction).
pub const DEFERRAL_PAGE_BUF_CAP: usize = 1 << 16;

/// Generate the `openvm_constants.h` content with compile-time constants
/// for the C tracer headers.
pub fn constants_header(text_start: u32, text_end: u32, dispatch_table_size: usize) -> String {
    let memory_mask = MEM_SIZE as u64 - 1;
    let chunk_bits = CHUNK.ilog2();

    format!(
        "\
#pragma once
#include <stdint.h>

static constexpr uint32_t MEMORY_MASK = 0x{memory_mask:x}u;
static constexpr uint32_t AS_REGISTER = {AS_REGISTER};
static constexpr uint32_t AS_MEMORY = {AS_MEMORY};
static constexpr uint32_t AS_PUBLIC_VALUES = {AS_PUBLIC_VALUES};
static constexpr uint32_t AS_DEFERRAL = {DEFERRAL_AS};
static constexpr uint32_t WORD_SIZE = {WORD_SIZE};
static constexpr uint32_t RV_TEXT_START = 0x{text_start:08x}u;
static constexpr uint32_t RV_TEXT_END = 0x{text_end:08x}u;
static constexpr uint32_t RV_DISPATCH_TABLE_SIZE = {dispatch_table_size}u;
static constexpr uint32_t TRACER_CHUNK_BITS = {chunk_bits};
static constexpr uint32_t TRACER_PAGE_BITS = {DEFAULT_PAGE_BITS};
static constexpr uint32_t TRACER_MEM_PAGE_BUF_CAP = {MEM_PAGE_BUF_CAP};
static constexpr uint32_t TRACER_PV_PAGE_BUF_CAP = {PV_PAGE_BUF_CAP};
static constexpr uint32_t TRACER_DEFERRAL_PAGE_BUF_CAP = {DEFERRAL_PAGE_BUF_CAP};
static constexpr uint32_t TRACER_SEGMENT_CHECK_INSNS = {DEFAULT_SEGMENT_CHECK_INSNS};
static constexpr uint32_t TRACER_MAX_MEM_PAGES_PER_INSN = {MAX_MEM_PAGES_PER_INSN};
"
    )
}
