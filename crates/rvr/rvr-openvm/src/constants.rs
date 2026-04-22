//! Compile-time constants emitted into generated native tracer headers.
//!
//! The buffer-capacity constants here are the single source of truth for
//! both the C tracer struct layout and the Rust-side `SegmentationState`
//! buffer allocations — they must match byte-for-byte.

use rvr_openvm_ext_ffi_common::{CHUNK, DEFAULT_PAGE_BITS, DEFAULT_SEGMENT_CHECK_INSNS, MEM_BITS};

/// Maximum AS_MEMORY page buffer entries per segment check interval.
///
/// **No bounds checks in C — capacity must be sufficient.**
///
/// Flushed at most every 2 × `DEFAULT_SEGMENT_CHECK_INSNS` instructions
/// (block-granular check can overshoot by up to one block, which is at
/// most `DEFAULT_SEGMENT_CHECK_INSNS` instructions).
/// Worst-case unique pages per instruction: ~10 (ECC setup / HINT_BUFFER).
/// 2000 insns × 10 pages = 20 000 — well under 65 536.
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
pub fn constants_header(_text_start: u32) -> String {
    let memory_mask = (1u64 << MEM_BITS) - 1;
    let chunk_bits = CHUNK.ilog2();

    format!(
        "\
#pragma once
#include <stdint.h>

static constexpr uint32_t MEMORY_MASK = 0x{memory_mask:x}u;
static constexpr uint32_t TRACER_CHUNK_BITS = {chunk_bits};
static constexpr uint32_t TRACER_PAGE_BITS = {DEFAULT_PAGE_BITS};
static constexpr uint32_t TRACER_MEM_PAGE_BUF_CAP = {MEM_PAGE_BUF_CAP};
static constexpr uint32_t TRACER_PV_PAGE_BUF_CAP = {PV_PAGE_BUF_CAP};
static constexpr uint32_t TRACER_DEFERRAL_PAGE_BUF_CAP = {DEFERRAL_PAGE_BUF_CAP};
static constexpr uint32_t TRACER_SEGMENT_CHECK_INSNS = {DEFAULT_SEGMENT_CHECK_INSNS};
"
    )
}
