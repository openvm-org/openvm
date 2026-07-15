//! Compile-time constants emitted into generated native tracer headers.
//!
//! The buffer-capacity constants here are the single source of truth for
//! both the C tracer struct layout and the Rust-side `SegmentationState`
//! buffer allocations — they must match byte-for-byte.

use openvm_instructions::{
    metering::{DEFAULT_SEGMENT_CHECK_INSNS, MAX_METERED_BLOCK_INSNS, PAGE_MASK_LEAF_BITS},
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    DEFERRAL_AS, MEMORY_DIGEST_WIDTH, PUBLIC_VALUES_AS,
};
use openvm_platform::{memory::MEM_SIZE, WORD_SIZE};
use openvm_riscv_guest::MAX_HINT_BUFFER_DWORDS;

const BYTE_SPACE_PTRS_PER_LEAF: usize = core::mem::size_of::<u16>() * MEMORY_DIGEST_WIDTH;
const DEFERRAL_PTRS_PER_LEAF: usize = MEMORY_DIGEST_WIDTH;

/// Worst-case AS_MEMORY pages a single instruction can touch.
///
/// Bound is set by `HINT_BUFFER`, which writes up to
/// `MAX_HINT_BUFFER_DWORDS * WORD_SIZE` contiguous bytes. One AS_MEMORY page
/// covers `BYTE_SPACE_PTRS_PER_LEAF * 2^PAGE_MASK_LEAF_BITS` bytes. The `+1`
/// covers worst-case misalignment of the range across page boundaries.
pub const MAX_MEM_PAGES_PER_INSN: usize = {
    let page_bytes = BYTE_SPACE_PTRS_PER_LEAF * (1 << PAGE_MASK_LEAF_BITS);
    let max_bytes = MAX_HINT_BUFFER_DWORDS * WORD_SIZE;
    max_bytes.div_ceil(page_bytes) + 1
};

/// Maximum AS_MEMORY page buffer entries per segment check interval.
///
/// **No bounds checks in C — capacity must be sufficient.**
///
/// Flushed after at most `DEFAULT_SEGMENT_CHECK_INSNS + MAX_METERED_BLOCK_INSNS`
/// instructions because a block-granular check can overshoot its interval by
/// one complete generated block.
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
pub fn constants_header(text_start: u64, text_end: u64, dispatch_table_size: usize) -> String {
    let memory_mask = MEM_SIZE as u64 - 1;
    let byte_space_ptrs_per_leaf_bits = BYTE_SPACE_PTRS_PER_LEAF.ilog2();
    let deferral_ptrs_per_leaf_bits = DEFERRAL_PTRS_PER_LEAF.ilog2();

    format!(
        "\
#pragma once
#include <stdint.h>

static constexpr uint64_t MEMORY_MASK = 0x{memory_mask:x}ull;
static constexpr uint32_t AS_REGISTER = {RV64_REGISTER_AS};
static constexpr uint32_t AS_MEMORY = {RV64_MEMORY_AS};
static constexpr uint32_t AS_PUBLIC_VALUES = {PUBLIC_VALUES_AS};
static constexpr uint32_t AS_DEFERRAL = {DEFERRAL_AS};
static constexpr uint32_t WORD_SIZE = {WORD_SIZE};
static constexpr uint32_t DEFERRAL_DIGEST_SIZE = {MEMORY_DIGEST_WIDTH};
static constexpr uint64_t RV_TEXT_START = 0x{text_start:08x}ull;
static constexpr uint64_t RV_TEXT_END = 0x{text_end:08x}ull;
static constexpr uint32_t RV_DISPATCH_TABLE_SIZE = {dispatch_table_size}u;
static constexpr uint32_t TRACER_BYTE_SPACE_PTRS_PER_LEAF_BITS = {byte_space_ptrs_per_leaf_bits};
static constexpr uint32_t TRACER_DEFERRAL_PTRS_PER_LEAF_BITS = {deferral_ptrs_per_leaf_bits};
static constexpr uint32_t TRACER_PAGE_BITS = {PAGE_MASK_LEAF_BITS};
static constexpr uint32_t TRACER_MEM_PAGE_BUF_CAP = {MEM_PAGE_BUF_CAP};
static constexpr uint32_t TRACER_PV_PAGE_BUF_CAP = {PV_PAGE_BUF_CAP};
static constexpr uint32_t TRACER_DEFERRAL_PAGE_BUF_CAP = {DEFERRAL_PAGE_BUF_CAP};
static constexpr uint32_t TRACER_SEGMENT_CHECK_INSNS = {DEFAULT_SEGMENT_CHECK_INSNS};
static constexpr uint32_t TRACER_MAX_BLOCK_INSNS = {MAX_METERED_BLOCK_INSNS};
static constexpr uint32_t TRACER_MAX_MEM_PAGES_PER_INSN = {MAX_MEM_PAGES_PER_INSN};
"
    )
}
