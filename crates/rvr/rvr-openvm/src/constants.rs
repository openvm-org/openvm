//! Compile-time constants emitted into generated native tracer headers.
//!
//! The buffer-capacity constants here are the single source of truth for
//! both the C tracer struct layout and the Rust-side `SegmentationState`
//! buffer allocations — they must match byte-for-byte.

use openvm_instructions::{
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    DEFAULT_SEGMENT_CHECK_INSNS, DEFERRAL_AS, DIGEST_WIDTH, MEMORY_PAGE_BITS, PUBLIC_VALUES_AS,
};
use openvm_platform::{memory::MEM_SIZE, WORD_SIZE};
use openvm_riscv_guest::MAX_HINT_BUFFER_DWORDS;
use rvr_openvm_ext_ffi_common::{
    PREFLIGHT_ADDSUB_RECORD_SIZE, PREFLIGHT_BRANCH2_RECORD_SIZE,
    PREFLIGHT_CHIP_RECORD_BUF_ALIGN, PREFLIGHT_CHIP_RECORD_BUF_SIZE,
    PREFLIGHT_DELTA_MEMORY_LOG_ENTRY_ALIGN, PREFLIGHT_DELTA_MEMORY_LOG_ENTRY_SIZE,
    PREFLIGHT_DELTA_RECORD_SIZE, PREFLIGHT_DEVICE_PROGRAM_ENTRY_ALIGN,
    PREFLIGHT_DEVICE_PROGRAM_ENTRY_SIZE, PREFLIGHT_INITIAL_TIMESTAMP, PREFLIGHT_MEMORY_KIND_READ,
    PREFLIGHT_MEMORY_KIND_TOUCH, PREFLIGHT_MEMORY_KIND_WRITE, PREFLIGHT_MEMORY_LOG_ENTRY_ALIGN,
    PREFLIGHT_MEMORY_LOG_ENTRY_SIZE, PREFLIGHT_PROGRAM_LOG_ENTRY_ALIGN,
    PREFLIGHT_PROGRAM_LOG_ENTRY_SIZE, PREFLIGHT_PROGRAM_RUN_ENTRY_ALIGN,
    PREFLIGHT_PROGRAM_RUN_ENTRY_SIZE, PREFLIGHT_RW1_RECORD_SIZE, PREFLIGHT_TOUCHED_BLOCK_ALIGN,
    PREFLIGHT_TOUCHED_BLOCK_SIZE, PREFLIGHT_TRACER_DATA_ALIGN, PREFLIGHT_TRACER_DATA_SIZE,
    PREFLIGHT_TRACER_KIND, PREFLIGHT_WR1_RECORD_SIZE,
};

const BYTE_SPACE_PTRS_PER_LEAF: usize = core::mem::size_of::<u16>() * DIGEST_WIDTH;
const DEFERRAL_PTRS_PER_LEAF: usize = DIGEST_WIDTH;

/// Worst-case AS_MEMORY pages a single instruction can touch.
///
/// Bound is set by `HINT_BUFFER`, which writes up to
/// `MAX_HINT_BUFFER_DWORDS * WORD_SIZE` contiguous bytes. One AS_MEMORY page
/// covers `BYTE_SPACE_PTRS_PER_LEAF * 2^MEMORY_PAGE_BITS` bytes. The `+1` covers worst-case
/// misalignment of the range across page boundaries.
pub const MAX_MEM_PAGES_PER_INSN: usize = {
    let page_bytes = BYTE_SPACE_PTRS_PER_LEAF * (1 << MEMORY_PAGE_BITS);
    let max_bytes = MAX_HINT_BUFFER_DWORDS * WORD_SIZE;
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
static constexpr uint32_t DEFERRAL_DIGEST_SIZE = {DIGEST_WIDTH};
static constexpr uint64_t RV_TEXT_START = 0x{text_start:08x}ull;
static constexpr uint64_t RV_TEXT_END = 0x{text_end:08x}ull;
static constexpr uint32_t RV_DISPATCH_TABLE_SIZE = {dispatch_table_size}u;
static constexpr uint32_t TRACER_BYTE_SPACE_PTRS_PER_LEAF_BITS = {byte_space_ptrs_per_leaf_bits};
static constexpr uint32_t TRACER_DEFERRAL_PTRS_PER_LEAF_BITS = {deferral_ptrs_per_leaf_bits};
static constexpr uint32_t TRACER_PAGE_BITS = {MEMORY_PAGE_BITS};
static constexpr uint32_t TRACER_MEM_PAGE_BUF_CAP = {MEM_PAGE_BUF_CAP};
static constexpr uint32_t TRACER_PV_PAGE_BUF_CAP = {PV_PAGE_BUF_CAP};
static constexpr uint32_t TRACER_DEFERRAL_PAGE_BUF_CAP = {DEFERRAL_PAGE_BUF_CAP};
static constexpr uint32_t TRACER_SEGMENT_CHECK_INSNS = {DEFAULT_SEGMENT_CHECK_INSNS};
static constexpr uint32_t TRACER_MAX_MEM_PAGES_PER_INSN = {MAX_MEM_PAGES_PER_INSN};
static constexpr uint32_t PREFLIGHT_TRACER_KIND = {PREFLIGHT_TRACER_KIND};
static constexpr uint32_t PREFLIGHT_INITIAL_TIMESTAMP = {PREFLIGHT_INITIAL_TIMESTAMP};
static constexpr uint32_t PREFLIGHT_MEMORY_KIND_READ = {PREFLIGHT_MEMORY_KIND_READ};
static constexpr uint32_t PREFLIGHT_MEMORY_KIND_WRITE = {PREFLIGHT_MEMORY_KIND_WRITE};
static constexpr uint32_t PREFLIGHT_MEMORY_KIND_TOUCH = {PREFLIGHT_MEMORY_KIND_TOUCH};
static constexpr uint32_t PREFLIGHT_PROGRAM_LOG_ENTRY_SIZE = {PREFLIGHT_PROGRAM_LOG_ENTRY_SIZE};
static constexpr uint32_t PREFLIGHT_PROGRAM_LOG_ENTRY_ALIGN = {PREFLIGHT_PROGRAM_LOG_ENTRY_ALIGN};
static constexpr uint32_t PREFLIGHT_PROGRAM_RUN_ENTRY_SIZE = {PREFLIGHT_PROGRAM_RUN_ENTRY_SIZE};
static constexpr uint32_t PREFLIGHT_PROGRAM_RUN_ENTRY_ALIGN = {PREFLIGHT_PROGRAM_RUN_ENTRY_ALIGN};
static constexpr uint32_t PREFLIGHT_DEVICE_PROGRAM_ENTRY_SIZE = {PREFLIGHT_DEVICE_PROGRAM_ENTRY_SIZE};
static constexpr uint32_t PREFLIGHT_DEVICE_PROGRAM_ENTRY_ALIGN = {PREFLIGHT_DEVICE_PROGRAM_ENTRY_ALIGN};
static constexpr uint32_t PREFLIGHT_MEMORY_LOG_ENTRY_SIZE = {PREFLIGHT_MEMORY_LOG_ENTRY_SIZE};
static constexpr uint32_t PREFLIGHT_MEMORY_LOG_ENTRY_ALIGN = {PREFLIGHT_MEMORY_LOG_ENTRY_ALIGN};
static constexpr uint32_t PREFLIGHT_DELTA_MEMORY_LOG_ENTRY_SIZE = {PREFLIGHT_DELTA_MEMORY_LOG_ENTRY_SIZE};
static constexpr uint32_t PREFLIGHT_DELTA_MEMORY_LOG_ENTRY_ALIGN = {PREFLIGHT_DELTA_MEMORY_LOG_ENTRY_ALIGN};
static constexpr uint32_t PREFLIGHT_TRACER_DATA_SIZE = {PREFLIGHT_TRACER_DATA_SIZE};
static constexpr uint32_t PREFLIGHT_TRACER_DATA_ALIGN = {PREFLIGHT_TRACER_DATA_ALIGN};
static constexpr uint32_t PREFLIGHT_TOUCHED_BLOCK_SIZE = {PREFLIGHT_TOUCHED_BLOCK_SIZE};
static constexpr uint32_t PREFLIGHT_TOUCHED_BLOCK_ALIGN = {PREFLIGHT_TOUCHED_BLOCK_ALIGN};
static constexpr uint32_t PREFLIGHT_CHIP_RECORD_BUF_SIZE = {PREFLIGHT_CHIP_RECORD_BUF_SIZE};
static constexpr uint32_t PREFLIGHT_CHIP_RECORD_BUF_ALIGN = {PREFLIGHT_CHIP_RECORD_BUF_ALIGN};
static constexpr uint32_t PREFLIGHT_ADDSUB_RECORD_SIZE = {PREFLIGHT_ADDSUB_RECORD_SIZE};
static constexpr uint32_t PREFLIGHT_BRANCH2_RECORD_SIZE = {PREFLIGHT_BRANCH2_RECORD_SIZE};
static constexpr uint32_t PREFLIGHT_WR1_RECORD_SIZE = {PREFLIGHT_WR1_RECORD_SIZE};
static constexpr uint32_t PREFLIGHT_RW1_RECORD_SIZE = {PREFLIGHT_RW1_RECORD_SIZE};
static constexpr uint32_t PREFLIGHT_DELTA_RECORD_SIZE = {PREFLIGHT_DELTA_RECORD_SIZE};
"
    )
}
