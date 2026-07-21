//! Constants shared by Rust metering state and generated C tracers.

use openvm_instructions::{
    metering::{PAGE_MASK_LEAF_BITS, SEGMENT_CHECK_INSNS},
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    DEFERRAL_AS, PUBLIC_VALUES_AS, VM_DIGEST_WIDTH,
};
use openvm_platform::{memory::MEM_SIZE, WORD_SIZE};
use openvm_riscv_guest::MAX_HINT_BUFFER_DWORDS;

const BYTE_SPACE_PTRS_PER_LEAF: usize = core::mem::size_of::<u16>() * VM_DIGEST_WIDTH;
const DEFERRAL_PTRS_PER_LEAF: usize = VM_DIGEST_WIDTH;

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
/// The C tracer flushes before crossing `SEGMENT_CHECK_INSNS` and statically
/// verifies this capacity against `MAX_MEM_PAGES_PER_INSN`.
pub const MEM_PAGE_BUF_CAP: usize = 1 << 16;

/// Worst-case AS_PUBLIC_VALUES pages a fixed-width reveal can touch.
const MAX_PV_PAGES_PER_INSN: usize = 2;

/// Maximum AS_PUBLIC_VALUES page buffer entries per segment check interval.
/// The C tracer does not bounds-check this buffer. A reveal can span two pages.
pub const PV_PAGE_BUF_CAP: usize = 1 << 12;

/// Maximum AS_DEFERRAL page buffer entries per segment check interval.
/// No bounds checks in C. Deferral CALL records two reads and two writes.
const MAX_DEFERRAL_PAGES_PER_INSN: usize = 4;
pub const DEFERRAL_PAGE_BUF_CAP: usize = 1 << 12;

/// Generate the `openvm_constants.h` content with compile-time constants
/// for the C tracing headers.
pub fn constants_header(
    text_start: u64,
    text_end: u64,
    dispatch_table_size: usize,
    num_airs: Option<u32>,
) -> String {
    let memory_mask = MEM_SIZE as u64 - 1;
    let byte_space_ptrs_per_leaf_bits = BYTE_SPACE_PTRS_PER_LEAF.ilog2();
    let deferral_ptrs_per_leaf_bits = DEFERRAL_PTRS_PER_LEAF.ilog2();

    let mut header = format!(
        "\
#pragma once
#include <stdint.h>

static constexpr uint64_t MEMORY_MASK = 0x{memory_mask:x}ull;
static constexpr uint32_t AS_REGISTER = {RV64_REGISTER_AS};
static constexpr uint32_t AS_MEMORY = {RV64_MEMORY_AS};
static constexpr uint32_t AS_PUBLIC_VALUES = {PUBLIC_VALUES_AS};
static constexpr uint32_t AS_DEFERRAL = {DEFERRAL_AS};
static constexpr uint32_t WORD_SIZE = {WORD_SIZE};
static_assert(WORD_SIZE == sizeof(uint64_t), \"OpenVM word size must match uint64_t\");
static constexpr uint32_t DEFERRAL_DIGEST_SIZE = {VM_DIGEST_WIDTH};
static constexpr uint64_t RV_TEXT_START = 0x{text_start:08x}ull;
static constexpr uint64_t RV_TEXT_END = 0x{text_end:08x}ull;
static constexpr uint32_t RV_DISPATCH_TABLE_SIZE = {dispatch_table_size}u;
static constexpr uint32_t TRACER_BYTE_SPACE_PTRS_PER_LEAF_BITS = {byte_space_ptrs_per_leaf_bits};
static constexpr uint32_t TRACER_DEFERRAL_PTRS_PER_LEAF_BITS = {deferral_ptrs_per_leaf_bits};
static constexpr uint32_t TRACER_PAGE_BITS = {PAGE_MASK_LEAF_BITS};
static constexpr uint32_t TRACER_MEM_PAGE_BUF_CAP = {MEM_PAGE_BUF_CAP};
static constexpr uint32_t TRACER_PV_PAGE_BUF_CAP = {PV_PAGE_BUF_CAP};
static constexpr uint32_t TRACER_DEFERRAL_PAGE_BUF_CAP = {DEFERRAL_PAGE_BUF_CAP};
static constexpr uint32_t TRACER_SEGMENT_CHECK_INSNS = {SEGMENT_CHECK_INSNS};
static constexpr uint32_t TRACER_MAX_MEM_PAGES_PER_INSN = {MAX_MEM_PAGES_PER_INSN};
static constexpr uint32_t TRACER_MAX_PV_PAGES_PER_INSN = {MAX_PV_PAGES_PER_INSN};
static constexpr uint32_t TRACER_MAX_DEFERRAL_PAGES_PER_INSN = {MAX_DEFERRAL_PAGES_PER_INSN};
"
    );
    if let Some(num_airs) = num_airs {
        header.push_str(&format!(
            "static constexpr uint32_t RV_NUM_AIRS = {num_airs}u;\n"
        ));
    }
    header
}
