/* OpenVM metered tracing helpers.
 *
 * Updates per-chip trace_heights matching OpenVM's MeteredCtx.
 *
 * All functions are static inline for zero-overhead inlining.
 */

#ifndef OPENVM_TRACER_METERED_H
#define OPENVM_TRACER_METERED_H

#include <stdint.h>

#include "openvm_state.h"

/* The metering state checks the lengths of these fixed-capacity page buffers,
 * but Clang cannot prove those bounds. */
#pragma clang unsafe_buffer_usage begin

static constexpr uint32_t NO_LAST_PAGE = UINT32_MAX;
static constexpr uint32_t TRACER_BYTES_PER_LEAF =
    1u << TRACER_BYTE_SPACE_PTRS_PER_LEAF_BITS;
static constexpr uint32_t TRACER_LEAF_BYTE_OFFSET_MASK =
    TRACER_BYTES_PER_LEAF - 1u;

static_assert(
    TRACER_MEM_PAGE_BUF_CAP >=
        TRACER_SEGMENT_CHECK_INSNS * TRACER_MAX_MEM_PAGES_PER_INSN,
    "MEM_PAGE_BUF_CAP too small for worst-case pages per flush interval");
static_assert(
    TRACER_PV_PAGE_BUF_CAP >=
        TRACER_SEGMENT_CHECK_INSNS * TRACER_MAX_PV_PAGES_PER_INSN,
    "PV_PAGE_BUF_CAP too small for worst-case pages per flush interval");
static_assert(
    TRACER_DEFERRAL_PAGE_BUF_CAP >=
        TRACER_SEGMENT_CHECK_INSNS * TRACER_MAX_DEFERRAL_PAGES_PER_INSN,
    "DEFERRAL_PAGE_BUF_CAP too small for worst-case pages per flush interval");

typedef struct PageTouch {
  /* Page table index plus a 64-bit leaf mask for that page. */
  uint32_t page_id;
  /* Align leaf_mask and make the shared 16-byte Rust/C layout explicit. */
  uint32_t _padding;
  uint64_t leaf_mask;
} PageTouch;

/* One page buffer per address space stores local page ids and leaf masks.
 * AS_REGISTER pages are handled entirely on the Rust side at init. */
typedef struct TraceMemory {
  uint32_t last_mem_page;
  uint32_t mem_page_buf_len;
  uint64_t last_mem_leaf_mask;
  PageTouch* mem_page_buf;
} TraceMemory;

/* ── Page tracking ─────────────────────────────────────────────────── */

/* Valid VM pointers fit in uint32_t. Check full RV64 operands before calling
 * these helpers. */
static __attribute__((always_inline)) inline uint32_t byte_addr_to_local_leaf(
    uint64_t ptr) {
  return (uint32_t)(ptr >> TRACER_BYTE_SPACE_PTRS_PER_LEAF_BITS);
}

static __attribute__((always_inline)) inline uint32_t
deferral_addr_to_local_leaf(uint64_t ptr) {
  return (uint32_t)(ptr >> TRACER_DEFERRAL_PTRS_PER_LEAF_BITS);
}

static __attribute__((always_inline)) inline uint32_t addr_to_local_leaf(
    uint32_t addr_space, uint64_t ptr) {
  if (likely(addr_space == AS_MEMORY || addr_space == AS_PUBLIC_VALUES)) {
    return byte_addr_to_local_leaf(ptr);
  }
  return deferral_addr_to_local_leaf(ptr);
}

static __attribute__((always_inline)) inline uint64_t leaf_mask(uint32_t leaf) {
  return 1ull << (leaf & ((1u << TRACER_PAGE_BITS) - 1u));
}

static __attribute__((always_inline)) inline uint64_t leaf_mask_range(
    uint32_t first_leaf, uint32_t last_leaf) {
  /* Convert an inclusive leaf range within one page into an occupancy mask. */
  assume(first_leaf <= last_leaf);
  assume((first_leaf >> TRACER_PAGE_BITS) == (last_leaf >> TRACER_PAGE_BITS));
  uint32_t start = first_leaf & ((1u << TRACER_PAGE_BITS) - 1u);
  uint32_t end = last_leaf & ((1u << TRACER_PAGE_BITS) - 1u);
  return (UINT64_MAX << start) &
         (UINT64_MAX >> (((1u << TRACER_PAGE_BITS) - 1u) - end));
}

/* ── Per-address-space page recording ─────────────────────────────── */

static __attribute__((always_inline)) inline void append_page_touch(
    PageTouch* restrict buf, uint32_t* restrict len, uint32_t page,
    uint64_t leaf_mask) {
  PageTouch* slot = &buf[(*len)++];
  slot->page_id = page;
  slot->leaf_mask = leaf_mask;
}

static __attribute__((always_inline)) inline void append_page_touch_range(
    PageTouch* restrict buf, uint32_t* restrict len, uint32_t first_leaf,
    uint32_t last_leaf) {
  uint32_t first_page = first_leaf >> TRACER_PAGE_BITS;
  uint32_t last_page = last_leaf >> TRACER_PAGE_BITS;
  for (uint32_t page = first_page; page <= last_page; page++) {
    uint32_t page_first_leaf = page << TRACER_PAGE_BITS;
    uint32_t page_last_leaf = page_first_leaf + (1u << TRACER_PAGE_BITS) - 1u;
    uint32_t start =
        first_leaf > page_first_leaf ? first_leaf : page_first_leaf;
    uint32_t end = last_leaf < page_last_leaf ? last_leaf : page_last_leaf;
    append_page_touch(buf, len, page, leaf_mask_range(start, end));
  }
}

/* No bounds check — see MEM_PAGE_BUF_CAP in metered.rs. */
static __attribute__((always_inline)) inline void record_mem_page(
    MeteringState* metering, uint32_t page, uint64_t leaf_mask) {
  if (likely(page == metering->last_mem_page)) {
    metering->mem_page_buf[metering->mem_page_buf_len - 1u].leaf_mask |=
        leaf_mask;
    return;
  }
  metering->last_mem_page = page;
  append_page_touch(metering->mem_page_buf, &metering->mem_page_buf_len, page,
                    leaf_mask);
}

static __attribute__((always_inline)) inline void record_mem_page_range(
    MeteringState* metering, uint32_t first_leaf, uint32_t last_leaf) {
  uint32_t first_page = first_leaf >> TRACER_PAGE_BITS;
  uint32_t last_page = last_leaf >> TRACER_PAGE_BITS;
  if (likely(first_page == metering->last_mem_page)) {
    metering->mem_page_buf[metering->mem_page_buf_len - 1u].leaf_mask |=
        leaf_mask_range(first_leaf,
                        first_page == last_page
                            ? last_leaf
                            : ((first_page + 1u) << TRACER_PAGE_BITS) - 1u);
    if (first_page == last_page) {
      return;
    }
    first_page++;
  }
  uint32_t len = metering->mem_page_buf_len;
  for (uint32_t page = first_page; page <= last_page; page++) {
    uint32_t page_first_leaf = page << TRACER_PAGE_BITS;
    uint32_t page_last_leaf = page_first_leaf + (1u << TRACER_PAGE_BITS) - 1u;
    uint32_t start =
        first_leaf > page_first_leaf ? first_leaf : page_first_leaf;
    uint32_t end = last_leaf < page_last_leaf ? last_leaf : page_last_leaf;
    append_page_touch(metering->mem_page_buf, &len, page,
                      leaf_mask_range(start, end));
  }
  metering->mem_page_buf_len = len;
  metering->last_mem_page = last_page;
}

/* No bounds check — see PV_PAGE_BUF_CAP in metered.rs. */
static __attribute__((always_inline)) inline void record_pv_page(
    MeteringState* metering, uint32_t page, uint64_t leaf_mask) {
  append_page_touch(metering->pv_page_buf, &metering->pv_page_buf_len, page,
                    leaf_mask);
}

static __attribute__((always_inline)) inline void record_pv_page_range(
    MeteringState* metering, uint32_t first_leaf, uint32_t last_leaf) {
  uint32_t len = metering->pv_page_buf_len;
  append_page_touch_range(metering->pv_page_buf, &len, first_leaf, last_leaf);
  metering->pv_page_buf_len = len;
}

/* No bounds check — see DEFERRAL_PAGE_BUF_CAP in metered.rs. */
static __attribute__((always_inline)) inline void record_deferral_page(
    MeteringState* metering, uint32_t page, uint64_t leaf_mask) {
  append_page_touch(metering->deferral_page_buf,
                    &metering->deferral_page_buf_len, page, leaf_mask);
}

static __attribute__((always_inline)) inline void record_deferral_page_range(
    MeteringState* metering, uint32_t first_leaf, uint32_t last_leaf) {
  uint32_t len = metering->deferral_page_buf_len;
  append_page_touch_range(metering->deferral_page_buf, &len, first_leaf,
                          last_leaf);
  metering->deferral_page_buf_len = len;
}

/* Record a single page access. `addr_space` is a compile-time constant at
 * every direct call site in generated C, so the branches below fold away. */
static __attribute__((always_inline)) inline void record_page(
    MeteringState* metering, uint32_t addr_space, uint64_t ptr, uint32_t size) {
  uint32_t first_leaf = addr_to_local_leaf(addr_space, ptr);
  uint32_t last_leaf = addr_to_local_leaf(addr_space, ptr + size - 1u);
  uint32_t first_page = first_leaf >> TRACER_PAGE_BITS;
  uint32_t last_page = last_leaf >> TRACER_PAGE_BITS;
  if (likely(addr_space == AS_MEMORY)) {
    if (likely(first_page == last_page)) {
      record_mem_page(metering, first_page,
                      leaf_mask_range(first_leaf, last_leaf));
    } else {
      record_mem_page_range(metering, first_leaf, last_leaf);
    }
  } else if (addr_space == AS_PUBLIC_VALUES) {
    if (first_page == last_page) {
      record_pv_page(metering, first_page,
                     leaf_mask_range(first_leaf, last_leaf));
    } else {
      record_pv_page_range(metering, first_leaf, last_leaf);
    }
  } else {
    if (first_page == last_page) {
      record_deferral_page(metering, first_page,
                           leaf_mask_range(first_leaf, last_leaf));
    } else {
      record_deferral_page_range(metering, first_leaf, last_leaf);
    }
  }
}

/* Record leaves touched by [first_addr, last_addr]. Duplicates are fine —
 * Rust-side checkpoint processing deduplicates by page mask. */
static __attribute__((always_inline)) inline void record_page_range(
    MeteringState* metering, uint32_t addr_space, uint64_t first_addr,
    uint64_t last_addr) {
  uint32_t first_leaf = addr_to_local_leaf(addr_space, first_addr);
  uint32_t last_leaf = addr_to_local_leaf(addr_space, last_addr);
  if (likely(addr_space == AS_MEMORY)) {
    record_mem_page_range(metering, first_leaf, last_leaf);
  } else if (addr_space == AS_PUBLIC_VALUES) {
    record_pv_page_range(metering, first_leaf, last_leaf);
  } else {
    record_deferral_page_range(metering, first_leaf, last_leaf);
  }
}

/* ── Block-local AS_MEMORY page cache ─────────────────────────────── */

static __attribute__((always_inline)) inline TraceMemory trace_memory_setup(
    MeteringState* restrict metering) {
  TraceMemory memory = {
      .last_mem_page = NO_LAST_PAGE,
      .mem_page_buf_len = metering->mem_page_buf_len,
      .last_mem_leaf_mask = 0,
      .mem_page_buf = metering->mem_page_buf,
  };
  return memory;
}

static __attribute__((always_inline)) inline void trace_memory_drain(
    TraceMemory* restrict memory) {
  if (memory->last_mem_page == NO_LAST_PAGE ||
      memory->last_mem_leaf_mask == 0) {
    memory->last_mem_page = NO_LAST_PAGE;
    memory->last_mem_leaf_mask = 0;
    return;
  }
  append_page_touch(memory->mem_page_buf, &memory->mem_page_buf_len,
                    memory->last_mem_page, memory->last_mem_leaf_mask);
  memory->last_mem_page = NO_LAST_PAGE;
  memory->last_mem_leaf_mask = 0;
}

static __attribute__((always_inline)) inline void trace_memory_flush(
    MeteringState* restrict metering, TraceMemory* restrict memory) {
  trace_memory_drain(memory);
  metering->last_mem_page = memory->last_mem_page;
  metering->mem_page_buf_len = memory->mem_page_buf_len;
}

static __attribute__((always_inline)) inline void trace_memory_reload(
    MeteringState* restrict metering, TraceMemory* restrict memory) {
  memory->last_mem_page = NO_LAST_PAGE;
  memory->mem_page_buf_len = metering->mem_page_buf_len;
  memory->last_mem_leaf_mask = 0;
  memory->mem_page_buf = metering->mem_page_buf;
}

static __attribute__((always_inline)) inline void trace_memory_access_page(
    TraceMemory* restrict memory, uint32_t page, uint64_t leaf_mask) {
  /* Keep one pending AS_MEMORY page in registers for the current generated
   * block. Consecutive accesses to that page merge by OR-ing leaf masks. */
  if (likely(page == memory->last_mem_page)) {
    memory->last_mem_leaf_mask |= leaf_mask;
    return;
  }
  trace_memory_drain(memory);
  memory->last_mem_page = page;
  memory->last_mem_leaf_mask = leaf_mask;
}

static __attribute__((always_inline)) inline void trace_memory_access_leaf(
    TraceMemory* restrict memory, uint64_t addr) {
  uint32_t leaf = byte_addr_to_local_leaf(addr);
  trace_memory_access_page(memory, leaf >> TRACER_PAGE_BITS, leaf_mask(leaf));
}

/* Record every memory leaf overlapped by [addr, addr + size). Generated
 * accesses are no wider than one leaf, so at most two leaves are touched. */
static __attribute__((always_inline)) inline void trace_memory_access_span(
    TraceMemory* restrict memory, uint64_t addr, uint32_t size) {
  assume(size != 0u && size <= TRACER_BYTES_PER_LEAF);
  assume((size & (size - 1u)) == 0u);
  if (likely((addr & (size - 1u)) == 0u)) {
    trace_memory_access_leaf(memory, addr);
    return;
  }

  uint32_t leaf_offset = addr & TRACER_LEAF_BYTE_OFFSET_MASK;
  uint32_t bytes_until_next_leaf = TRACER_BYTES_PER_LEAF - leaf_offset;
  if (likely(size <= bytes_until_next_leaf)) {
    trace_memory_access_leaf(memory, addr);
    return;
  }

  uint32_t first_leaf = byte_addr_to_local_leaf(addr);
  uint32_t last_leaf = first_leaf + 1u;
  uint32_t first_page = first_leaf >> TRACER_PAGE_BITS;
  uint32_t last_page = last_leaf >> TRACER_PAGE_BITS;
  uint64_t first_mask = leaf_mask(first_leaf);
  uint64_t last_mask = leaf_mask(last_leaf);
  if (likely(first_page == last_page)) {
    trace_memory_access_page(memory, first_page, first_mask | last_mask);
  } else {
    trace_memory_access_page(memory, first_page, first_mask);
    trace_memory_access_page(memory, last_page, last_mask);
  }
}

static __attribute__((always_inline)) inline void trace_memory_access(
    TraceMemory* restrict memory, uint64_t addr) {
  trace_memory_access_leaf(memory, addr);
}

/* ── Trace-only register access (no-ops in metered mode) ─────────── */

static __attribute__((always_inline)) inline void trace_reg_read(
    RvState* restrict state, uint8_t idx, uint32_t val) {}
static __attribute__((always_inline)) inline void trace_reg_write(
    RvState* restrict state, uint8_t idx, uint32_t new_val) {}

static __attribute__((always_inline)) inline void trace_timestamp(
    RvState* restrict state) {}

/* ── Trace-only memory reads (record page in metered mode) ───────── */

static __attribute__((always_inline)) inline void trace_rd_mem_u8(
    RvState* restrict state, uint64_t addr, uint8_t val) {
  record_page(&state->mode_state, AS_MEMORY, addr, sizeof(uint8_t));
}

static __attribute__((always_inline)) inline void trace_rd_mem_i8(
    RvState* restrict state, uint64_t addr, int8_t val) {
  record_page(&state->mode_state, AS_MEMORY, addr, sizeof(int8_t));
}

static __attribute__((always_inline)) inline void trace_rd_mem_u16(
    RvState* restrict state, uint64_t addr, uint16_t val) {
  record_page(&state->mode_state, AS_MEMORY, addr, sizeof(uint16_t));
}

static __attribute__((always_inline)) inline void trace_rd_mem_i16(
    RvState* restrict state, uint64_t addr, int16_t val) {
  record_page(&state->mode_state, AS_MEMORY, addr, sizeof(int16_t));
}

static __attribute__((always_inline)) inline void trace_rd_mem_u32(
    RvState* restrict state, uint64_t addr, uint32_t val) {
  record_page(&state->mode_state, AS_MEMORY, addr, sizeof(uint32_t));
}

static __attribute__((always_inline)) inline void trace_rd_mem_i32(
    RvState* restrict state, uint64_t addr, int32_t val) {
  record_page(&state->mode_state, AS_MEMORY, addr, sizeof(int32_t));
}

static __attribute__((always_inline)) inline void trace_rd_mem_u64(
    RvState* restrict state, uint64_t addr, uint64_t val) {
  record_page(&state->mode_state, AS_MEMORY, addr, sizeof(uint64_t));
}

/* ── Trace-only memory writes (record page in metered mode) ──────── */

static __attribute__((always_inline)) inline void trace_wr_mem_u8(
    RvState* restrict state, uint64_t addr, uint8_t new_val) {
  record_page(&state->mode_state, AS_MEMORY, addr, sizeof(uint8_t));
}

static __attribute__((always_inline)) inline void trace_wr_mem_u16(
    RvState* restrict state, uint64_t addr, uint16_t new_val) {
  record_page(&state->mode_state, AS_MEMORY, addr, sizeof(uint16_t));
}

static __attribute__((always_inline)) inline void trace_wr_mem_u32(
    RvState* restrict state, uint64_t addr, uint32_t new_val) {
  record_page(&state->mode_state, AS_MEMORY, addr, sizeof(uint32_t));
}

static __attribute__((always_inline)) inline void trace_wr_mem_u64(
    RvState* restrict state, uint64_t addr, uint64_t new_val) {
  record_page(&state->mode_state, AS_MEMORY, addr, sizeof(uint64_t));
}

/* ── Trace-only word-range memory access ──────────────────────────── */

/* Extension range access includes page accounting in metered mode.
 * Precondition for all range functions: num_words >= 1.
 * Callers are responsible for guarding empty ranges (e.g. xorin with len=0)
 * so we can skip the branch on the hot path. */

static __attribute__((always_inline)) inline void read_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, uint64_t* restrict out,
    uint32_t num_words) {
  read_mem_u64_range_raw(state, base_addr, out, num_words);
  assume(num_words > 0);
  uint64_t last_addr = base_addr + num_words * sizeof(uint64_t) - 1u;
  record_page_range(&state->mode_state, AS_MEMORY, base_addr, last_addr);
}

static __attribute__((always_inline)) inline void write_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, const uint64_t* restrict vals,
    uint32_t num_words) {
  assume(num_words > 0);
  uint64_t last_addr = base_addr + num_words * sizeof(uint64_t) - 1u;
  record_page_range(&state->mode_state, AS_MEMORY, base_addr, last_addr);
  write_mem_u64_range_raw(state, base_addr, vals, num_words);
}

/* Peeking at a value does not create a VM memory access. */
static __attribute__((always_inline)) inline uint64_t peek_mem_u64(
    RvState* restrict state, uint64_t addr) {
  return read_mem_u64(state->memory, addr);
}

static __attribute__((always_inline)) inline void peek_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, uint64_t* restrict out,
    uint32_t num_words) {
  read_mem_u64_range_raw(state, base_addr, out, num_words);
}

/* ── Page accounting ──────────────────────────────────────────────── */

static __attribute__((always_inline)) inline void trace_page_access(
    RvState* restrict state, uint64_t addr, uint32_t size,
    uint32_t addr_space) {
  record_page(&state->mode_state, addr_space, addr, size);
}

static __attribute__((always_inline)) inline void trace_page_access_u64_range(
    RvState* restrict state, uint64_t base_addr, uint64_t num_dwords,
    uint32_t addr_space) {
  assume(num_dwords > 0);
  uint64_t last_addr = base_addr + num_dwords * WORD_SIZE - 1u;
  record_page_range(&state->mode_state, addr_space, base_addr, last_addr);
}

/* Drain AS_MEMORY page touches while preserving the instruction counter and
 * current segmentation checkpoint. */
static __attribute__((always_inline)) inline void
flush_main_memory_page_buffer(RvState* restrict state) {
  state->mode_state.on_memory_flush(&state->mode_state);
}

#pragma clang unsafe_buffer_usage end

#endif /* OPENVM_TRACER_METERED_H */
