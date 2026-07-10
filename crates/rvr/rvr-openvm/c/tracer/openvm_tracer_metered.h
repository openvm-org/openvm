/* OpenVM metered tracer.
 *
 * Updates per-chip trace_heights matching OpenVM's MeteredCtx.
 *
 * All functions are static inline for zero-overhead inlining.
 */

#ifndef OPENVM_TRACER_METERED_H
#define OPENVM_TRACER_METERED_H

#include <stdint.h>

#include "openvm_state.h"

static constexpr uint32_t NO_LAST_PAGE = UINT32_MAX;

static constexpr uint32_t MAX_PV_PAGES_PER_INSN = 1;
_Static_assert(
    TRACER_MEM_PAGE_BUF_CAP >= 2 * TRACER_SEGMENT_CHECK_INSNS * TRACER_MAX_MEM_PAGES_PER_INSN,
    "MEM_PAGE_BUF_CAP too small for worst-case pages per flush interval");
_Static_assert(
    TRACER_PV_PAGE_BUF_CAP >= 2 * TRACER_SEGMENT_CHECK_INSNS * MAX_PV_PAGES_PER_INSN,
    "PV_PAGE_BUF_CAP too small for worst-case pages per flush interval");
/* No static assert for DEFERRAL — justifying the capacity is a TODO
 * (see DEFERRAL_PAGE_BUF_CAP in metered.rs). */

typedef struct PageAccess {
  /* Page table index plus a 64-bit leaf mask for that page. */
  uint32_t page_id;
  uint64_t leaf_mask;
} PageAccess;

/* One page buffer per address space stores local page ids and leaf masks.
 * AS_REGISTER pages are handled entirely on the Rust side at init. */
typedef struct Tracer {
  uint32_t* trace_heights;
  PageAccess* mem_page_buf;
  PageAccess* pv_page_buf;
  PageAccess* deferral_page_buf;
  /* Always initialized by Rust; called unconditionally on the cold checkpoint
   * path to avoid a hot-path null check. */
  uint8_t (*on_check)(struct Tracer*);
  void* seg_state;
  uint32_t mem_page_buf_len;
  uint32_t pv_page_buf_len;
  uint32_t deferral_page_buf_len;
  uint32_t check_counter;
  /* Dedup cache: skip consecutive accesses to the same AS_MEMORY page.
   * Reset to NO_LAST_PAGE on every buffer flush (required for correctness
   * across segment boundaries that clear the global BitSet). */
  uint32_t last_mem_page;
} Tracer;

typedef struct TraceMemory {
  uint32_t last_mem_page;
  uint32_t mem_page_buf_len;
  uint64_t last_mem_leaf_mask;
  PageAccess* mem_page_buf;
} TraceMemory;

/* ── Page tracking ─────────────────────────────────────────────────── */

static __attribute__((always_inline)) inline uint32_t byte_addr_to_local_leaf(
    uint64_t ptr) {
  return ptr >> TRACER_BYTE_SPACE_PTRS_PER_LEAF_BITS;
}

static __attribute__((always_inline)) inline uint32_t deferral_addr_to_local_leaf(
    uint64_t ptr) {
  return ptr >> TRACER_DEFERRAL_PTRS_PER_LEAF_BITS;
}

static __attribute__((always_inline)) inline uint32_t addr_to_local_leaf(
    uint32_t addr_space, uint64_t ptr) {
  if (likely(addr_space == AS_MEMORY || addr_space == AS_PUBLIC_VALUES)) {
    return byte_addr_to_local_leaf(ptr);
  }
  return deferral_addr_to_local_leaf(ptr);
}

static __attribute__((always_inline)) inline uint64_t leaf_mask_range(
    uint32_t first_leaf, uint32_t last_leaf) {
  /* Convert an inclusive leaf range within a page into a uint64_t occupancy mask. */
  uint32_t start = first_leaf & ((1u << TRACER_PAGE_BITS) - 1u);
  uint32_t end = (last_leaf & ((1u << TRACER_PAGE_BITS) - 1u)) + 1u;
  uint32_t width = end - start;
  uint64_t mask = width == 64u ? UINT64_MAX : ((1ull << width) - 1ull);
  return mask << start;
}

/* ── Per-address-space page recording ─────────────────────────────── */

static __attribute__((always_inline)) inline void append_page_access(
    PageAccess* restrict buf, uint32_t* restrict len, uint32_t page,
    uint64_t leaf_mask) {
  PageAccess* slot = &buf[(*len)++];
  slot->page_id = page;
  slot->leaf_mask = leaf_mask;
}

static __attribute__((always_inline)) inline void append_page_access_range(
    PageAccess* restrict buf, uint32_t* restrict len, uint32_t first_leaf,
    uint32_t last_leaf) {
  uint32_t first_page = first_leaf >> TRACER_PAGE_BITS;
  uint32_t last_page = last_leaf >> TRACER_PAGE_BITS;
  for (uint32_t page = first_page; page <= last_page; page++) {
    uint32_t page_first_leaf = page << TRACER_PAGE_BITS;
    uint32_t page_last_leaf = page_first_leaf + (1u << TRACER_PAGE_BITS) - 1u;
    uint32_t start = first_leaf > page_first_leaf ? first_leaf : page_first_leaf;
    uint32_t end = last_leaf < page_last_leaf ? last_leaf : page_last_leaf;
    append_page_access(buf, len, page, leaf_mask_range(start, end));
  }
}

/* No bounds check — see MEM_PAGE_BUF_CAP in metered.rs. */
static __attribute__((always_inline)) inline void record_mem_page(
    Tracer* t, uint32_t page, uint64_t leaf_mask) {
  if (likely(page == t->last_mem_page)) {
    t->mem_page_buf[t->mem_page_buf_len - 1u].leaf_mask |= leaf_mask;
    return;
  }
  t->last_mem_page = page;
  append_page_access(t->mem_page_buf, &t->mem_page_buf_len, page, leaf_mask);
}

static __attribute__((always_inline)) inline void record_mem_page_range(
    Tracer* t, uint32_t first_leaf, uint32_t last_leaf) {
  uint32_t first_page = first_leaf >> TRACER_PAGE_BITS;
  uint32_t last_page = last_leaf >> TRACER_PAGE_BITS;
  if (likely(first_page == t->last_mem_page)) {
    t->mem_page_buf[t->mem_page_buf_len - 1u].leaf_mask |=
        leaf_mask_range(first_leaf, first_page == last_page
                                        ? last_leaf
                                        : ((first_page + 1u) << TRACER_PAGE_BITS) - 1u);
    if (first_page == last_page) {
      return;
    }
    first_page++;
  }
  uint32_t len = t->mem_page_buf_len;
  for (uint32_t page = first_page; page <= last_page; page++) {
    uint32_t page_first_leaf = page << TRACER_PAGE_BITS;
    uint32_t page_last_leaf = page_first_leaf + (1u << TRACER_PAGE_BITS) - 1u;
    uint32_t start = first_leaf > page_first_leaf ? first_leaf : page_first_leaf;
    uint32_t end = last_leaf < page_last_leaf ? last_leaf : page_last_leaf;
    append_page_access(t->mem_page_buf, &len, page, leaf_mask_range(start, end));
  }
  t->mem_page_buf_len = len;
  t->last_mem_page = last_page;
}

/* No bounds check — see PV_PAGE_BUF_CAP in metered.rs. */
static __attribute__((always_inline)) inline void record_pv_page(
    Tracer* t, uint32_t page, uint64_t leaf_mask) {
  append_page_access(t->pv_page_buf, &t->pv_page_buf_len, page, leaf_mask);
}

static __attribute__((always_inline)) inline void record_pv_page_range(
    Tracer* t, uint32_t first_leaf, uint32_t last_leaf) {
  uint32_t len = t->pv_page_buf_len;
  append_page_access_range(t->pv_page_buf, &len, first_leaf, last_leaf);
  t->pv_page_buf_len = len;
}

/* No bounds check — see DEFERRAL_PAGE_BUF_CAP in metered.rs. */
static __attribute__((always_inline)) inline void record_deferral_page(
    Tracer* t, uint32_t page, uint64_t leaf_mask) {
  append_page_access(t->deferral_page_buf, &t->deferral_page_buf_len, page, leaf_mask);
}

static __attribute__((always_inline)) inline void record_deferral_page_range(
    Tracer* t, uint32_t first_leaf, uint32_t last_leaf) {
  uint32_t len = t->deferral_page_buf_len;
  append_page_access_range(t->deferral_page_buf, &len, first_leaf, last_leaf);
  t->deferral_page_buf_len = len;
}

/* Record a single page access. `addr_space` is a compile-time constant at
 * every direct call site in generated C, so the branches below fold away. */
static __attribute__((always_inline)) inline void record_page(
    Tracer* t, uint32_t addr_space, uint64_t ptr, uint32_t size) {
  uint32_t first_leaf = addr_to_local_leaf(addr_space, ptr);
  uint32_t last_leaf = addr_to_local_leaf(addr_space, ptr + size - 1u);
  uint32_t first_page = first_leaf >> TRACER_PAGE_BITS;
  uint32_t last_page = last_leaf >> TRACER_PAGE_BITS;
  if (likely(addr_space == AS_MEMORY)) {
    if (likely(first_page == last_page)) {
      record_mem_page(t, first_page, leaf_mask_range(first_leaf, last_leaf));
    } else {
      record_mem_page_range(t, first_leaf, last_leaf);
    }
  } else if (addr_space == AS_PUBLIC_VALUES) {
    if (first_page == last_page) {
      record_pv_page(t, first_page, leaf_mask_range(first_leaf, last_leaf));
    } else {
      record_pv_page_range(t, first_leaf, last_leaf);
    }
  } else {
    if (first_page == last_page) {
      record_deferral_page(t, first_page, leaf_mask_range(first_leaf, last_leaf));
    } else {
      record_deferral_page_range(t, first_leaf, last_leaf);
    }
  }
}

/* Record leaves touched by [first_addr, last_addr]. Duplicates are fine —
 * Rust-side checkpoint processing deduplicates by page mask. */
static __attribute__((always_inline)) inline void record_page_range(
    Tracer* t, uint32_t addr_space, uint64_t first_addr, uint64_t last_addr) {
  uint32_t first_leaf = addr_to_local_leaf(addr_space, first_addr);
  uint32_t last_leaf = addr_to_local_leaf(addr_space, last_addr);
  if (likely(addr_space == AS_MEMORY)) {
    record_mem_page_range(t, first_leaf, last_leaf);
  } else if (addr_space == AS_PUBLIC_VALUES) {
    record_pv_page_range(t, first_leaf, last_leaf);
  } else {
    record_deferral_page_range(t, first_leaf, last_leaf);
  }
}

/* ── Block-local AS_MEMORY page cache ─────────────────────────────── */

static __attribute__((always_inline)) inline TraceMemory trace_memory_setup(
    Tracer* restrict t) {
  TraceMemory memory = {
      .last_mem_page = NO_LAST_PAGE,
      .mem_page_buf_len = t->mem_page_buf_len,
      .last_mem_leaf_mask = 0,
      .mem_page_buf = t->mem_page_buf,
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
  append_page_access(memory->mem_page_buf, &memory->mem_page_buf_len,
                     memory->last_mem_page, memory->last_mem_leaf_mask);
  memory->last_mem_page = NO_LAST_PAGE;
  memory->last_mem_leaf_mask = 0;
}

static __attribute__((always_inline)) inline void trace_memory_flush(
    Tracer* restrict t, TraceMemory* restrict memory) {
  trace_memory_drain(memory);
  t->last_mem_page = memory->last_mem_page;
  t->mem_page_buf_len = memory->mem_page_buf_len;
}

static __attribute__((always_inline)) inline void trace_memory_reload(
    Tracer* restrict t, TraceMemory* restrict memory) {
  memory->last_mem_page = NO_LAST_PAGE;
  memory->mem_page_buf_len = t->mem_page_buf_len;
  memory->last_mem_leaf_mask = 0;
  memory->mem_page_buf = t->mem_page_buf;
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
  trace_memory_access_page(memory, leaf >> TRACER_PAGE_BITS,
                           1ull << (leaf & ((1u << TRACER_PAGE_BITS) - 1u)));
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
  record_page(state->tracer, AS_MEMORY, addr, sizeof(uint8_t));
}

static __attribute__((always_inline)) inline void trace_rd_mem_i8(
    RvState* restrict state, uint64_t addr, int8_t val) {
  record_page(state->tracer, AS_MEMORY, addr, sizeof(int8_t));
}

static __attribute__((always_inline)) inline void trace_rd_mem_u16(
    RvState* restrict state, uint64_t addr, uint16_t val) {
  record_page(state->tracer, AS_MEMORY, addr, sizeof(uint16_t));
}

static __attribute__((always_inline)) inline void trace_rd_mem_i16(
    RvState* restrict state, uint64_t addr, int16_t val) {
  record_page(state->tracer, AS_MEMORY, addr, sizeof(int16_t));
}

static __attribute__((always_inline)) inline void trace_rd_mem_u32(
    RvState* restrict state, uint64_t addr, uint32_t val) {
  record_page(state->tracer, AS_MEMORY, addr, sizeof(uint32_t));
}

static __attribute__((always_inline)) inline void trace_rd_mem_i32(
    RvState* restrict state, uint64_t addr, int32_t val) {
  record_page(state->tracer, AS_MEMORY, addr, sizeof(int32_t));
}

static __attribute__((always_inline)) inline void trace_rd_mem_u64(
    RvState* restrict state, uint64_t addr, uint64_t val) {
  record_page(state->tracer, AS_MEMORY, addr, sizeof(uint64_t));
}

/* ── Trace-only memory writes (record page in metered mode) ──────── */

static __attribute__((always_inline)) inline void trace_wr_mem_u8(
    RvState* restrict state, uint64_t addr, uint8_t new_val) {
  record_page(state->tracer, AS_MEMORY, addr, sizeof(uint8_t));
}

static __attribute__((always_inline)) inline void trace_wr_mem_u16(
    RvState* restrict state, uint64_t addr, uint16_t new_val) {
  record_page(state->tracer, AS_MEMORY, addr, sizeof(uint16_t));
}

static __attribute__((always_inline)) inline void trace_wr_mem_u32(
    RvState* restrict state, uint64_t addr, uint32_t new_val) {
  record_page(state->tracer, AS_MEMORY, addr, sizeof(uint32_t));
}

static __attribute__((always_inline)) inline void trace_wr_mem_u64(
    RvState* restrict state, uint64_t addr, uint64_t new_val) {
  record_page(state->tracer, AS_MEMORY, addr, sizeof(uint64_t));
}

/* ── Trace-only word-range memory access ──────────────────────────── */

/* Precondition for all *_range trace functions: num_words >= 1.
 * Callers are responsible for guarding empty ranges (e.g. xorin with len=0)
 * so we can skip the branch on the hot path. */

static __attribute__((always_inline)) inline void trace_rd_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, const uint64_t* vals,
    uint32_t num_words) {
  assume(num_words > 0);
  uint64_t last_addr = base_addr + num_words * sizeof(uint64_t) - 1u;
  record_page_range(state->tracer, AS_MEMORY, base_addr, last_addr);
}

static __attribute__((always_inline)) inline void trace_wr_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, const uint64_t* vals,
    uint32_t num_words) {
  assume(num_words > 0);
  uint64_t last_addr = base_addr + num_words * sizeof(uint64_t) - 1u;
  record_page_range(state->tracer, AS_MEMORY, base_addr, last_addr);
}

/* ── Trace-only operations ────────────────────────────────────────── */

static __attribute__((always_inline)) inline void trace_mem_access(
    RvState* restrict state, uint64_t addr, uint32_t addr_space) {
  record_page(state->tracer, addr_space, addr, WORD_SIZE);
}

static __attribute__((always_inline)) inline void trace_mem_access_u64_range(
    RvState* restrict state, uint64_t base_addr, uint32_t num_dwords,
    uint32_t addr_space) {
  assume(num_dwords > 0);
  uint64_t last_addr = base_addr + num_dwords * WORD_SIZE - 1u;
  record_page_range(state->tracer, addr_space, base_addr, last_addr);
}

static __attribute__((always_inline)) inline void trace_wr_as_u64(
    RvState* restrict state, uint64_t addr, uint64_t new_val,
    uint32_t addr_space) {
  record_page(state->tracer, addr_space, addr, WORD_SIZE);
}

static __attribute__((always_inline)) inline void trace_pc(
    RvState* restrict state, uint64_t pc) {}

static __attribute__((always_inline)) inline void trace_chip(
    RvState* restrict state, uint32_t chip_idx, uint32_t count) {
  state->tracer->trace_heights[chip_idx] += count;
}

static __attribute__((always_inline)) inline void trace_block(
    RvState* restrict state, uint64_t pc, uint32_t block_insn_count) {
  if (unlikely(state->tracer->check_counter < block_insn_count)) {
    state->tracer->on_check(state->tracer);
  }
  state->tracer->check_counter -= block_insn_count;
}

static __attribute__((always_inline)) inline uint8_t
trace_block_with_segment_check(RvState* restrict state, uint64_t pc,
                               uint32_t block_insn_count) {
  if (unlikely(state->tracer->check_counter < block_insn_count)) {
    if (state->tracer->on_check(state->tracer)) {
      return 1;
    }
  }
  state->tracer->check_counter -= block_insn_count;
  return 0;
}

#endif /* OPENVM_TRACER_METERED_H */
