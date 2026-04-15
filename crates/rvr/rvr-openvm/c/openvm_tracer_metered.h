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

/* One page buffer per address space so each stores only page ids (1 u32).
 * AS_REGISTER pages are handled entirely on the Rust side at init. */
typedef struct Tracer {
  uint32_t* trace_heights;
  uint32_t* mem_page_buf;
  uint32_t* pv_page_buf;
  uint32_t* deferral_page_buf;
  void (*on_check)(struct Tracer*);
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

static constexpr uint32_t NO_CHIP = UINT32_MAX;
static constexpr uint32_t NO_LAST_PAGE = UINT32_MAX;

static constexpr uint32_t MAX_MEM_PAGES_PER_INSN = 10;
static constexpr uint32_t MAX_PV_PAGES_PER_INSN = 1;
_Static_assert(
    TRACER_MEM_PAGE_BUF_CAP >= 2 * TRACER_SEGMENT_CHECK_INSNS * MAX_MEM_PAGES_PER_INSN,
    "MEM_PAGE_BUF_CAP too small for worst-case pages per flush interval");
_Static_assert(
    TRACER_PV_PAGE_BUF_CAP >= 2 * TRACER_SEGMENT_CHECK_INSNS * MAX_PV_PAGES_PER_INSN,
    "PV_PAGE_BUF_CAP too small for worst-case pages per flush interval");
/* No static assert for DEFERRAL — justifying the capacity is a TODO
 * (see DEFERRAL_PAGE_BUF_CAP in metered.rs). */

/* ── Page tracking ─────────────────────────────────────────────────── */

static __attribute__((always_inline)) inline uint32_t addr_to_local_page(uint32_t ptr) {
  return (ptr >> TRACER_CHUNK_BITS) >> TRACER_PAGE_BITS;
}

/* ── Per-address-space page recording ─────────────────────────────── */

/* No bounds check — see MEM_PAGE_BUF_CAP in metered.rs. */
static __attribute__((always_inline)) inline void record_mem_page(Tracer* t, uint32_t page) {
  if (likely(page == t->last_mem_page)) {
    return;
  }
  t->last_mem_page = page;
  t->mem_page_buf[t->mem_page_buf_len++] = page;
}

static __attribute__((always_inline)) inline void record_mem_page_range(Tracer* t, uint32_t first_page, uint32_t last_page) {
  if (likely(first_page == t->last_mem_page)) {
    if (first_page == last_page) {
      return;
    }
    first_page++;
  }
  uint32_t n = last_page - first_page + 1;
  uint32_t len = t->mem_page_buf_len;
  for (uint32_t i = 0; i < n; i++) {
    t->mem_page_buf[len + i] = first_page + i;
  }
  t->mem_page_buf_len = len + n;
  t->last_mem_page = last_page;
}

/* No bounds check — see PV_PAGE_BUF_CAP in metered.rs. */
static __attribute__((always_inline)) inline void record_pv_page(Tracer* t, uint32_t page) {
  t->pv_page_buf[t->pv_page_buf_len++] = page;
}

static __attribute__((always_inline)) inline void record_pv_page_range(Tracer* t, uint32_t first_page, uint32_t last_page) {
  uint32_t n = last_page - first_page + 1;
  uint32_t len = t->pv_page_buf_len;
  for (uint32_t i = 0; i < n; i++) {
    t->pv_page_buf[len + i] = first_page + i;
  }
  t->pv_page_buf_len = len + n;
}

/* No bounds check — see DEFERRAL_PAGE_BUF_CAP in metered.rs. */
static __attribute__((always_inline)) inline void record_deferral_page(Tracer* t, uint32_t page) {
  t->deferral_page_buf[t->deferral_page_buf_len++] = page;
}

static __attribute__((always_inline)) inline void record_deferral_page_range(Tracer* t, uint32_t first_page, uint32_t last_page) {
  uint32_t n = last_page - first_page + 1;
  uint32_t len = t->deferral_page_buf_len;
  for (uint32_t i = 0; i < n; i++) {
    t->deferral_page_buf[len + i] = first_page + i;
  }
  t->deferral_page_buf_len = len + n;
}

/* Record a single page access. `addr_space` is a compile-time constant at
 * every direct call site in generated C, so the branches below fold away. */
static __attribute__((always_inline)) inline void record_page(Tracer* t, uint32_t addr_space, uint32_t ptr) {
  uint32_t page = addr_to_local_page(ptr);
  if (likely(addr_space == AS_MEMORY)) {
    record_mem_page(t, page);
  } else if (addr_space == AS_PUBLIC_VALUES) {
    record_pv_page(t, page);
  } else {
    record_deferral_page(t, page);
  }
}

/* Record pages touched by [first_addr, last_addr]. Duplicates are fine —
 * flush_page_buffer deduplicates via BitSet. */
static __attribute__((always_inline)) inline void record_page_range(Tracer* t, uint32_t addr_space, uint32_t first_addr,
                                                                    uint32_t last_addr) {
  uint32_t first_page = addr_to_local_page(first_addr);
  uint32_t last_page = addr_to_local_page(last_addr);
  if (likely(addr_space == AS_MEMORY)) {
    record_mem_page_range(t, first_page, last_page);
  } else if (addr_space == AS_PUBLIC_VALUES) {
    record_pv_page_range(t, first_page, last_page);
  } else {
    record_deferral_page_range(t, first_page, last_page);
  }
}

/* ── Trace-only register access (no-ops in metered mode) ─────────── */

static __attribute__((always_inline)) inline void trace_reg_read(RvState* restrict state, uint8_t idx, uint32_t val) {}
static __attribute__((always_inline)) inline void trace_reg_write(RvState* restrict state, uint8_t idx, uint32_t new_val) {}

/* ── Trace-only memory reads (record page in metered mode) ───────── */

static __attribute__((always_inline)) inline void trace_rd_mem_u8(RvState* restrict state, uint32_t addr, uint8_t val) {
  record_mem_page(state->tracer, addr_to_local_page(addr));
}

static __attribute__((always_inline)) inline void trace_rd_mem_i8(RvState* restrict state, uint32_t addr, int8_t val) {
  record_mem_page(state->tracer, addr_to_local_page(addr));
}

static __attribute__((always_inline)) inline void trace_rd_mem_u16(RvState* restrict state, uint32_t addr, uint16_t val) {
  record_mem_page(state->tracer, addr_to_local_page(addr));
}

static __attribute__((always_inline)) inline void trace_rd_mem_i16(RvState* restrict state, uint32_t addr, int16_t val) {
  record_mem_page(state->tracer, addr_to_local_page(addr));
}

static __attribute__((always_inline)) inline void trace_rd_mem_u32(RvState* restrict state, uint32_t addr, uint32_t val) {
  record_mem_page(state->tracer, addr_to_local_page(addr));
}

/* ── Trace-only memory writes (record page in metered mode) ──────── */

static __attribute__((always_inline)) inline void trace_wr_mem_u8(RvState* restrict state, uint32_t addr, uint8_t new_val) {
  record_mem_page(state->tracer, addr_to_local_page(addr));
}

static __attribute__((always_inline)) inline void trace_wr_mem_u16(RvState* restrict state, uint32_t addr, uint16_t new_val) {
  record_mem_page(state->tracer, addr_to_local_page(addr));
}

static __attribute__((always_inline)) inline void trace_wr_mem_u32(RvState* restrict state, uint32_t addr, uint32_t new_val) {
  record_mem_page(state->tracer, addr_to_local_page(addr));
}

/* ── Trace-only word-range memory access ──────────────────────────── */

/* Precondition for all *_range trace functions: num_words >= 1.
 * Callers are responsible for guarding empty ranges (e.g. xorin with len=0)
 * so we can skip the branch on the hot path. */

static __attribute__((always_inline)) inline void trace_rd_mem_u32_range(RvState* restrict state, uint32_t base_addr,
                                                                         const uint32_t* vals, uint32_t num_words) {
  (void)vals;
  assume(num_words > 0);
  uint32_t last_addr = base_addr + (num_words - 1) * WORD_SIZE;
  record_mem_page_range(state->tracer, addr_to_local_page(base_addr), addr_to_local_page(last_addr));
}

static __attribute__((always_inline)) inline void trace_wr_mem_u32_range(RvState* restrict state, uint32_t base_addr,
                                                                         const uint32_t* vals, uint32_t num_words) {
  (void)vals;
  assume(num_words > 0);
  uint32_t last_addr = base_addr + (num_words - 1) * WORD_SIZE;
  record_mem_page_range(state->tracer, addr_to_local_page(base_addr), addr_to_local_page(last_addr));
}

/* ── Trace-only operations ────────────────────────────────────────── */

static __attribute__((always_inline)) inline void trace_mem_access(RvState* restrict state, uint32_t addr, uint32_t addr_space) {
  record_page(state->tracer, addr_space, addr);
}

static __attribute__((always_inline)) inline void trace_mem_access_u32_range(RvState* restrict state, uint32_t base_addr,
                                                                             uint32_t num_words, uint32_t addr_space) {
  assume(num_words > 0);
  uint32_t last_addr = base_addr + (num_words - 1) * WORD_SIZE;
  record_page_range(state->tracer, addr_space, base_addr, last_addr);
}

static __attribute__((always_inline)) inline void trace_pc(RvState* restrict state, uint32_t pc) {}

static __attribute__((always_inline)) inline void trace_block(RvState* restrict state, uint32_t pc, uint32_t block_insn_count) {
  if (unlikely(state->tracer->check_counter < block_insn_count)) {
    if (state->tracer->on_check) {
      state->tracer->on_check(state->tracer);
    }
  }
  state->tracer->check_counter -= block_insn_count;
}

static __attribute__((always_inline)) inline void trace_chip(RvState* restrict state, uint32_t chip_idx, uint32_t count) {
  if (likely(chip_idx != NO_CHIP)) {
    state->tracer->trace_heights[chip_idx] += count;
  }
}

#endif /* OPENVM_TRACER_METERED_H */
