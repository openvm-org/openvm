/*
 * rvr_ext_wrappers.c — Non-inline wrappers around static inline memory
 * functions.
 *
 * Extension FFI code (implemented in Rust as staticlibs) calls these functions
 * for memory access. They delegate to the static inline functions selected by
 * the execution mode.
 * Register access stays in generated C; extensions receive resolved register
 * values as function parameters.
 *
 * This file is only compiled when extensions are present.
 */

#include "openvm.h"
#include "rvr_ext_wrappers.h"

/* ── Extension memory boundary ─────────────────────────────────────── */

uint64_t peek_mem_u64_wrapper(RvState* state, uint64_t addr) {
  return peek_mem_u64(state, addr);
}

void peek_mem_u64_range_wrapper(RvState* state, uint64_t base_addr,
                                uint64_t* out, uint32_t num_words) {
  peek_mem_u64_range(state, base_addr, out, num_words);
}

void read_mem_u64_range_wrapper(RvState* state, uint64_t base_addr,
                                uint64_t* out, uint32_t num_words) {
  read_mem_u64_range(state, base_addr, out, num_words);
}

void write_mem_u64_range_wrapper(RvState* state, uint64_t base_addr,
                                 const uint64_t* vals, uint32_t num_words) {
  write_mem_u64_range(state, base_addr, vals, num_words);
}

void touch_mem_u64_range_wrapper(RvState* state, uint64_t base_addr,
                                 uint64_t* out, uint32_t num_words) {
  peek_mem_u64_range(state, base_addr, out, num_words);
  trace_mem_access_u64_range(state, base_addr, num_words, AS_MEMORY);
}

void record_page_access_u64_range_wrapper(RvState* state, uint64_t base_addr,
                                          uint32_t num_words,
                                          uint32_t addr_space) {
  trace_page_access_u64_range(state, base_addr, num_words, addr_space);
}

void trace_rd_mem_u64_range_wrapper(RvState* state, uint64_t base_addr,
                                    const uint64_t* vals,
                                    uint32_t num_words) {
  trace_rd_mem_u64_range(state, base_addr, vals, num_words);
}

void trace_wr_mem_u64_range_wrapper(RvState* state, uint64_t base_addr,
                                    const uint64_t* vals,
                                    uint32_t num_words) {
  trace_wr_mem_u64_range(state, base_addr, vals, num_words);
}

void trace_chip_wrapper(RvState* state, uint32_t chip_idx, uint32_t count) {
  trace_chip(state, chip_idx, count);
}

void trace_timestamp_wrapper(RvState* s) { trace_timestamp(s); }
