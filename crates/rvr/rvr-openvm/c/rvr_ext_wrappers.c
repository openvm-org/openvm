/*
 * rvr_ext_wrappers.c — Non-inline wrappers around static inline tracing
 * functions.
 *
 * Extension FFI code (implemented in Rust as staticlibs) calls these wrappers
 * for traced memory access and chip cost. The wrappers delegate to the static
 * inline functions defined in the tracer headers.
 * Register access stays in generated C; extensions receive resolved register
 * values as function parameters.
 *
 * This file is only compiled when extensions are present.
 */

#include "openvm.h"
#include "rvr_ext_wrappers.h"

/* ── Memory access (single word) ───────────────────────────────────── */

uint64_t rd_mem_u64_wrapper(RvState* s, uint64_t addr) {
  return rd_mem_u64(s->memory, addr);
}

void rd_mem_u64_range_wrapper(RvState* s, uint64_t base_addr, uint64_t* out,
                              uint32_t num_words) {
  rd_mem_u64_range(s, base_addr, out, num_words);
}

void wr_mem_u64_range_wrapper(RvState* s, uint64_t base_addr,
                              const uint64_t* vals, uint32_t num_words) {
  wr_mem_u64_range(s, base_addr, vals, num_words);
}

void trace_rd_mem_u64_range_wrapper(RvState* s, uint64_t base_addr,
                                    const uint64_t* vals, uint32_t num_words) {
  trace_rd_mem_u64_range(s, base_addr, vals, num_words);
}

void trace_wr_mem_u64_range_wrapper(RvState* s, uint64_t base_addr,
                                    const uint64_t* vals, uint32_t num_words) {
  trace_wr_mem_u64_range(s, base_addr, vals, num_words);
}

void trace_mem_access_u64_range_wrapper(RvState* s, uint64_t base_addr,
                                        uint32_t num_words,
                                        uint32_t addr_space) {
  trace_mem_access_u64_range(s, base_addr, num_words, addr_space);
}
