/* OpenVM pure execution helpers. */

#ifndef OPENVM_TRACER_PURE_H
#define OPENVM_TRACER_PURE_H

#include "openvm_state.h"

/* Extension memory operations use one interface in every execution mode.
 * Pure execution performs only the requested memory access. */
static __attribute__((always_inline)) inline void read_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, uint64_t* restrict out,
    uint32_t num_words) {
  read_mem_u64_range_raw(state, base_addr, out, num_words);
}

static __attribute__((always_inline)) inline void write_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, const uint64_t* restrict vals,
    uint32_t num_words) {
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

/* Some extension callbacks report page-only accesses through a shared ABI. */
static __attribute__((always_inline)) inline void trace_page_access_u64_range(
    RvState* restrict state [[maybe_unused]],
    uint64_t base_addr [[maybe_unused]], uint64_t num_dwords [[maybe_unused]],
    uint32_t addr_space [[maybe_unused]], bool is_write [[maybe_unused]]) {}

#endif /* OPENVM_TRACER_PURE_H */
