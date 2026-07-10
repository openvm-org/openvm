/* OpenVM metered-cost execution helpers. */

#ifndef OPENVM_TRACER_METERED_COST_H
#define OPENVM_TRACER_METERED_COST_H

#include "openvm_state.h"

/* ── Trace-only register access (no-ops in metered cost mode) ────── */

static __attribute__((always_inline)) inline void trace_reg_read(
    RvState* restrict state, uint8_t idx, uint32_t val) {}
static __attribute__((always_inline)) inline void trace_reg_write(
    RvState* restrict state, uint8_t idx, uint32_t new_val) {}

static __attribute__((always_inline)) inline void trace_timestamp(
    RvState* restrict state) {}

/* ── Trace-only memory reads (no-ops in metered cost mode) ───────── */

static __attribute__((always_inline)) inline void trace_rd_mem_u8(
    RvState* restrict state, uint64_t addr, uint8_t val) {}
static __attribute__((always_inline)) inline void trace_rd_mem_i8(
    RvState* restrict state, uint64_t addr, int8_t val) {}
static __attribute__((always_inline)) inline void trace_rd_mem_u16(
    RvState* restrict state, uint64_t addr, uint16_t val) {}
static __attribute__((always_inline)) inline void trace_rd_mem_i16(
    RvState* restrict state, uint64_t addr, int16_t val) {}
static __attribute__((always_inline)) inline void trace_rd_mem_u32(
    RvState* restrict state, uint64_t addr, uint32_t val) {}
static __attribute__((always_inline)) inline void trace_rd_mem_i32(
    RvState* restrict state, uint64_t addr, int32_t val) {}
static __attribute__((always_inline)) inline void trace_rd_mem_u64(
    RvState* restrict state, uint64_t addr, uint64_t val) {}

/* ── Trace-only memory writes (no-ops in metered cost mode) ──────── */

static __attribute__((always_inline)) inline void trace_wr_mem_u8(
    RvState* restrict state, uint64_t addr, uint8_t new_val) {}
static __attribute__((always_inline)) inline void trace_wr_mem_u16(
    RvState* restrict state, uint64_t addr, uint16_t new_val) {}
static __attribute__((always_inline)) inline void trace_wr_mem_u32(
    RvState* restrict state, uint64_t addr, uint32_t new_val) {}
static __attribute__((always_inline)) inline void trace_wr_mem_u64(
    RvState* restrict state, uint64_t addr, uint64_t new_val) {}

/* ── Trace-only word-range memory access (no-ops in metered cost mode) */

static __attribute__((always_inline)) inline void trace_rd_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, const uint64_t* vals,
    uint32_t num_words) {}
static __attribute__((always_inline)) inline void trace_wr_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, const uint64_t* vals,
    uint32_t num_words) {}

/* ── Trace-only operations ────────────────────────────────────────── */

static __attribute__((always_inline)) inline void trace_mem_access(
    RvState* restrict state, uint64_t addr, uint32_t addr_space) {}

static __attribute__((always_inline)) inline void trace_mem_access_u64_range(
    RvState* restrict state, uint64_t base_addr, uint32_t num_dwords,
    uint32_t addr_space) {}

static __attribute__((always_inline)) inline void trace_wr_as_u64(
    RvState* restrict state, uint64_t addr, uint64_t new_val,
    uint32_t addr_space) {}

static __attribute__((always_inline)) inline void trace_wr_as(
    RvState* restrict state, uint64_t addr, uint64_t new_val, uint32_t width,
    uint32_t addr_space) {}

static __attribute__((always_inline)) inline void trace_pc(
    RvState* restrict state, uint64_t pc) {}

static __attribute__((always_inline)) inline void trace_chip(
    RvState* restrict state, uint32_t chip_idx, uint32_t count) {}

/* Memory page accounting does not contribute to scalar metered cost. */
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

/* Peeking at a value does not contribute to scalar metered cost. */
static __attribute__((always_inline)) inline uint64_t peek_mem_u64(
    RvState* restrict state, uint64_t addr) {
  return read_mem_u64(state->memory, addr);
}

static __attribute__((always_inline)) inline void peek_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, uint64_t* restrict out,
    uint32_t num_words) {
  read_mem_u64_range_raw(state, base_addr, out, num_words);
}

/* Page locations do not affect scalar metered cost. */
static __attribute__((always_inline)) inline void trace_page_access_u64_range(
    RvState* restrict state [[maybe_unused]],
    uint64_t base_addr [[maybe_unused]], uint64_t num_dwords [[maybe_unused]],
    uint32_t addr_space [[maybe_unused]]) {}

static __attribute__((always_inline)) inline void
flush_main_memory_page_buffer(RvState* restrict state [[maybe_unused]]) {}

#endif /* OPENVM_TRACER_METERED_COST_H */
