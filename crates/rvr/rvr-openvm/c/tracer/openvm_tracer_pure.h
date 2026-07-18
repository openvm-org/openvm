/* OpenVM pure tracing helpers.
 *
 * All trace functions are identity/no-ops. Pure instruction-limit handling,
 * when selected, is emitted directly at generated block boundaries.
 */

#ifndef OPENVM_TRACER_PURE_H
#define OPENVM_TRACER_PURE_H

#include <stdint.h>

#include "openvm_state.h"

/* ── Trace-only register access (no-ops in pure mode) ────────────── */

static __attribute__((always_inline)) inline void trace_reg_read(
    RvState* restrict state [[maybe_unused]], uint8_t idx [[maybe_unused]],
    uint64_t val [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void trace_reg_write(
    RvState* restrict state [[maybe_unused]], uint8_t idx [[maybe_unused]],
    uint64_t new_val [[maybe_unused]]) {}

/* ── Trace-only memory reads (no-ops in pure mode) ───────────────── */

static __attribute__((always_inline)) inline void trace_rd_mem_u8(
    RvState* restrict state [[maybe_unused]], uint64_t addr [[maybe_unused]],
    uint8_t val [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void trace_rd_mem_i8(
    RvState* restrict state [[maybe_unused]], uint64_t addr [[maybe_unused]],
    int8_t val [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void trace_rd_mem_u16(
    RvState* restrict state [[maybe_unused]], uint64_t addr [[maybe_unused]],
    uint16_t val [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void trace_rd_mem_i16(
    RvState* restrict state [[maybe_unused]], uint64_t addr [[maybe_unused]],
    int16_t val [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void trace_rd_mem_u32(
    RvState* restrict state [[maybe_unused]], uint64_t addr [[maybe_unused]],
    uint32_t val [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void trace_rd_mem_i32(
    RvState* restrict state [[maybe_unused]], uint64_t addr [[maybe_unused]],
    int32_t val [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void trace_rd_mem_u64(
    RvState* restrict state [[maybe_unused]], uint64_t addr [[maybe_unused]],
    uint64_t val [[maybe_unused]]) {}

/* ── Trace-only memory writes (no-ops in pure mode) ──────────────── */

static __attribute__((always_inline)) inline void trace_wr_mem_u8(
    RvState* restrict state [[maybe_unused]], uint64_t addr [[maybe_unused]],
    uint8_t new_val [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void trace_wr_mem_u16(
    RvState* restrict state [[maybe_unused]], uint64_t addr [[maybe_unused]],
    uint16_t new_val [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void trace_wr_mem_u32(
    RvState* restrict state [[maybe_unused]], uint64_t addr [[maybe_unused]],
    uint32_t new_val [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void trace_wr_mem_u64(
    RvState* restrict state [[maybe_unused]], uint64_t addr [[maybe_unused]],
    uint64_t new_val [[maybe_unused]]) {}

/* ── Trace-only word-range memory access (no-ops in pure mode) ───── */

static __attribute__((always_inline)) inline void trace_rd_mem_u64_range(
    RvState* restrict state [[maybe_unused]], uint64_t base_addr [[maybe_unused]],
    const uint64_t* vals [[maybe_unused]], uint32_t num_words [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void trace_wr_mem_u64_range(
    RvState* restrict state [[maybe_unused]], uint64_t base_addr [[maybe_unused]],
    const uint64_t* vals [[maybe_unused]], uint32_t num_words [[maybe_unused]]) {}

/* ── Trace-only operations (no-ops in pure mode) ──────────────────── */

static __attribute__((always_inline)) inline void trace_mem_access(
    RvState* restrict state [[maybe_unused]], uint64_t addr [[maybe_unused]],
    uint32_t addr_space [[maybe_unused]]) {}

static __attribute__((always_inline)) inline void trace_mem_access_u64_range(
    RvState* restrict state [[maybe_unused]], uint64_t base_addr [[maybe_unused]],
    uint64_t num_dwords [[maybe_unused]], uint32_t addr_space [[maybe_unused]]) {}

static __attribute__((always_inline)) inline void trace_pc(
    RvState* restrict state [[maybe_unused]], uint64_t pc [[maybe_unused]]) {}

#endif /* OPENVM_TRACER_PURE_H */
