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
    RvState* restrict /* state */, uint8_t /* idx */, uint64_t /* val */) {}
static __attribute__((always_inline)) inline void trace_reg_write(
    RvState* restrict /* state */, uint8_t /* idx */, uint64_t /* new_val */) {}

/* ── Trace-only memory reads (no-ops in pure mode) ───────────────── */

static __attribute__((always_inline)) inline void trace_rd_mem_u8(
    RvState* restrict /* state */, uint64_t /* addr */, uint8_t /* val */) {}
static __attribute__((always_inline)) inline void trace_rd_mem_i8(
    RvState* restrict /* state */, uint64_t /* addr */, int8_t /* val */) {}
static __attribute__((always_inline)) inline void trace_rd_mem_u16(
    RvState* restrict /* state */, uint64_t /* addr */, uint16_t /* val */) {}
static __attribute__((always_inline)) inline void trace_rd_mem_i16(
    RvState* restrict /* state */, uint64_t /* addr */, int16_t /* val */) {}
static __attribute__((always_inline)) inline void trace_rd_mem_u32(
    RvState* restrict /* state */, uint64_t /* addr */, uint32_t /* val */) {}
static __attribute__((always_inline)) inline void trace_rd_mem_i32(
    RvState* restrict /* state */, uint64_t /* addr */, int32_t /* val */) {}
static __attribute__((always_inline)) inline void trace_rd_mem_u64(
    RvState* restrict /* state */, uint64_t /* addr */, uint64_t /* val */) {}

/* ── Trace-only memory writes (no-ops in pure mode) ──────────────── */

static __attribute__((always_inline)) inline void trace_wr_mem_u8(
    RvState* restrict /* state */, uint64_t /* addr */, uint8_t /* new_val */) {}
static __attribute__((always_inline)) inline void trace_wr_mem_u16(
    RvState* restrict /* state */, uint64_t /* addr */, uint16_t /* new_val */) {}
static __attribute__((always_inline)) inline void trace_wr_mem_u32(
    RvState* restrict /* state */, uint64_t /* addr */, uint32_t /* new_val */) {}
static __attribute__((always_inline)) inline void trace_wr_mem_u64(
    RvState* restrict /* state */, uint64_t /* addr */, uint64_t /* new_val */) {}

/* ── Trace-only word-range memory access (no-ops in pure mode) ───── */

static __attribute__((always_inline)) inline void trace_rd_mem_u64_range(
    RvState* restrict /* state */, uint64_t /* base_addr */,
    const uint64_t* /* vals */, uint32_t /* num_words */) {}
static __attribute__((always_inline)) inline void trace_wr_mem_u64_range(
    RvState* restrict /* state */, uint64_t /* base_addr */,
    const uint64_t* /* vals */, uint32_t /* num_words */) {}

/* ── Trace-only operations (no-ops in pure mode) ──────────────────── */

static __attribute__((always_inline)) inline void trace_mem_access(
    RvState* restrict /* state */, uint64_t /* addr */, uint32_t /* addr_space */) {}

static __attribute__((always_inline)) inline void trace_mem_access_u64_range(
    RvState* restrict /* state */, uint64_t /* base_addr */, uint64_t /* num_dwords */,
    uint32_t /* addr_space */) {}

static __attribute__((always_inline)) inline void trace_pc(
    RvState* restrict /* state */, uint64_t /* pc */) {}

static __attribute__((always_inline)) inline void trace_chip(
    RvState* restrict /* state */, uint32_t /* chip_idx */, uint32_t /* count */) {}

#endif /* OPENVM_TRACER_PURE_H */
