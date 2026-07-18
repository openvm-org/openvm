/* RvState struct, shared memory/register access, and common constants. */

#ifndef OPENVM_STATE_H
#define OPENVM_STATE_H

#include <stdint.h>
#include <string.h>

#include "openvm_check_mem_bounds.h"
#include "openvm_constants.h"
#include "openvm_util.h"

typedef enum OpenVmExecStatus : uint8_t {
  OPENVM_EXEC_RUNNING = 0,
  OPENVM_EXEC_TERMINATED = 1,
  OPENVM_EXEC_SUSPENDED = 2,
  OPENVM_EXEC_TRAPPED = 3,
} OpenVmExecStatus;

static_assert(OPENVM_EXEC_RUNNING == 0,
              "must match rvr_state::ExecutionStatus::Running");
static_assert(OPENVM_EXEC_TERMINATED == 1,
              "must match rvr_state::ExecutionStatus::Terminated");
static_assert(OPENVM_EXEC_SUSPENDED == 2,
              "must match rvr_state::ExecutionStatus::Suspended");
static_assert(OPENVM_EXEC_TRAPPED == 3,
              "must match rvr_state::ExecutionStatus::Trapped");

/* Generated per execution kind before this header is compiled. */
#include "openvm_state_layout.h"

static __attribute__((always_inline)) inline void rv_set_status(
    RvState* restrict state, OpenVmExecStatus status, uint8_t exit_code) {
  state->status = status;
  state->exit_code = exit_code;
}

static __attribute__((always_inline)) inline void rv_set_status_at(
    RvState* restrict state, uint64_t pc, OpenVmExecStatus status,
    uint8_t exit_code) {
  state->pc = pc;
  rv_set_status(state, status, exit_code);
}

static __attribute__((always_inline)) inline uint64_t rv_dispatch_index(
    uint64_t pc) {
  return (pc - RV_TEXT_START) >> 2;
}

/* True if pc names a 4-byte dispatch slot in the dense table. The alignment
 * check prevents an unaligned pc from aliasing to a neighboring slot. */
static __attribute__((always_inline)) inline bool rv_pc_is_dispatchable(
    uint64_t pc) {
  return pc >= RV_TEXT_START && pc <= RV_TEXT_END &&
         ((pc - RV_TEXT_START) & 3ull) == 0;
}

/* ── Guest memory pointer ────────────────────────────────────────── */

/* Guard regions immediately before and after guest memory catch accesses
 * that cross either boundary.
 * TODO: addr &= MEMORY_MASK for defense-in-depth.
 * Clang's C unsafe-buffer heuristic cannot model the bounds assumptions used
 * by these low-level memory and fixed-register-array helpers. */
#pragma clang unsafe_buffer_usage begin
static __attribute__((always_inline)) inline uint8_t* mem_ptr(
    uint8_t* restrict memory, uint64_t addr) {
  assume(addr <= MEMORY_MASK);
  return memory + addr;
}

/* ── Register access ─────────────────────────────────────────────── */

static __attribute__((always_inline)) inline uint64_t reg_read(
    RvState* restrict state, uint8_t idx) {
  return state->regs[idx];
}

static __attribute__((always_inline)) inline void reg_write(
    RvState* restrict state, uint8_t idx, uint64_t val) {
  state->regs[idx] = val;
}
#pragma clang unsafe_buffer_usage end

/* ── Per-width memory reads ──────────────────────────────────────── */

static __attribute__((always_inline)) inline uint8_t rd_mem_u8(
    uint8_t* restrict memory, uint64_t addr) {
  check_mem_bounds_u8(addr);
  return *mem_ptr(memory, addr);
}

static __attribute__((always_inline)) inline int8_t rd_mem_i8(
    uint8_t* restrict memory, uint64_t addr) {
  check_mem_bounds_i8(addr);
  return (int8_t)*mem_ptr(memory, addr);
}

static __attribute__((always_inline)) inline uint16_t rd_mem_u16(
    uint8_t* restrict memory, uint64_t addr) {
  check_mem_bounds_u16(addr);
  uint16_t v;
  memcpy(&v, mem_ptr(memory, addr), sizeof(v));
  return v;
}

static __attribute__((always_inline)) inline int16_t rd_mem_i16(
    uint8_t* restrict memory, uint64_t addr) {
  check_mem_bounds_i16(addr);
  int16_t v;
  memcpy(&v, mem_ptr(memory, addr), sizeof(v));
  return v;
}

static __attribute__((always_inline)) inline int32_t rd_mem_i32(
    uint8_t* restrict memory, uint64_t addr) {
  check_mem_bounds_u32(addr);
  int32_t v;
  memcpy(&v, mem_ptr(memory, addr), sizeof(v));
  return v;
}

static __attribute__((always_inline)) inline uint32_t rd_mem_u32(
    uint8_t* restrict memory, uint64_t addr) {
  check_mem_bounds_u32(addr);
  uint32_t v;
  memcpy(&v, mem_ptr(memory, addr), sizeof(v));
  return v;
}

static __attribute__((always_inline)) inline uint64_t rd_mem_u64(
    uint8_t* restrict memory, uint64_t addr) {
  check_mem_bounds_u64(addr);
  uint64_t v;
  memcpy(&v, mem_ptr(memory, addr), sizeof(v));
  return v;
}

/* ── Per-width memory writes ─────────────────────────────────────── */

static __attribute__((always_inline)) inline void wr_mem_u8(
    uint8_t* restrict memory, uint64_t addr, uint8_t val) {
  check_mem_bounds_u8(addr);
  *mem_ptr(memory, addr) = val;
}

static __attribute__((always_inline)) inline void wr_mem_u16(
    uint8_t* restrict memory, uint64_t addr, uint16_t val) {
  check_mem_bounds_u16(addr);
  memcpy(mem_ptr(memory, addr), &val, sizeof(val));
}

static __attribute__((always_inline)) inline void wr_mem_u32(
    uint8_t* restrict memory, uint64_t addr, uint32_t val) {
  check_mem_bounds_u32(addr);
  memcpy(mem_ptr(memory, addr), &val, sizeof(val));
}

static __attribute__((always_inline)) inline void wr_mem_u64(
    uint8_t* restrict memory, uint64_t addr, uint64_t val) {
  check_mem_bounds_u64(addr);
  memcpy(mem_ptr(memory, addr), &val, sizeof(val));
}

/* ── Word-aligned range memory access ────────────────────────────── */

/* `base_addr` is word-aligned; the guest pointer is word-aligned too,
 * so these lower to a single memcpy. */
static __attribute__((always_inline)) inline void rd_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, uint64_t* restrict out,
    uint32_t num_words) {
  check_mem_bounds_u64_range(base_addr, num_words);
  const void* p =
      __builtin_assume_aligned(mem_ptr(state->memory, base_addr), sizeof(uint64_t));
  memcpy(out, p, (size_t)num_words * sizeof(uint64_t));
}

static __attribute__((always_inline)) inline void wr_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, const uint64_t* restrict vals,
    uint32_t num_words) {
  check_mem_bounds_u64_range(base_addr, num_words);
  void* p =
      __builtin_assume_aligned(mem_ptr(state->memory, base_addr), sizeof(uint64_t));
  memcpy(p, vals, (size_t)num_words * sizeof(uint64_t));
}

/* ── Traced memory helpers ───────────────────────────────────────── */

static __attribute__((always_inline)) inline void trace_rd_mem_u8(
    RvState* restrict state, uint64_t addr, uint8_t val);
static __attribute__((always_inline)) inline void trace_rd_mem_i8(
    RvState* restrict state, uint64_t addr, int8_t val);
static __attribute__((always_inline)) inline void trace_rd_mem_u16(
    RvState* restrict state, uint64_t addr, uint16_t val);
static __attribute__((always_inline)) inline void trace_rd_mem_i16(
    RvState* restrict state, uint64_t addr, int16_t val);
static __attribute__((always_inline)) inline void trace_rd_mem_u32(
    RvState* restrict state, uint64_t addr, uint32_t val);
static __attribute__((always_inline)) inline void trace_rd_mem_i32(
    RvState* restrict state, uint64_t addr, int32_t val);
static __attribute__((always_inline)) inline void trace_rd_mem_u64(
    RvState* restrict state, uint64_t addr, uint64_t val);
static __attribute__((always_inline)) inline void trace_wr_mem_u8(
    RvState* restrict state, uint64_t addr, uint8_t val);
static __attribute__((always_inline)) inline void trace_wr_mem_u16(
    RvState* restrict state, uint64_t addr, uint16_t val);
static __attribute__((always_inline)) inline void trace_wr_mem_u32(
    RvState* restrict state, uint64_t addr, uint32_t val);
static __attribute__((always_inline)) inline void trace_wr_mem_u64(
    RvState* restrict state, uint64_t addr, uint64_t val);
static __attribute__((always_inline)) inline void trace_rd_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, const uint64_t* vals,
    uint32_t num_words);
static __attribute__((always_inline)) inline void trace_wr_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, const uint64_t* vals,
    uint32_t num_words);

/* Per-width traced reads return a 32-bit C type; callers widen to uint64_t at
 * register assignment via C implicit promotion. */
static __attribute__((always_inline)) inline uint32_t rd_mem_u8_traced(
    RvState* restrict state, uint64_t addr) {
  uint8_t v = rd_mem_u8(state->memory, addr);
  trace_rd_mem_u8(state, addr, v);
  return v;
}

static __attribute__((always_inline)) inline int32_t rd_mem_i8_traced(
    RvState* restrict state, uint64_t addr) {
  int8_t v = rd_mem_i8(state->memory, addr);
  trace_rd_mem_i8(state, addr, v);
  return v;
}

static __attribute__((always_inline)) inline uint32_t rd_mem_u16_traced(
    RvState* restrict state, uint64_t addr) {
  uint16_t v = rd_mem_u16(state->memory, addr);
  trace_rd_mem_u16(state, addr, v);
  return v;
}

static __attribute__((always_inline)) inline int32_t rd_mem_i16_traced(
    RvState* restrict state, uint64_t addr) {
  int16_t v = rd_mem_i16(state->memory, addr);
  trace_rd_mem_i16(state, addr, v);
  return v;
}

static __attribute__((always_inline)) inline int32_t rd_mem_i32_traced(
    RvState* restrict state, uint64_t addr) {
  int32_t v = rd_mem_i32(state->memory, addr);
  trace_rd_mem_i32(state, addr, v);
  return v;
}

static __attribute__((always_inline)) inline uint32_t rd_mem_u32_traced(
    RvState* restrict state, uint64_t addr) {
  uint32_t v = rd_mem_u32(state->memory, addr);
  trace_rd_mem_u32(state, addr, v);
  return v;
}

static __attribute__((always_inline)) inline uint64_t rd_mem_u64_traced(
    RvState* restrict state, uint64_t addr) {
  uint64_t v = rd_mem_u64(state->memory, addr);
  trace_rd_mem_u64(state, addr, v);
  return v;
}

static __attribute__((always_inline)) inline void wr_mem_u8_traced(
    RvState* restrict state, uint64_t addr, uint8_t val) {
  trace_wr_mem_u8(state, addr, val);
  wr_mem_u8(state->memory, addr, val);
}

static __attribute__((always_inline)) inline void wr_mem_u16_traced(
    RvState* restrict state, uint64_t addr, uint16_t val) {
  trace_wr_mem_u16(state, addr, val);
  wr_mem_u16(state->memory, addr, val);
}

static __attribute__((always_inline)) inline void wr_mem_u32_traced(
    RvState* restrict state, uint64_t addr, uint32_t val) {
  trace_wr_mem_u32(state, addr, val);
  wr_mem_u32(state->memory, addr, val);
}

static __attribute__((always_inline)) inline void wr_mem_u64_traced(
    RvState* restrict state, uint64_t addr, uint64_t val) {
  trace_wr_mem_u64(state, addr, val);
  wr_mem_u64(state->memory, addr, val);
}

static __attribute__((always_inline)) inline void rd_mem_u64_range_traced(
    RvState* restrict state, uint64_t base_addr, uint64_t* restrict out,
    uint32_t num_words) {
  rd_mem_u64_range(state, base_addr, out, num_words);
  trace_rd_mem_u64_range(state, base_addr, out, num_words);
}

static __attribute__((always_inline)) inline void wr_mem_u64_range_traced(
    RvState* restrict state, uint64_t base_addr, const uint64_t* restrict vals,
    uint32_t num_words) {
  trace_wr_mem_u64_range(state, base_addr, vals, num_words);
  wr_mem_u64_range(state, base_addr, vals, num_words);
}

#endif /* OPENVM_STATE_H */
