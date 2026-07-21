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
  uint64_t offset = pc - RV_TEXT_START;
  return offset <= RV_TEXT_END - RV_TEXT_START && (offset & 3ull) == 0;
}

/* ── Guest memory pointer ────────────────────────────────────────── */

/* Guard regions immediately before and after guest memory catch accesses
 * that cross either boundary.
 * TODO: Mask addr with MEMORY_MASK as an extra safety measure.
 * Clang cannot prove the bounds used by these low-level memory and fixed-size
 * register-array helpers. */
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

static __attribute__((always_inline)) inline uint8_t read_mem_u8(
    uint8_t* restrict memory, uint64_t addr) {
  check_mem_bounds_u8(addr);
  return *mem_ptr(memory, addr);
}

static __attribute__((always_inline)) inline int8_t read_mem_i8(
    uint8_t* restrict memory, uint64_t addr) {
  check_mem_bounds_i8(addr);
  return (int8_t)*mem_ptr(memory, addr);
}

static __attribute__((always_inline)) inline uint16_t read_mem_u16(
    uint8_t* restrict memory, uint64_t addr) {
  check_mem_bounds_u16(addr);
  uint16_t v;
  memcpy(&v, mem_ptr(memory, addr), sizeof(v));
  return v;
}

static __attribute__((always_inline)) inline int16_t read_mem_i16(
    uint8_t* restrict memory, uint64_t addr) {
  check_mem_bounds_i16(addr);
  int16_t v;
  memcpy(&v, mem_ptr(memory, addr), sizeof(v));
  return v;
}

static __attribute__((always_inline)) inline int32_t read_mem_i32(
    uint8_t* restrict memory, uint64_t addr) {
  check_mem_bounds_u32(addr);
  int32_t v;
  memcpy(&v, mem_ptr(memory, addr), sizeof(v));
  return v;
}

static __attribute__((always_inline)) inline uint32_t read_mem_u32(
    uint8_t* restrict memory, uint64_t addr) {
  check_mem_bounds_u32(addr);
  uint32_t v;
  memcpy(&v, mem_ptr(memory, addr), sizeof(v));
  return v;
}

static __attribute__((always_inline)) inline uint64_t read_mem_u64(
    uint8_t* restrict memory, uint64_t addr) {
  check_mem_bounds_u64(addr);
  uint64_t v;
  memcpy(&v, mem_ptr(memory, addr), sizeof(v));
  return v;
}

/* ── Per-width memory writes ─────────────────────────────────────── */

static __attribute__((always_inline)) inline void write_mem_u8(
    uint8_t* restrict memory, uint64_t addr, uint8_t val) {
  check_mem_bounds_u8(addr);
  *mem_ptr(memory, addr) = val;
}

static __attribute__((always_inline)) inline void write_mem_u16(
    uint8_t* restrict memory, uint64_t addr, uint16_t val) {
  check_mem_bounds_u16(addr);
  memcpy(mem_ptr(memory, addr), &val, sizeof(val));
}

static __attribute__((always_inline)) inline void write_mem_u32(
    uint8_t* restrict memory, uint64_t addr, uint32_t val) {
  check_mem_bounds_u32(addr);
  memcpy(mem_ptr(memory, addr), &val, sizeof(val));
}

static __attribute__((always_inline)) inline void write_mem_u64(
    uint8_t* restrict memory, uint64_t addr, uint64_t val) {
  check_mem_bounds_u64(addr);
  memcpy(mem_ptr(memory, addr), &val, sizeof(val));
}

/* ── Word-aligned range memory access ────────────────────────────── */

/* Data-only range helpers used to implement the execution-mode interfaces.
 * `base_addr` and the guest pointer are word-aligned, so these lower to one
 * memcpy. */
static __attribute__((always_inline)) inline void read_mem_u64_range_raw(
    RvState* restrict state, uint64_t base_addr, uint64_t* restrict out,
    uint32_t num_words) {
  check_mem_bounds_u64_range(base_addr, num_words);
  const void* p = __builtin_assume_aligned(mem_ptr(state->memory, base_addr),
                                           sizeof(uint64_t));
  memcpy(out, p, (size_t)num_words * sizeof(uint64_t));
}

static __attribute__((always_inline)) inline void write_mem_u64_range_raw(
    RvState* restrict state, uint64_t base_addr, const uint64_t* restrict vals,
    uint32_t num_words) {
  check_mem_bounds_u64_range(base_addr, num_words);
  void* p = __builtin_assume_aligned(mem_ptr(state->memory, base_addr),
                                     sizeof(uint64_t));
  memcpy(p, vals, (size_t)num_words * sizeof(uint64_t));
}

#endif /* OPENVM_STATE_H */
