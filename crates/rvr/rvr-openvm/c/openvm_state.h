/* RvState struct, shared memory/register access, and common constants.
 *
 * The Tracer type is forward-declared; include the appropriate tracer
 * header to get the full definition and tracing functions.
 */

#ifndef OPENVM_STATE_H
#define OPENVM_STATE_H

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "openvm_constants.h"

/* Branch prediction hints. */
static __attribute__((always_inline)) inline int likely(int x) { return __builtin_expect(!!(x), 1); }
static __attribute__((always_inline)) inline int unlikely(int x) { return __builtin_expect(!!(x), 0); }
static __attribute__((always_inline)) inline void assume(int x) { __builtin_assume(x); }
static __attribute__((always_inline)) inline void debug_assume(int x) {
  assert(x);
  assume(x);
}

struct Tracer;

typedef enum OpenVmExecStatus : uint8_t {
  OPENVM_EXEC_RUNNING = 0,
  OPENVM_EXEC_TERMINATED = 1,
  OPENVM_EXEC_SUSPENDED = 2,
  OPENVM_EXEC_TRAPPED = 3,
} OpenVmExecStatus;

static_assert(OPENVM_EXEC_RUNNING == 0, "must match rvr_state::ExecutionStatus::Running");
static_assert(OPENVM_EXEC_TERMINATED == 1, "must match rvr_state::ExecutionStatus::Terminated");
static_assert(OPENVM_EXEC_SUSPENDED == 2, "must match rvr_state::ExecutionStatus::Suspended");
static_assert(OPENVM_EXEC_TRAPPED == 3, "must match rvr_state::ExecutionStatus::Trapped");

typedef struct RvState {
  uint32_t regs[32];
  uint32_t pc;
  uint64_t instret;
  uint64_t target_instret;
  uint32_t reservation_addr;
  uint8_t reservation_valid;
  /* Field name preserved for ABI compatibility with rvr's RvState.
   * Semantically stores an OpenVmExecStatus value. */
  uint8_t has_exited;
  uint8_t exit_code;
  uint8_t _pad0;
  uint32_t brk;
  uint32_t start_brk;
  uint8_t* memory;
  struct Tracer* tracer;
  uint32_t csrs[4096];
} RvState;

/* OpenVM address spaces. */
static constexpr uint32_t AS_REGISTER = 1;
static constexpr uint32_t AS_MEMORY = 2;
static constexpr uint32_t AS_PUBLIC_VALUES = 3;
static constexpr uint32_t AS_DEFERRAL = 4;

/* Guest word size in bytes. */
static constexpr uint32_t WORD_SIZE = 4;

/* ── Guest memory pointer ────────────────────────────────────────── */

/* GuardedMemory's mmap guard pages catch OOB. Buffer is page-aligned,
 * so guest multi-byte accesses are naturally aligned.
 * TODO: addr &= MEMORY_MASK for defense-in-depth. */
static __attribute__((always_inline)) inline uint8_t* mem_ptr(RvState* restrict state, uint32_t addr) {
  assume(addr <= MEMORY_MASK);
  return state->memory + addr;
}

/* ── Register access ─────────────────────────────────────────────── */

static __attribute__((always_inline)) inline uint32_t reg_read(RvState* restrict state, uint8_t idx) { return state->regs[idx]; }

static __attribute__((always_inline)) inline void reg_write(RvState* restrict state, uint8_t idx, uint32_t val) {
  state->regs[idx] = val;
}

/* ── Per-width memory reads ──────────────────────────────────────── */

static __attribute__((always_inline)) inline uint8_t rd_mem_u8(RvState* restrict state, uint32_t addr) {
  return *mem_ptr(state, addr);
}

static __attribute__((always_inline)) inline int8_t rd_mem_i8(RvState* restrict state, uint32_t addr) {
  return (int8_t)*mem_ptr(state, addr);
}

static __attribute__((always_inline)) inline uint16_t rd_mem_u16(RvState* restrict state, uint32_t addr) {
  uint16_t v;
  const void* p = __builtin_assume_aligned(mem_ptr(state, addr), sizeof(v));
  memcpy(&v, p, sizeof(v));
  return v;
}

static __attribute__((always_inline)) inline int16_t rd_mem_i16(RvState* restrict state, uint32_t addr) {
  int16_t v;
  const void* p = __builtin_assume_aligned(mem_ptr(state, addr), sizeof(v));
  memcpy(&v, p, sizeof(v));
  return v;
}

static __attribute__((always_inline)) inline uint32_t rd_mem_u32(RvState* restrict state, uint32_t addr) {
  uint32_t v;
  const void* p = __builtin_assume_aligned(mem_ptr(state, addr), sizeof(v));
  memcpy(&v, p, sizeof(v));
  return v;
}

/* ── Per-width memory writes ─────────────────────────────────────── */

static __attribute__((always_inline)) inline void wr_mem_u8(RvState* restrict state, uint32_t addr, uint8_t val) {
  *mem_ptr(state, addr) = val;
}

static __attribute__((always_inline)) inline void wr_mem_u16(RvState* restrict state, uint32_t addr, uint16_t val) {
  void* p = __builtin_assume_aligned(mem_ptr(state, addr), sizeof(val));
  memcpy(p, &val, sizeof(val));
}

static __attribute__((always_inline)) inline void wr_mem_u32(RvState* restrict state, uint32_t addr, uint32_t val) {
  void* p = __builtin_assume_aligned(mem_ptr(state, addr), sizeof(val));
  memcpy(p, &val, sizeof(val));
}

/* ── Word-aligned range memory access ────────────────────────────── */

/* `base_addr` is word-aligned; the guest pointer is word-aligned too,
 * so these lower to a single memcpy. */
static __attribute__((always_inline)) inline void rd_mem_u32_range(RvState* restrict state, uint32_t base_addr,
                                                                   uint32_t* restrict out, uint32_t num_words) {
  const void* p = __builtin_assume_aligned(mem_ptr(state, base_addr), WORD_SIZE);
  memcpy(out, p, (size_t)num_words * sizeof(uint32_t));
}

static __attribute__((always_inline)) inline void wr_mem_u32_range(RvState* restrict state, uint32_t base_addr,
                                                                   const uint32_t* restrict vals, uint32_t num_words) {
  void* p = __builtin_assume_aligned(mem_ptr(state, base_addr), WORD_SIZE);
  memcpy(p, vals, (size_t)num_words * sizeof(uint32_t));
}

/* ── Traced memory helpers ───────────────────────────────────────── */

static __attribute__((always_inline)) inline void trace_rd_mem_u8(RvState* restrict state, uint32_t addr, uint8_t val);
static __attribute__((always_inline)) inline void trace_rd_mem_i8(RvState* restrict state, uint32_t addr, int8_t val);
static __attribute__((always_inline)) inline void trace_rd_mem_u16(RvState* restrict state, uint32_t addr, uint16_t val);
static __attribute__((always_inline)) inline void trace_rd_mem_i16(RvState* restrict state, uint32_t addr, int16_t val);
static __attribute__((always_inline)) inline void trace_rd_mem_u32(RvState* restrict state, uint32_t addr, uint32_t val);
static __attribute__((always_inline)) inline void trace_wr_mem_u8(RvState* restrict state, uint32_t addr, uint8_t val);
static __attribute__((always_inline)) inline void trace_wr_mem_u16(RvState* restrict state, uint32_t addr, uint16_t val);
static __attribute__((always_inline)) inline void trace_wr_mem_u32(RvState* restrict state, uint32_t addr, uint32_t val);
static __attribute__((always_inline)) inline void trace_rd_mem_u32_range(RvState* restrict state, uint32_t base_addr,
                                                                         const uint32_t* vals, uint32_t num_words);
static __attribute__((always_inline)) inline void trace_wr_mem_u32_range(RvState* restrict state, uint32_t base_addr,
                                                                         const uint32_t* vals, uint32_t num_words);

/* Per-width traced reads widen to 32 bits. */
static __attribute__((always_inline)) inline uint32_t rd_mem_u8_traced(RvState* restrict state, uint32_t addr) {
  uint8_t v = rd_mem_u8(state, addr);
  trace_rd_mem_u8(state, addr, v);
  return v;
}

static __attribute__((always_inline)) inline int32_t rd_mem_i8_traced(RvState* restrict state, uint32_t addr) {
  int8_t v = rd_mem_i8(state, addr);
  trace_rd_mem_i8(state, addr, v);
  return v;
}

static __attribute__((always_inline)) inline uint32_t rd_mem_u16_traced(RvState* restrict state, uint32_t addr) {
  uint16_t v = rd_mem_u16(state, addr);
  trace_rd_mem_u16(state, addr, v);
  return v;
}

static __attribute__((always_inline)) inline int32_t rd_mem_i16_traced(RvState* restrict state, uint32_t addr) {
  int16_t v = rd_mem_i16(state, addr);
  trace_rd_mem_i16(state, addr, v);
  return v;
}

static __attribute__((always_inline)) inline uint32_t rd_mem_u32_traced(RvState* restrict state, uint32_t addr) {
  uint32_t v = rd_mem_u32(state, addr);
  trace_rd_mem_u32(state, addr, v);
  return v;
}

static __attribute__((always_inline)) inline void wr_mem_u8_traced(RvState* restrict state, uint32_t addr, uint8_t val) {
  trace_wr_mem_u8(state, addr, val);
  wr_mem_u8(state, addr, val);
}

static __attribute__((always_inline)) inline void wr_mem_u16_traced(RvState* restrict state, uint32_t addr, uint16_t val) {
  trace_wr_mem_u16(state, addr, val);
  wr_mem_u16(state, addr, val);
}

static __attribute__((always_inline)) inline void wr_mem_u32_traced(RvState* restrict state, uint32_t addr, uint32_t val) {
  trace_wr_mem_u32(state, addr, val);
  wr_mem_u32(state, addr, val);
}

static __attribute__((always_inline)) inline void rd_mem_u32_range_traced(RvState* restrict state, uint32_t base_addr,
                                                                          uint32_t* restrict out, uint32_t num_words) {
  rd_mem_u32_range(state, base_addr, out, num_words);
  trace_rd_mem_u32_range(state, base_addr, out, num_words);
}

static __attribute__((always_inline)) inline void wr_mem_u32_range_traced(RvState* restrict state, uint32_t base_addr,
                                                                          const uint32_t* restrict vals, uint32_t num_words) {
  trace_wr_mem_u32_range(state, base_addr, vals, num_words);
  wr_mem_u32_range(state, base_addr, vals, num_words);
}

#endif /* OPENVM_STATE_H */
