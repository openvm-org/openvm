/* OpenVM preflight tracer.
 *
 * Appends program and memory events into host-provided buffers. The buffers
 * are preallocated by Rust and aliased through RvState->tracer for the duration
 * of execution.
 *
 * R1 (self-contained events): every memory event carries its own
 * `prev_timestamp` and `prev_value` — the block's previous-access timestamp and
 * value. `prev_timestamp` comes from a per-address-space timestamp shadow (the
 * C mirror of the interpreter's `TracingMemory.meta`); `prev_value` is the
 * block's stored value read just before this access mutates it. The host side
 * therefore never replays the log to recover memory-record aux data. The first
 * time a block is touched this segment it is appended to `touched` so the host
 * can finalize `touched_memory` in O(touched-blocks) instead of O(accesses).
 *
 * Only the U16-cell address spaces (register, main memory, public values) are
 * traced through this path, so every block is WORD_SIZE (8) bytes and
 * `prev_value` fits a uint64_t. Deferral stores (16-byte blocks) are rejected
 * to the interpreter route upstream and never reach this tracer.
 */

#ifndef OPENVM_TRACER_PREFLIGHT_H
#define OPENVM_TRACER_PREFLIGHT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "openvm_state.h"

typedef struct ProgramLogEntry {
  uint16_t opcode;
  uint16_t _pad0;
  uint32_t timestamp;
  uint64_t pc;
} ProgramLogEntry;

typedef struct MemoryLogEntry {
  uint32_t timestamp;
  uint32_t prev_timestamp;
  uint8_t kind;
  uint8_t addr_space;
  uint8_t width;
  uint8_t _pad0;
  uint32_t _pad1;
  uint64_t address;
  uint64_t value;
  uint64_t prev_value;
} MemoryLogEntry;

/* A block touched (for the first time) this segment. `block_addr` is the
 * block-aligned byte address; the host derives the AS-native block pointer and
 * reads the final value from live memory + the final timestamp from the
 * shadow. */
typedef struct TouchedBlock {
  uint32_t addr_space;
  uint32_t block_addr;
} TouchedBlock;

typedef struct Tracer {
  ProgramLogEntry* program_log;
  MemoryLogEntry* memory_log;
  uint32_t* chip_counts;
  /* Per-address-space last-access timestamp shadows, indexed by block index
   * (block_addr / WORD_SIZE). A value of 0 means "untouched this segment". */
  uint32_t* shadow_register;
  uint32_t* shadow_memory;
  uint32_t* shadow_public_values;
  /* Public-values byte buffer, aliased so reveal writes can read the block's
   * previous value for `prev_value`. Registers and main memory read theirs
   * from `state->regs` / `state->memory`. */
  uint8_t* public_values_base;
  TouchedBlock* touched;
  uint32_t program_log_len;
  uint32_t memory_log_len;
  uint32_t program_log_cap;
  uint32_t memory_log_cap;
  uint32_t chip_counts_len;
  uint32_t touched_len;
  uint32_t touched_cap;
  uint32_t timestamp;
} Tracer;

_Static_assert(sizeof(ProgramLogEntry) == PREFLIGHT_PROGRAM_LOG_ENTRY_SIZE,
               "ProgramLogEntry size drift");
_Static_assert(_Alignof(ProgramLogEntry) == PREFLIGHT_PROGRAM_LOG_ENTRY_ALIGN,
               "ProgramLogEntry align drift");
_Static_assert(offsetof(ProgramLogEntry, opcode) == 0,
               "ProgramLogEntry opcode offset drift");
_Static_assert(offsetof(ProgramLogEntry, _pad0) == 2,
               "ProgramLogEntry _pad0 offset drift");
_Static_assert(offsetof(ProgramLogEntry, timestamp) == 4,
               "ProgramLogEntry timestamp offset drift");
_Static_assert(offsetof(ProgramLogEntry, pc) == 8,
               "ProgramLogEntry pc offset drift");
_Static_assert(sizeof(MemoryLogEntry) == PREFLIGHT_MEMORY_LOG_ENTRY_SIZE,
               "MemoryLogEntry size drift");
_Static_assert(_Alignof(MemoryLogEntry) == PREFLIGHT_MEMORY_LOG_ENTRY_ALIGN,
               "MemoryLogEntry align drift");
_Static_assert(offsetof(MemoryLogEntry, timestamp) == 0,
               "MemoryLogEntry timestamp offset drift");
_Static_assert(offsetof(MemoryLogEntry, prev_timestamp) == 4,
               "MemoryLogEntry prev_timestamp offset drift");
_Static_assert(offsetof(MemoryLogEntry, kind) == 8,
               "MemoryLogEntry kind offset drift");
_Static_assert(offsetof(MemoryLogEntry, addr_space) == 9,
               "MemoryLogEntry addr_space offset drift");
_Static_assert(offsetof(MemoryLogEntry, width) == 10,
               "MemoryLogEntry width offset drift");
_Static_assert(offsetof(MemoryLogEntry, _pad0) == 11,
               "MemoryLogEntry _pad0 offset drift");
_Static_assert(offsetof(MemoryLogEntry, address) == 16,
               "MemoryLogEntry address offset drift");
_Static_assert(offsetof(MemoryLogEntry, value) == 24,
               "MemoryLogEntry value offset drift");
_Static_assert(offsetof(MemoryLogEntry, prev_value) == 32,
               "MemoryLogEntry prev_value offset drift");
_Static_assert(sizeof(TouchedBlock) == PREFLIGHT_TOUCHED_BLOCK_SIZE,
               "TouchedBlock size drift");
_Static_assert(_Alignof(TouchedBlock) == PREFLIGHT_TOUCHED_BLOCK_ALIGN,
               "TouchedBlock align drift");
_Static_assert(offsetof(TouchedBlock, addr_space) == 0,
               "TouchedBlock addr_space offset drift");
_Static_assert(offsetof(TouchedBlock, block_addr) == 4,
               "TouchedBlock block_addr offset drift");
_Static_assert(sizeof(Tracer) == PREFLIGHT_TRACER_DATA_SIZE,
               "Preflight Tracer size drift");
_Static_assert(_Alignof(Tracer) == PREFLIGHT_TRACER_DATA_ALIGN,
               "Preflight Tracer align drift");
_Static_assert(offsetof(Tracer, program_log) == 0,
               "Tracer program_log offset drift");
_Static_assert(offsetof(Tracer, memory_log) == 8,
               "Tracer memory_log offset drift");
_Static_assert(offsetof(Tracer, chip_counts) == 16,
               "Tracer chip_counts offset drift");
_Static_assert(offsetof(Tracer, shadow_register) == 24,
               "Tracer shadow_register offset drift");
_Static_assert(offsetof(Tracer, shadow_memory) == 32,
               "Tracer shadow_memory offset drift");
_Static_assert(offsetof(Tracer, shadow_public_values) == 40,
               "Tracer shadow_public_values offset drift");
_Static_assert(offsetof(Tracer, public_values_base) == 48,
               "Tracer public_values_base offset drift");
_Static_assert(offsetof(Tracer, touched) == 56,
               "Tracer touched offset drift");
_Static_assert(offsetof(Tracer, program_log_len) == 64,
               "Tracer program_log_len offset drift");
_Static_assert(offsetof(Tracer, memory_log_len) == 68,
               "Tracer memory_log_len offset drift");
_Static_assert(offsetof(Tracer, program_log_cap) == 72,
               "Tracer program_log_cap offset drift");
_Static_assert(offsetof(Tracer, memory_log_cap) == 76,
               "Tracer memory_log_cap offset drift");
_Static_assert(offsetof(Tracer, chip_counts_len) == 80,
               "Tracer chip_counts_len offset drift");
_Static_assert(offsetof(Tracer, touched_len) == 84,
               "Tracer touched_len offset drift");
_Static_assert(offsetof(Tracer, touched_cap) == 88,
               "Tracer touched_cap offset drift");
_Static_assert(offsetof(Tracer, timestamp) == 92,
               "Tracer timestamp offset drift");

/* ── Timestamp shadow ─────────────────────────────────────────────── */

/* The traced address spaces all use WORD_SIZE-byte blocks; select the shadow
 * array for `addr_space`. Only register / main-memory / public-values reach
 * this tracer (deferral stores are routed to the interpreter upstream). */
static __attribute__((always_inline)) inline uint32_t* preflight_shadow_for(
    Tracer* restrict t, uint8_t addr_space) {
  if (addr_space == AS_REGISTER) {
    return t->shadow_register;
  }
  if (addr_space == AS_PUBLIC_VALUES) {
    return t->shadow_public_values;
  }
  return t->shadow_memory;
}

/* ── Append helpers ───────────────────────────────────────────────── */

static __attribute__((always_inline)) inline void preflight_append_program(
    Tracer* restrict t, uint64_t pc) {
  uint32_t idx = t->program_log_len++;
  if (likely(idx < t->program_log_cap)) {
    ProgramLogEntry entry = {
        .opcode = 0,
        ._pad0 = 0,
        .pc = pc,
        .timestamp = t->timestamp,
    };
    t->program_log[idx] = entry;
  }
}

static __attribute__((always_inline)) inline uint64_t preflight_block_addr(
    uint64_t addr) {
  return addr & ~(uint64_t)(WORD_SIZE - 1u);
}

static __attribute__((always_inline)) inline uint64_t preflight_read_mem_block(
    RvState* restrict state, uint64_t addr) {
  return rd_mem_u64(state->memory, preflight_block_addr(addr));
}

static __attribute__((always_inline)) inline uint64_t preflight_read_pv_block(
    Tracer* restrict t, uint64_t block_addr) {
  uint64_t v;
  memcpy(&v, t->public_values_base + block_addr, sizeof(v));
  return v;
}

static __attribute__((always_inline)) inline uint64_t preflight_patch_mem_block(
    uint64_t block, uint64_t addr, uint8_t width, uint64_t value) {
  uint32_t shift = (addr & (WORD_SIZE - 1u)) * 8u;
  uint64_t mask = width == WORD_SIZE ? UINT64_MAX
                                     : ((1ull << (width * 8u)) - 1ull);
  return (block & ~(mask << shift)) | ((value & mask) << shift);
}

/* Append one self-contained memory event. `address` need not be block-aligned;
 * the shadow index and touched-block key are derived from the aligned block.
 * `prev_value` is the block's value before this access (only consumed for
 * writes) and is supplied by the caller, which holds the store pointer. */
static __attribute__((always_inline)) inline void preflight_append_memory(
    Tracer* restrict t, uint8_t kind, uint8_t addr_space, uint64_t address,
    uint8_t width, uint64_t value, uint64_t prev_value) {
  uint32_t timestamp = t->timestamp++;
  uint64_t block_addr = address & ~(uint64_t)(WORD_SIZE - 1u);
  uint32_t block_idx = (uint32_t)(block_addr / WORD_SIZE);

  uint32_t* restrict shadow = preflight_shadow_for(t, addr_space);
  uint32_t prev_timestamp = shadow[block_idx];
  shadow[block_idx] = timestamp;
  if (prev_timestamp == 0u) {
    uint32_t ti = t->touched_len++;
    if (likely(ti < t->touched_cap)) {
      t->touched[ti].addr_space = addr_space;
      t->touched[ti].block_addr = (uint32_t)block_addr;
    }
  }

  uint32_t idx = t->memory_log_len++;
  if (likely(idx < t->memory_log_cap)) {
    MemoryLogEntry entry = {
        .timestamp = timestamp,
        .prev_timestamp = prev_timestamp,
        .kind = kind,
        .addr_space = addr_space,
        .width = width,
        ._pad0 = 0,
        ._pad1 = 0,
        .address = address,
        .value = value,
        .prev_value = prev_value,
    };
    t->memory_log[idx] = entry;
  }
}

static __attribute__((always_inline)) inline void trace_timestamp(
    RvState* restrict state) {
  state->tracer->timestamp++;
}

/* ── Trace-only register access ──────────────────────────────────── */

static __attribute__((always_inline)) inline void trace_reg_read(
    RvState* restrict state, uint8_t idx, uint32_t val) {
  uint64_t reg_value = idx == 0 ? 0 : state->regs[idx];
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ,
                          AS_REGISTER, (uint32_t)idx * WORD_SIZE, WORD_SIZE,
                          reg_value, reg_value);
}
/* Traced BEFORE the register store (see `write_reg` codegen), so
 * `state->regs[idx]` still holds the previous value for `prev_value`; the new
 * value arrives as `new_val`. */
static __attribute__((always_inline)) inline void trace_reg_write(
    RvState* restrict state, uint8_t idx, uint32_t new_val) {
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE,
                          AS_REGISTER, (uint32_t)idx * WORD_SIZE, WORD_SIZE,
                          new_val, state->regs[idx]);
}

/* ── Trace-only memory reads ─────────────────────────────────────── */

static __attribute__((always_inline)) inline void trace_rd_mem_u8(
    RvState* restrict state, uint64_t addr, uint8_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t block = preflight_read_mem_block(state, block_addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                          block_addr, WORD_SIZE, block, block);
}
static __attribute__((always_inline)) inline void trace_rd_mem_i8(
    RvState* restrict state, uint64_t addr, int8_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t block = preflight_read_mem_block(state, block_addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                          block_addr, WORD_SIZE, block, block);
}
static __attribute__((always_inline)) inline void trace_rd_mem_u16(
    RvState* restrict state, uint64_t addr, uint16_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t block = preflight_read_mem_block(state, block_addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                          block_addr, WORD_SIZE, block, block);
}
static __attribute__((always_inline)) inline void trace_rd_mem_i16(
    RvState* restrict state, uint64_t addr, int16_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t block = preflight_read_mem_block(state, block_addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                          block_addr, WORD_SIZE, block, block);
}
static __attribute__((always_inline)) inline void trace_rd_mem_u32(
    RvState* restrict state, uint64_t addr, uint32_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t block = preflight_read_mem_block(state, block_addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                          block_addr, WORD_SIZE, block, block);
}
static __attribute__((always_inline)) inline void trace_rd_mem_i32(
    RvState* restrict state, uint64_t addr, int32_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t block = preflight_read_mem_block(state, block_addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                          block_addr, WORD_SIZE, block, block);
}
static __attribute__((always_inline)) inline void trace_rd_mem_u64(
    RvState* restrict state, uint64_t addr, uint64_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t block = preflight_read_mem_block(state, block_addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                          block_addr, WORD_SIZE, block, block);
}

/* ── Trace-only memory writes ────────────────────────────────────── */

static __attribute__((always_inline)) inline void trace_wr_mem_u8(
    RvState* restrict state, uint64_t addr, uint8_t new_val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t prev_block = preflight_read_mem_block(state, block_addr);
  uint64_t block =
      preflight_patch_mem_block(prev_block, addr, 1, (uint64_t)new_val);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, AS_MEMORY,
                          block_addr, WORD_SIZE, block, prev_block);
}
static __attribute__((always_inline)) inline void trace_wr_mem_u16(
    RvState* restrict state, uint64_t addr, uint16_t new_val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t prev_block = preflight_read_mem_block(state, block_addr);
  uint64_t block =
      preflight_patch_mem_block(prev_block, addr, 2, (uint64_t)new_val);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, AS_MEMORY,
                          block_addr, WORD_SIZE, block, prev_block);
}
static __attribute__((always_inline)) inline void trace_wr_mem_u32(
    RvState* restrict state, uint64_t addr, uint32_t new_val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t prev_block = preflight_read_mem_block(state, block_addr);
  uint64_t block =
      preflight_patch_mem_block(prev_block, addr, 4, (uint64_t)new_val);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, AS_MEMORY,
                          block_addr, WORD_SIZE, block, prev_block);
}
static __attribute__((always_inline)) inline void trace_wr_mem_u64(
    RvState* restrict state, uint64_t addr, uint64_t new_val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t prev_block = preflight_read_mem_block(state, block_addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, AS_MEMORY,
                          block_addr, WORD_SIZE, new_val, prev_block);
}

/* ── Trace-only word-range memory access ─────────────────────────── */

static __attribute__((always_inline)) inline void trace_rd_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, const uint64_t* vals,
    uint32_t num_words) {
  /* Reads leave the block unchanged, so prev_value == value per word. */
  for (uint32_t i = 0; i < num_words; i++) {
    preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ,
                            AS_MEMORY, base_addr + i * WORD_SIZE, WORD_SIZE,
                            vals[i], vals[i]);
  }
}
static __attribute__((always_inline)) inline void trace_wr_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, const uint64_t* vals,
    uint32_t num_words) {
  /* Traced before the store (see `wr_mem_u64_range_traced`), so each block's
   * previous value is still readable from live memory. */
  for (uint32_t i = 0; i < num_words; i++) {
    uint64_t block_addr = base_addr + i * WORD_SIZE;
    uint64_t prev_block = preflight_read_mem_block(state, block_addr);
    preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE,
                            AS_MEMORY, block_addr, WORD_SIZE, vals[i],
                            prev_block);
  }
}

/* ── Trace-only operations ───────────────────────────────────────── */

static __attribute__((always_inline)) inline void trace_mem_access(
    RvState* restrict state, uint64_t addr, uint32_t addr_space) {
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_TOUCH,
                          (uint8_t)addr_space, addr, 0, 0, 0);
}

static __attribute__((always_inline)) inline void trace_mem_access_u64_range(
    RvState* restrict state, uint64_t base_addr, uint32_t num_dwords,
    uint32_t addr_space) {
  for (uint32_t i = 0; i < num_dwords; i++) {
    preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_TOUCH,
                            (uint8_t)addr_space, base_addr + i * WORD_SIZE,
                            WORD_SIZE, 0, 0);
  }
}

static __attribute__((always_inline)) inline void trace_wr_as_u64(
    RvState* restrict state, uint64_t addr, uint64_t new_val,
    uint32_t addr_space) {
  uint64_t block_addr = preflight_block_addr(addr);
  /* Public-values reveal writes are traced before the store, so the previous
   * value is still in the aliased public-values buffer. */
  uint64_t prev_block =
      addr_space == AS_PUBLIC_VALUES
          ? preflight_read_pv_block(state->tracer, block_addr)
          : 0u;
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE,
                          (uint8_t)addr_space, block_addr, WORD_SIZE, new_val,
                          prev_block);
}

static __attribute__((always_inline)) inline void trace_wr_as(
    RvState* restrict state, uint64_t addr, uint64_t new_val, uint32_t width,
    uint32_t addr_space) {
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE,
                          (uint8_t)addr_space, addr, (uint8_t)width, new_val);
}

static __attribute__((always_inline)) inline void trace_pc(
    RvState* restrict state, uint64_t pc) {
  preflight_append_program(state->tracer, pc);
}

static __attribute__((always_inline)) inline void trace_chip(
    RvState* restrict state, uint32_t chip_idx, uint32_t count) {
  if (likely(chip_idx < state->tracer->chip_counts_len)) {
    state->tracer->chip_counts[chip_idx] += count;
  }
}

static __attribute__((always_inline)) inline void trace_block(
    RvState* restrict state, uint64_t pc, uint32_t block_insn_count) {}

#endif /* OPENVM_TRACER_PREFLIGHT_H */
