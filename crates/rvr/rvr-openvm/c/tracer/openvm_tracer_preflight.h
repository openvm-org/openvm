/* OpenVM preflight tracer.
 *
 * Appends program and memory events into host-provided buffers. The buffers
 * are preallocated by Rust and aliased through RvState->tracer for the duration
 * of execution.
 */

#ifndef OPENVM_TRACER_PREFLIGHT_H
#define OPENVM_TRACER_PREFLIGHT_H

#include <stddef.h>
#include <stdint.h>

#include "openvm_state.h"

typedef struct ProgramLogEntry {
  uint16_t opcode;
  uint16_t _pad0;
  uint32_t timestamp;
  uint64_t pc;
} ProgramLogEntry;

typedef struct MemoryLogEntry {
  uint32_t timestamp;
  uint8_t kind;
  uint8_t addr_space;
  uint8_t width;
  uint8_t _pad0;
  uint64_t address;
  uint64_t value;
} MemoryLogEntry;

typedef struct Tracer {
  ProgramLogEntry* program_log;
  MemoryLogEntry* memory_log;
  uint32_t* chip_counts;
  uint32_t program_log_len;
  uint32_t memory_log_len;
  uint32_t program_log_cap;
  uint32_t memory_log_cap;
  uint32_t chip_counts_len;
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
_Static_assert(offsetof(MemoryLogEntry, kind) == 4,
               "MemoryLogEntry kind offset drift");
_Static_assert(offsetof(MemoryLogEntry, addr_space) == 5,
               "MemoryLogEntry addr_space offset drift");
_Static_assert(offsetof(MemoryLogEntry, width) == 6,
               "MemoryLogEntry width offset drift");
_Static_assert(offsetof(MemoryLogEntry, _pad0) == 7,
               "MemoryLogEntry _pad0 offset drift");
_Static_assert(offsetof(MemoryLogEntry, address) == 8,
               "MemoryLogEntry address offset drift");
_Static_assert(offsetof(MemoryLogEntry, value) == 16,
               "MemoryLogEntry value offset drift");
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
_Static_assert(offsetof(Tracer, program_log_len) == 24,
               "Tracer program_log_len offset drift");
_Static_assert(offsetof(Tracer, memory_log_len) == 28,
               "Tracer memory_log_len offset drift");
_Static_assert(offsetof(Tracer, program_log_cap) == 32,
               "Tracer program_log_cap offset drift");
_Static_assert(offsetof(Tracer, memory_log_cap) == 36,
               "Tracer memory_log_cap offset drift");
_Static_assert(offsetof(Tracer, chip_counts_len) == 40,
               "Tracer chip_counts_len offset drift");
_Static_assert(offsetof(Tracer, timestamp) == 44,
               "Tracer timestamp offset drift");

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

static __attribute__((always_inline)) inline void preflight_append_memory(
    Tracer* restrict t, uint8_t kind, uint8_t addr_space, uint64_t address,
    uint8_t width, uint64_t value) {
  uint32_t timestamp = t->timestamp++;
  uint32_t idx = t->memory_log_len++;
  if (likely(idx < t->memory_log_cap)) {
    MemoryLogEntry entry = {
        .timestamp = timestamp,
        .address = address,
        .value = value,
        .kind = kind,
        .addr_space = addr_space,
        .width = width,
        ._pad0 = 0,
    };
    t->memory_log[idx] = entry;
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

static __attribute__((always_inline)) inline uint64_t preflight_patch_mem_block(
    uint64_t block, uint64_t addr, uint8_t width, uint64_t value) {
  uint32_t shift = (addr & (WORD_SIZE - 1u)) * 8u;
  uint64_t mask = width == WORD_SIZE ? UINT64_MAX
                                     : ((1ull << (width * 8u)) - 1ull);
  return (block & ~(mask << shift)) | ((value & mask) << shift);
}

static __attribute__((always_inline)) inline void preflight_append_memory_range(
    Tracer* restrict t, uint8_t kind, uint8_t addr_space, uint64_t base_addr,
    const uint64_t* vals, uint32_t num_words) {
  for (uint32_t i = 0; i < num_words; i++) {
    preflight_append_memory(t, kind, addr_space, base_addr + i * WORD_SIZE,
                            WORD_SIZE, vals[i]);
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
                          reg_value);
}
static __attribute__((always_inline)) inline void trace_reg_write(
    RvState* restrict state, uint8_t idx, uint32_t new_val) {
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE,
                          AS_REGISTER, (uint32_t)idx * WORD_SIZE, WORD_SIZE,
                          state->regs[idx]);
}

/* ── Trace-only memory reads ─────────────────────────────────────── */

static __attribute__((always_inline)) inline void trace_rd_mem_u8(
    RvState* restrict state, uint64_t addr, uint8_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                          block_addr, WORD_SIZE,
                          preflight_read_mem_block(state, block_addr));
}
static __attribute__((always_inline)) inline void trace_rd_mem_i8(
    RvState* restrict state, uint64_t addr, int8_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                          block_addr, WORD_SIZE,
                          preflight_read_mem_block(state, block_addr));
}
static __attribute__((always_inline)) inline void trace_rd_mem_u16(
    RvState* restrict state, uint64_t addr, uint16_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                          block_addr, WORD_SIZE,
                          preflight_read_mem_block(state, block_addr));
}
static __attribute__((always_inline)) inline void trace_rd_mem_i16(
    RvState* restrict state, uint64_t addr, int16_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                          block_addr, WORD_SIZE,
                          preflight_read_mem_block(state, block_addr));
}
static __attribute__((always_inline)) inline void trace_rd_mem_u32(
    RvState* restrict state, uint64_t addr, uint32_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                          block_addr, WORD_SIZE,
                          preflight_read_mem_block(state, block_addr));
}
static __attribute__((always_inline)) inline void trace_rd_mem_i32(
    RvState* restrict state, uint64_t addr, int32_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                          block_addr, WORD_SIZE,
                          preflight_read_mem_block(state, block_addr));
}
static __attribute__((always_inline)) inline void trace_rd_mem_u64(
    RvState* restrict state, uint64_t addr, uint64_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                          block_addr, WORD_SIZE,
                          preflight_read_mem_block(state, block_addr));
}

/* ── Trace-only memory writes ────────────────────────────────────── */

static __attribute__((always_inline)) inline void trace_wr_mem_u8(
    RvState* restrict state, uint64_t addr, uint8_t new_val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t block = preflight_patch_mem_block(
      preflight_read_mem_block(state, block_addr), addr, 1, (uint64_t)new_val);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, AS_MEMORY,
                          block_addr, WORD_SIZE, block);
}
static __attribute__((always_inline)) inline void trace_wr_mem_u16(
    RvState* restrict state, uint64_t addr, uint16_t new_val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t block = preflight_patch_mem_block(
      preflight_read_mem_block(state, block_addr), addr, 2, (uint64_t)new_val);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, AS_MEMORY,
                          block_addr, WORD_SIZE, block);
}
static __attribute__((always_inline)) inline void trace_wr_mem_u32(
    RvState* restrict state, uint64_t addr, uint32_t new_val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t block = preflight_patch_mem_block(
      preflight_read_mem_block(state, block_addr), addr, 4, (uint64_t)new_val);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, AS_MEMORY,
                          block_addr, WORD_SIZE, block);
}
static __attribute__((always_inline)) inline void trace_wr_mem_u64(
    RvState* restrict state, uint64_t addr, uint64_t new_val) {
  uint64_t block_addr = preflight_block_addr(addr);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, AS_MEMORY,
                          block_addr, WORD_SIZE, new_val);
}

/* ── Trace-only word-range memory access ─────────────────────────── */

static __attribute__((always_inline)) inline void trace_rd_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, const uint64_t* vals,
    uint32_t num_words) {
  preflight_append_memory_range(state->tracer, PREFLIGHT_MEMORY_KIND_READ,
                                AS_MEMORY, base_addr, vals, num_words);
}
static __attribute__((always_inline)) inline void trace_wr_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, const uint64_t* vals,
    uint32_t num_words) {
  preflight_append_memory_range(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE,
                                AS_MEMORY, base_addr, vals, num_words);
}

/* ── Trace-only operations ───────────────────────────────────────── */

static __attribute__((always_inline)) inline void trace_mem_access(
    RvState* restrict state, uint64_t addr, uint32_t addr_space) {
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_TOUCH,
                          (uint8_t)addr_space, addr, 0, 0);
}

static __attribute__((always_inline)) inline void trace_mem_access_u64_range(
    RvState* restrict state, uint64_t base_addr, uint32_t num_dwords,
    uint32_t addr_space) {
  for (uint32_t i = 0; i < num_dwords; i++) {
    preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_TOUCH,
                            (uint8_t)addr_space, base_addr + i * WORD_SIZE,
                            WORD_SIZE, 0);
  }
}

static __attribute__((always_inline)) inline void trace_wr_as_u64(
    RvState* restrict state, uint64_t addr, uint64_t new_val,
    uint32_t addr_space) {
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE,
                          (uint8_t)addr_space, preflight_block_addr(addr),
                          WORD_SIZE, new_val);
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
