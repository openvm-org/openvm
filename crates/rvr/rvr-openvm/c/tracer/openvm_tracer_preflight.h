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

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "openvm_state.h"

/* Store for the write-once compact-record stream. Plain stores by default:
 * per-dword MOVNTI at the 44-byte record stride, interleaved with the
 * tracer's regular stores, leaves write-combining lines partially filled and
 * measured 13x SLOWER end to end on Zen (partial WC-line flushes) — define
 * OPENVM_RVR_NT_RECORDS to re-enable non-temporal stores only together with
 * full-cache-line batching. The host issues an sfence at the
 * execution/harvest boundary either way, before the buffers are read
 * (possibly from other threads). Output is byte-identical in both modes. */
static __attribute__((always_inline)) inline void nt_store_u32(
    uint32_t* restrict dst, uint32_t value) {
#if defined(__x86_64__) && defined(OPENVM_RVR_NT_RECORDS)
  _mm_stream_si32((int*)dst, (int)value);
#else
  *dst = value;
#endif
}

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

/* R3/R4: one per-chip inline-record buffer descriptor. `base` points at a
 * host-provided byte buffer for one chip; `len` is the byte cursor, advancing
 * by `stride` per record; `cap` the byte capacity (a multiple of `stride`).
 * Record i therefore sits at `base + i*stride`. Compact-wire buffers set
 * `stride` to the packed record size; arena-native buffers set it to the
 * arena row/record pitch (base is then 32-aligned by host contract so a
 * zero-copy DenseRecordArena adopt cannot slice a shifted range). A null
 * `base` means the chip is not migrated and uses the verbose memory log. */
typedef struct ChipRecordBuf {
  uint8_t* base;
  uint32_t len;
  uint32_t cap;
  uint32_t stride;
  /* R4 arena-native: byte offset of the core record within each record slot
   * (adapter fields sit at offset 0). Zero in compact-wire mode. */
  uint32_t core_off;
  uint32_t flags;
} ChipRecordBuf;

static constexpr uint32_t PREFLIGHT_RECORD_DIRECT_FINAL = 1u;

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
  /* R3: array of `chip_counts_len` per-chip inline-record buffers, indexed by
   * chip (AIR) index. Appended last so R1 field offsets are unchanged. */
  ChipRecordBuf* chip_records;
  /* ZG2: compile-time-indexed execution frequencies. Direct-final inline
   * records increment this array without emitting a duplicate program log. */
  uint32_t* exec_frequencies;
  uint32_t exec_frequencies_len;
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
_Static_assert(offsetof(Tracer, chip_records) == 96,
               "Tracer chip_records offset drift");
_Static_assert(offsetof(Tracer, exec_frequencies) == 104,
               "Tracer exec_frequencies offset drift");
_Static_assert(offsetof(Tracer, exec_frequencies_len) == 112,
               "Tracer exec_frequencies_len offset drift");
_Static_assert(sizeof(ChipRecordBuf) == PREFLIGHT_CHIP_RECORD_BUF_SIZE,
               "ChipRecordBuf size drift");
_Static_assert(_Alignof(ChipRecordBuf) == PREFLIGHT_CHIP_RECORD_BUF_ALIGN,
               "ChipRecordBuf align drift");
_Static_assert(offsetof(ChipRecordBuf, base) == 0,
               "ChipRecordBuf base offset drift");
_Static_assert(offsetof(ChipRecordBuf, len) == 8,
               "ChipRecordBuf len offset drift");
_Static_assert(offsetof(ChipRecordBuf, cap) == 12,
               "ChipRecordBuf cap offset drift");
_Static_assert(offsetof(ChipRecordBuf, stride) == 16,
               "ChipRecordBuf stride offset drift");
_Static_assert(offsetof(ChipRecordBuf, core_off) == 20,
               "ChipRecordBuf core_off offset drift");
_Static_assert(offsetof(ChipRecordBuf, flags) == 24,
               "ChipRecordBuf flags offset drift");

/* R3 (L1+L5 compact): base-ALU AddSub record holding the dynamic witness
 * only. Program-redundant operands (rd_ptr/rs1_ptr/rs2/rs2_as/rs2_imm_sign/
 * local_opcode) are re-derived from the instruction at `from_pc` during host
 * record assembly, and the layout has no padding. Field names/offsets mirror
 * the riscv circuit's compact-record mirror (its rvr record-ABI guard). */
typedef struct PreflightAddSubRecord {
  uint32_t from_pc;                   /* 0  */
  uint32_t from_timestamp;            /* 4  */
  uint32_t reads_aux[2];              /* 8: [prev_timestamp; 2] */
  uint32_t writes_aux_prev_timestamp; /* 16 */
  uint16_t writes_aux_prev_data[4];   /* 20 */
  uint16_t b[4];                      /* 28 */
  uint16_t c[4];                      /* 36 */
} PreflightAddSubRecord;

_Static_assert(sizeof(PreflightAddSubRecord) == PREFLIGHT_ADDSUB_RECORD_SIZE,
               "PreflightAddSubRecord size drift");
_Static_assert(_Alignof(PreflightAddSubRecord) == 4,
               "PreflightAddSubRecord align drift");
_Static_assert(offsetof(PreflightAddSubRecord, from_pc) == 0,
               "PreflightAddSubRecord from_pc offset drift");
_Static_assert(offsetof(PreflightAddSubRecord, from_timestamp) == 4,
               "PreflightAddSubRecord from_timestamp offset drift");
_Static_assert(offsetof(PreflightAddSubRecord, reads_aux) == 8,
               "PreflightAddSubRecord reads_aux offset drift");
_Static_assert(offsetof(PreflightAddSubRecord, writes_aux_prev_timestamp) == 16,
               "PreflightAddSubRecord writes_aux_prev_timestamp offset drift");
_Static_assert(offsetof(PreflightAddSubRecord, writes_aux_prev_data) == 20,
               "PreflightAddSubRecord writes_aux_prev_data offset drift");
_Static_assert(offsetof(PreflightAddSubRecord, b) == 28,
               "PreflightAddSubRecord b offset drift");
_Static_assert(offsetof(PreflightAddSubRecord, c) == 36,
               "PreflightAddSubRecord c offset drift");

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

/* Advance the timestamp, update the block's shadow entry, and record the block
 * on first touch this segment. Returns the block's previous-access timestamp
 * (0 = first touch) and writes the consumed timestamp to `*out_timestamp`.
 *
 * Every traced access must call this — whether it goes on to append a verbose
 * `MemoryLogEntry` (non-migrated opcodes) or to fill an inline compact record
 * (migrated RV64IM opcodes). Keeping the shadow/touched updates common to both
 * is what makes cross-instruction `prev_timestamp` chains and `touched_memory`
 * finalization byte-identical across a mixed-mode segment. */
static __attribute__((always_inline)) inline uint32_t preflight_touch(
    Tracer* restrict t, uint8_t addr_space, uint64_t address,
    uint32_t* restrict out_timestamp) {
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

  *out_timestamp = timestamp;
  return prev_timestamp;
}

/* Append one self-contained memory event. `address` need not be block-aligned;
 * the shadow index and touched-block key are derived from the aligned block.
 * `prev_value` is the block's value before this access (only consumed for
 * writes) and is supplied by the caller, which holds the store pointer. */
static __attribute__((always_inline)) inline uint32_t preflight_append_memory(
    Tracer* restrict t, uint8_t kind, uint8_t addr_space, uint64_t address,
    uint8_t width, uint64_t value, uint64_t prev_value) {
  uint32_t timestamp;
  uint32_t prev_timestamp = preflight_touch(t, addr_space, address, &timestamp);

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
  return prev_timestamp;
}

static __attribute__((always_inline)) inline void trace_timestamp(
    RvState* restrict state) {
  state->tracer->timestamp++;
}

/* ── Trace-only register access ──────────────────────────────────── */

static __attribute__((always_inline)) inline uint32_t trace_reg_read(
    RvState* restrict state, uint8_t idx, uint32_t val) {
  uint64_t reg_value = idx == 0 ? 0 : state->regs[idx];
  return preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ,
                                 AS_REGISTER, (uint32_t)idx * WORD_SIZE,
                                 WORD_SIZE, reg_value, reg_value);
}
/* Traced BEFORE the register store (see `write_reg` codegen), so
 * `state->regs[idx]` still holds the previous value for `prev_value`; the new
 * value arrives as `new_val`. Returns the register block's `prev_timestamp`
 * (consumed by inline record emission). */
static __attribute__((always_inline)) inline uint32_t trace_reg_write(
    RvState* restrict state, uint8_t idx, uint32_t new_val) {
  return preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE,
                                 AS_REGISTER, (uint32_t)idx * WORD_SIZE,
                                 WORD_SIZE, new_val, state->regs[idx]);
}

/* Touch-only main-memory block access for migrated load/store opcodes (R3):
 * identical timestamp/shadow/touched bookkeeping to the logging trace_rd/wr
 * helpers, no MemoryLogEntry. `block_addr` must be block-aligned. Returns the
 * block's previous-access timestamp. */
static __attribute__((always_inline)) inline uint32_t trace_mem_touch(
    RvState* restrict state, uint64_t block_addr) {
  uint32_t timestamp;
  return preflight_touch(state->tracer, AS_MEMORY, block_addr, &timestamp);
}

/* Touch-only public-values block access for inline REVEAL records. Mirrors
 * the verbose trace_wr_as_u64 timestamp/shadow/touched bookkeeping without
 * appending a MemoryLogEntry. */
static __attribute__((always_inline)) inline uint32_t trace_pv_touch(
    RvState* restrict state, uint64_t block_addr) {
  uint32_t timestamp;
  return preflight_touch(state->tracer, AS_PUBLIC_VALUES, block_addr,
                         &timestamp);
}

/* Touch-only register access for opcodes migrated to inline compact records
 * (R3): advances the timestamp and updates the shadow/touched bookkeeping
 * exactly like the logging `trace_reg_*` helpers, but appends no
 * `MemoryLogEntry` — the compact record carries the aux data instead. The tick
 * model must stay byte-identical to the logging variants. Returns the register
 * block's previous-access timestamp. */
static __attribute__((always_inline)) inline uint32_t trace_reg_touch(
    RvState* restrict state, uint8_t idx) {
  uint32_t timestamp;
  return preflight_touch(state->tracer, AS_REGISTER, (uint32_t)idx * WORD_SIZE,
                         &timestamp);
}

/* R3: emit one compact base-ALU AddSub record into chip `chip_idx`'s inline
 * buffer: the dynamic witness the caller already holds — the two operand
 * values (`rs1_val`/`rs2_val`, split into the `b`/`c` u16 limbs), the three
 * access `prev_timestamp`s from the touches, and the old `rd` block value
 * (`writes_aux` prev_data). A null buffer array / null buffer / out-of-range
 * chip / full buffer is a no-op: the chip then has a missing inline record,
 * which the host record assembly detects as a byte-count mismatch and rejects
 * loudly rather than silently corrupting. */
static __attribute__((always_inline)) inline void preflight_emit_alu3(
    RvState* restrict state, uint32_t chip_idx, uint32_t from_pc,
    uint32_t from_timestamp, uint32_t rs1_prev_ts, uint32_t rs2_prev_ts,
    uint32_t rd_prev_ts, uint64_t rd_prev_value, uint64_t rs1_val,
    uint64_t rs2_val) {
  Tracer* restrict t = state->tracer;
  if (unlikely(t->chip_records == NULL || chip_idx >= t->chip_counts_len)) {
    return;
  }
  ChipRecordBuf* restrict buf = &t->chip_records[chip_idx];
  if (buf->base == NULL) {
    return;
  }
  uint32_t off = buf->len;
  if (unlikely(off + buf->stride > buf->cap)) {
    return;
  }
  buf->len = off + buf->stride;
  /* The 44-byte record is exactly 11 u32 words: the u16 limb arrays are the
   * low/high halves of the u64 values they were split from. */
  uint32_t* restrict words = (uint32_t*)(buf->base + off);
  nt_store_u32(&words[0], from_pc);
  nt_store_u32(&words[1], from_timestamp);
  nt_store_u32(&words[2], rs1_prev_ts);
  nt_store_u32(&words[3], rs2_prev_ts);
  nt_store_u32(&words[4], rd_prev_ts);
  nt_store_u32(&words[5], (uint32_t)rd_prev_value);
  nt_store_u32(&words[6], (uint32_t)(rd_prev_value >> 32));
  nt_store_u32(&words[7], (uint32_t)rs1_val);
  nt_store_u32(&words[8], (uint32_t)(rs1_val >> 32));
  nt_store_u32(&words[9], (uint32_t)rs2_val);
  nt_store_u32(&words[10], (uint32_t)(rs2_val >> 32));
}

/* R4: store a u64 value at `dst` as two u32 stores. U16-cell blocks hold
 * the value's little-endian [u16;4] limbs, which IS the raw u64 bytes, so
 * this matches the host's u16x4 reinterpret exactly. Two u32 stores (not
 * one u64) because Matrix rows are only 4-byte aligned. */
static __attribute__((always_inline)) inline void arena_store_u64_le(
    uint8_t* restrict dst, uint64_t v) {
  *(uint32_t*)dst = (uint32_t)v;
  *(uint32_t*)(dst + 4) = (uint32_t)(v >> 32);
}

/* R3/R4: claim one record slot (`stride` bytes) from chip `chip_idx`'s
 * inline record buffer, or NULL when the chip is unmigrated / the buffer is
 * full (the host record assembly detects the resulting byte-count mismatch
 * and rejects loudly). */
static __attribute__((always_inline)) inline uint32_t* preflight_claim_record(
    RvState* restrict state, uint32_t chip_idx) {
  Tracer* restrict t = state->tracer;
  if (unlikely(t->chip_records == NULL || chip_idx >= t->chip_counts_len)) {
    return NULL;
  }
  ChipRecordBuf* restrict buf = &t->chip_records[chip_idx];
  if (buf->base == NULL) {
    return NULL;
  }
  uint32_t off = buf->len;
  if (unlikely(off + buf->stride > buf->cap)) {
    return NULL;
  }
  buf->len = off + buf->stride;
  return (uint32_t*)(buf->base + off);
}

/* R3: compact branch record (2 reads, no write); see PreflightBranch2 layout
 * on the host side. */
static __attribute__((always_inline)) inline void preflight_emit_branch2(
    RvState* restrict state, uint32_t chip_idx, uint32_t from_pc,
    uint32_t from_timestamp, uint32_t rs1_prev_ts, uint32_t rs2_prev_ts,
    uint64_t rs1_val, uint64_t rs2_val) {
  uint32_t* restrict words =
      preflight_claim_record(state, chip_idx);
  if (unlikely(words == NULL)) {
    return;
  }
  nt_store_u32(&words[0], from_pc);
  nt_store_u32(&words[1], from_timestamp);
  nt_store_u32(&words[2], rs1_prev_ts);
  nt_store_u32(&words[3], rs2_prev_ts);
  nt_store_u32(&words[4], (uint32_t)rs1_val);
  nt_store_u32(&words[5], (uint32_t)(rs1_val >> 32));
  nt_store_u32(&words[6], (uint32_t)rs2_val);
  nt_store_u32(&words[7], (uint32_t)(rs2_val >> 32));
}

/* R3: compact write-only record (JalLui/Auipc). For a suppressed write
 * (rd = x0) the caller passes zeros; the host decides from the instruction's
 * enable flag. */
static __attribute__((always_inline)) inline void preflight_emit_wr1(
    RvState* restrict state, uint32_t chip_idx, uint32_t from_pc,
    uint32_t from_timestamp, uint32_t rd_prev_ts, uint64_t rd_prev_value) {
  uint32_t* restrict words =
      preflight_claim_record(state, chip_idx);
  if (unlikely(words == NULL)) {
    return;
  }
  nt_store_u32(&words[0], from_pc);
  nt_store_u32(&words[1], from_timestamp);
  nt_store_u32(&words[2], rd_prev_ts);
  nt_store_u32(&words[3], (uint32_t)rd_prev_value);
  nt_store_u32(&words[4], (uint32_t)(rd_prev_value >> 32));
}

/* R3: compact read + conditional-write record (Jalr). */
static __attribute__((always_inline)) inline void preflight_emit_rw1(
    RvState* restrict state, uint32_t chip_idx, uint32_t from_pc,
    uint32_t from_timestamp, uint32_t rs1_prev_ts, uint32_t rd_prev_ts,
    uint64_t rs1_val, uint64_t rd_prev_value) {
  uint32_t* restrict words =
      preflight_claim_record(state, chip_idx);
  if (unlikely(words == NULL)) {
    return;
  }
  nt_store_u32(&words[0], from_pc);
  nt_store_u32(&words[1], from_timestamp);
  nt_store_u32(&words[2], rs1_prev_ts);
  nt_store_u32(&words[3], rd_prev_ts);
  nt_store_u32(&words[4], (uint32_t)rs1_val);
  nt_store_u32(&words[5], (uint32_t)(rs1_val >> 32));
  nt_store_u32(&words[6], (uint32_t)rd_prev_value);
  nt_store_u32(&words[7], (uint32_t)(rd_prev_value >> 32));
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
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t prev_block =
      addr_space == AS_PUBLIC_VALUES
          ? preflight_read_pv_block(state->tracer, block_addr)
          : 0u;
  uint64_t block =
      preflight_patch_mem_block(prev_block, addr, width, new_val);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE,
                          (uint8_t)addr_space, block_addr, WORD_SIZE, block,
                          prev_block);
}

static __attribute__((always_inline)) inline void trace_pc(
    RvState* restrict state, uint64_t pc) {
  preflight_append_program(state->tracer, pc);
}

/* ZG2 single-pass dispatch accounting. Every instruction increments its
 * compile-time filtered program index. An inline instruction whose chip
 * target is DIRECT_FINAL already carries its pc/timestamp in the final wire
 * stream, so the duplicate ProgramLogEntry is suppressed. Pooled/unstaged
 * compact records keep the log because host expansion still consumes it. */
static __attribute__((always_inline)) inline void trace_pc_indexed(
    RvState* restrict state, uint64_t pc, uint32_t exec_idx,
    uint32_t inline_chip_idx) {
  Tracer* restrict t = state->tracer;
  if (likely(t->exec_frequencies != NULL &&
             exec_idx < t->exec_frequencies_len)) {
    t->exec_frequencies[exec_idx]++;
  }
  bool direct_final =
      inline_chip_idx < t->chip_counts_len && t->chip_records != NULL &&
      (t->chip_records[inline_chip_idx].flags &
       PREFLIGHT_RECORD_DIRECT_FINAL) != 0u;
  if (!direct_final) {
    preflight_append_program(t, pc);
  }
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
