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
  uint32_t timestamp;
  /* OpenVM pcs are 4-byte aligned; bit 0 is the writer-complete guard. */
  uint32_t pc_and_flags;
  uint64_t write_value;
} ProgramLogEntry;

static constexpr uint32_t PREFLIGHT_PROGRAM_WRITE_COMPLETE = 1u;

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

/* Delta-only residual-memory wire. The chronological decoder reconstructs
 * prev_timestamp and prev_value. CPU, compact, partial-direct, and non-delta
 * routes continue to use the full MemoryLogEntry above. */
typedef struct DeltaMemoryLogEntry {
  uint32_t timestamp;
  uint32_t address;
  uint64_t value;
  uint8_t kind;
  uint8_t addr_space;
  uint8_t width;
  uint8_t complete;
  uint32_t _reserved;
} DeltaMemoryLogEntry;

/* A block touched (for the first time) this segment. `block_addr` is the
 * block-aligned byte address; the host derives the AS-native block pointer and
 * reads the final value from live memory + the final timestamp from the
 * shadow. */
typedef struct TouchedBlock {
  uint32_t addr_space;
  uint32_t block_addr;
  uint64_t initial_value;
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
static constexpr uint32_t PREFLIGHT_RECORD_OVERFLOW = 2u;
static constexpr uint32_t PREFLIGHT_RECORD_RESIDUAL_MEMORY_CHRONOLOGY = 4u;
static constexpr uint32_t PREFLIGHT_RECORD_VARIABLE_ROWS = 8u;
static constexpr uint32_t PREFLIGHT_RECORD_VARIABLE_ROW_STRIDE = 16u;
static constexpr uint32_t PREFLIGHT_RECORD_COMPACT_RESIDUAL_MEMORY = 32u;

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
  /* Stage-2 chronological delta stream. Unlike chip_records this is a single
   * cross-AIR stream, so record order makes prev-timestamps reconstructible
   * on the device. */
  ChipRecordBuf* delta_records;
  /* Instruction-local full predecessor capture for custom direct-final
   * emitters when memory_log points at compact 24-byte entries. UINT32_MAX
   * means inactive. The Rust caller owns the bounded scratch backing. */
  MemoryLogEntry* custom_memory_scratch;
  uint32_t custom_memory_scratch_len;
  uint32_t custom_memory_scratch_cap;
} Tracer;

_Static_assert(sizeof(ProgramLogEntry) == PREFLIGHT_PROGRAM_LOG_ENTRY_SIZE,
               "ProgramLogEntry size drift");
_Static_assert(_Alignof(ProgramLogEntry) == PREFLIGHT_PROGRAM_LOG_ENTRY_ALIGN,
               "ProgramLogEntry align drift");
_Static_assert(offsetof(ProgramLogEntry, timestamp) == 0,
               "ProgramLogEntry timestamp offset drift");
_Static_assert(offsetof(ProgramLogEntry, pc_and_flags) == 4,
               "ProgramLogEntry pc_and_flags offset drift");
_Static_assert(offsetof(ProgramLogEntry, write_value) == 8,
               "ProgramLogEntry write_value offset drift");
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
_Static_assert(sizeof(DeltaMemoryLogEntry) == PREFLIGHT_DELTA_MEMORY_LOG_ENTRY_SIZE,
               "DeltaMemoryLogEntry size drift");
_Static_assert(_Alignof(DeltaMemoryLogEntry) == PREFLIGHT_DELTA_MEMORY_LOG_ENTRY_ALIGN,
               "DeltaMemoryLogEntry align drift");
_Static_assert(offsetof(DeltaMemoryLogEntry, timestamp) == 0,
               "DeltaMemoryLogEntry timestamp offset drift");
_Static_assert(offsetof(DeltaMemoryLogEntry, address) == 4,
               "DeltaMemoryLogEntry address offset drift");
_Static_assert(offsetof(DeltaMemoryLogEntry, value) == 8,
               "DeltaMemoryLogEntry value offset drift");
_Static_assert(offsetof(DeltaMemoryLogEntry, kind) == 16,
               "DeltaMemoryLogEntry kind offset drift");
_Static_assert(offsetof(DeltaMemoryLogEntry, addr_space) == 17,
               "DeltaMemoryLogEntry addr_space offset drift");
_Static_assert(offsetof(DeltaMemoryLogEntry, width) == 18,
               "DeltaMemoryLogEntry width offset drift");
_Static_assert(offsetof(DeltaMemoryLogEntry, complete) == 19,
               "DeltaMemoryLogEntry complete offset drift");
_Static_assert(offsetof(DeltaMemoryLogEntry, _reserved) == 20,
               "DeltaMemoryLogEntry reserved offset drift");
_Static_assert(sizeof(TouchedBlock) == PREFLIGHT_TOUCHED_BLOCK_SIZE,
               "TouchedBlock size drift");
_Static_assert(_Alignof(TouchedBlock) == PREFLIGHT_TOUCHED_BLOCK_ALIGN,
               "TouchedBlock align drift");
_Static_assert(offsetof(TouchedBlock, addr_space) == 0,
               "TouchedBlock addr_space offset drift");
_Static_assert(offsetof(TouchedBlock, block_addr) == 4,
               "TouchedBlock block_addr offset drift");
_Static_assert(offsetof(TouchedBlock, initial_value) == 8,
               "TouchedBlock initial_value offset drift");
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
_Static_assert(offsetof(Tracer, delta_records) == 120,
               "Tracer delta_records offset drift");
_Static_assert(offsetof(Tracer, custom_memory_scratch) == 128,
               "Tracer custom_memory_scratch offset drift");
_Static_assert(offsetof(Tracer, custom_memory_scratch_len) == 136,
               "Tracer custom_memory_scratch_len offset drift");
_Static_assert(offsetof(Tracer, custom_memory_scratch_cap) == 140,
               "Tracer custom_memory_scratch_cap offset drift");
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

/* Stage-2 global chronological record. The old wire's three previous-access
 * timestamps and post-write value are intentionally absent: a decoder merges
 * this stream with the residual memory log, reconstructs access chronology,
 * and derives the write from the opcode and the two source values. */
typedef struct PreflightDeltaRecord {
  uint32_t from_pc;
  uint32_t from_timestamp;
  uint64_t v1;
  uint64_t v2;
} PreflightDeltaRecord;

_Static_assert(sizeof(PreflightDeltaRecord) == PREFLIGHT_DELTA_RECORD_SIZE,
               "PreflightDeltaRecord size drift");
_Static_assert(_Alignof(PreflightDeltaRecord) == 8,
               "PreflightDeltaRecord align drift");

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
  if (unlikely(pc > UINT32_MAX || (pc & 3u) != 0u)) {
    if (t->delta_records != NULL) {
      t->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
    }
    return;
  }
  uint32_t idx = t->program_log_len++;
  if (likely(idx < t->program_log_cap)) {
    ProgramLogEntry entry = {
        .timestamp = t->timestamp,
        .pc_and_flags = (uint32_t)pc,
        .write_value = 0,
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

static __attribute__((always_inline)) inline uint64_t
preflight_read_block_for_seed(RvState* restrict state, uint8_t addr_space,
                              uint64_t block_addr) {
  if (addr_space == AS_REGISTER) {
    uint32_t reg = (uint32_t)(block_addr / WORD_SIZE);
    return reg == 0u ? 0u : state->regs[reg];
  }
  if (addr_space == AS_PUBLIC_VALUES) {
    return preflight_read_pv_block(state->tracer, block_addr);
  }
  return preflight_read_mem_block(state, block_addr);
}

static __attribute__((always_inline)) inline void preflight_append_touched(
    Tracer* restrict t, uint8_t addr_space, uint64_t block_addr,
    uint64_t initial_value) {
  uint32_t ti = t->touched_len++;
  if (likely(ti < t->touched_cap)) {
    t->touched[ti].addr_space = addr_space;
    t->touched[ti].block_addr = (uint32_t)block_addr;
    t->touched[ti].initial_value = initial_value;
  }
}

static __attribute__((always_inline)) inline uint64_t preflight_patch_mem_block(
    uint64_t block, uint64_t addr, uint8_t width, uint64_t value) {
  uint32_t shift = (addr & (WORD_SIZE - 1u)) * 8u;
  uint64_t mask = width == WORD_SIZE ? UINT64_MAX
                                     : ((1ull << (width * 8u)) - 1ull);
  return (block & ~(mask << shift)) | ((value & mask) << shift);
}

static __attribute__((always_inline)) inline bool
preflight_compact_residual_memory(Tracer* restrict t) {
  return t->delta_records != NULL &&
         (t->delta_records->flags &
          PREFLIGHT_RECORD_COMPACT_RESIDUAL_MEMORY) != 0u;
}

/* Custom VecHeap emitters run after their traced accesses, so compact residual
 * entries no longer contain the predecessor fields they need. Capture only
 * that instruction's events in bounded scratch; ordinary instructions pay a
 * single predictable inactive check and write no side log. */
static __attribute__((always_inline)) inline void
preflight_begin_custom_memory_capture(RvState* restrict state) {
  Tracer* restrict t = state->tracer;
  if (preflight_compact_residual_memory(t)) {
    if (unlikely(t->custom_memory_scratch == NULL ||
                 t->custom_memory_scratch_cap == 0u)) {
      t->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
      t->custom_memory_scratch_len = UINT32_MAX;
      return;
    }
    t->custom_memory_scratch_len = 0u;
  }
}

static __attribute__((always_inline)) inline MemoryLogEntry*
preflight_take_custom_memory_events(Tracer* restrict t, uint32_t event_count) {
  if (!preflight_compact_residual_memory(t)) {
    if (unlikely(t->memory_log_len < event_count ||
                 t->memory_log_len > t->memory_log_cap)) {
      return NULL;
    }
    return t->memory_log + t->memory_log_len - event_count;
  }
  uint32_t captured = t->custom_memory_scratch_len;
  t->custom_memory_scratch_len = UINT32_MAX;
  if (unlikely(captured != event_count ||
               event_count > t->custom_memory_scratch_cap)) {
    t->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
    return NULL;
  }
  return t->custom_memory_scratch;
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
    uint64_t initial_value, uint32_t* restrict out_timestamp) {
  uint32_t timestamp = t->timestamp++;
  uint64_t block_addr = address & ~(uint64_t)(WORD_SIZE - 1u);
  uint32_t block_idx = (uint32_t)(block_addr / WORD_SIZE);

  uint32_t* restrict shadow = preflight_shadow_for(t, addr_space);
  uint32_t prev_timestamp = shadow[block_idx];
  shadow[block_idx] = timestamp;
  if (prev_timestamp == 0u) {
    preflight_append_touched(t, addr_space, block_addr, initial_value);
  }

  *out_timestamp = timestamp;
  return prev_timestamp;
}

/* Touch-only custom-family access. Resolve the first-touch seed after the
 * shadow transition so the common path performs one shadow load, not the
 * prior probe followed by a second lookup in `preflight_touch`. */
static __attribute__((always_inline)) inline uint32_t
preflight_touch_seed_from_state(RvState* restrict state, uint8_t addr_space,
                                uint64_t address,
                                uint64_t* restrict out_initial_value,
                                uint32_t* restrict out_timestamp) {
  Tracer* restrict t = state->tracer;
  uint32_t timestamp = t->timestamp++;
  uint64_t block_addr = preflight_block_addr(address);
  uint32_t block_idx = (uint32_t)(block_addr / WORD_SIZE);

  uint32_t* restrict shadow = preflight_shadow_for(t, addr_space);
  uint32_t prev_timestamp = shadow[block_idx];
  uint64_t initial_value = 0u;
  if (prev_timestamp == 0u) {
    initial_value =
        preflight_read_block_for_seed(state, addr_space, block_addr);
  }
  shadow[block_idx] = timestamp;
  if (prev_timestamp == 0u) {
    preflight_append_touched(t, addr_space, block_addr, initial_value);
  }

  *out_initial_value = initial_value;
  *out_timestamp = timestamp;
  return prev_timestamp;
}

static __attribute__((always_inline)) inline void
preflight_append_memory_record(Tracer* restrict t, uint8_t kind,
                               uint8_t addr_space, uint64_t address,
                               uint8_t width, uint64_t value,
                               uint64_t prev_value, uint32_t timestamp,
                               uint32_t prev_timestamp,
                               bool compact_residual) {
  uint32_t idx = t->memory_log_len++;
  if (likely(idx < t->memory_log_cap)) {
    if (unlikely(t->custom_memory_scratch_len != UINT32_MAX)) {
      uint32_t scratch_idx = t->custom_memory_scratch_len++;
      if (likely(scratch_idx < t->custom_memory_scratch_cap)) {
        MemoryLogEntry scratch_entry = {
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
        t->custom_memory_scratch[scratch_idx] = scratch_entry;
      } else {
        t->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
      }
    }
    if (compact_residual) {
      DeltaMemoryLogEntry entry = {
          .timestamp = timestamp,
          .address = (uint32_t)address,
          .value = value,
          .kind = kind,
          .addr_space = addr_space,
          .width = width,
          .complete = 1,
          ._reserved = 0,
      };
      ((DeltaMemoryLogEntry*)t->memory_log)[idx] = entry;
      return;
    }
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

/* Append one self-contained memory event. `address` need not be block-aligned;
 * the shadow index and touched-block key are derived from the aligned block.
 * `prev_value` is the block's value before this access (only consumed for
 * writes) and is supplied by the caller, which holds the store pointer. */
static __attribute__((always_inline)) inline uint32_t preflight_append_memory(
    Tracer* restrict t, uint8_t kind, uint8_t addr_space, uint64_t address,
    uint8_t width, uint64_t value, uint64_t prev_value) {
  bool compact_residual = preflight_compact_residual_memory(t);
  if (unlikely(compact_residual && address > UINT32_MAX)) {
    t->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
    return 0;
  }
  uint32_t timestamp;
  uint32_t prev_timestamp =
      preflight_touch(t, addr_space, address, prev_value, &timestamp);
  preflight_append_memory_record(t, kind, addr_space, address, width, value,
                                 prev_value, timestamp, prev_timestamp,
                                 compact_residual);
  return prev_timestamp;
}

static __attribute__((always_inline)) inline void trace_timestamp(
    RvState* restrict state) {
  state->tracer->timestamp++;
}

/* ── Trace-only register access ──────────────────────────────────── */

static __attribute__((always_inline)) inline uint32_t trace_reg_read(
    RvState* restrict state, uint8_t idx, uint64_t val) {
  uint64_t reg_value = idx == 0 ? 0 : val;
  return preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ,
                                 AS_REGISTER, (uint32_t)idx * WORD_SIZE,
                                 WORD_SIZE, reg_value, reg_value);
}
/* Traced BEFORE the register store (see `write_reg` codegen), so
 * `state->regs[idx]` still holds the previous value for `prev_value`; the new
 * value arrives as `new_val`. Returns the register block's `prev_timestamp`
 * (consumed by inline record emission). */
static __attribute__((always_inline)) inline uint32_t trace_reg_write(
    RvState* restrict state, uint8_t idx, uint64_t new_val) {
  return preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE,
                                 AS_REGISTER, (uint32_t)idx * WORD_SIZE,
                                 WORD_SIZE, new_val, state->regs[idx]);
}

/* Touch-only main-memory block access for migrated load/store opcodes (R3):
 * identical timestamp/shadow/touched bookkeeping to the logging trace_rd/wr
 * helpers, no MemoryLogEntry. `block_addr` must be block-aligned. Returns the
 * block's previous-access timestamp. */
static __attribute__((always_inline)) inline uint32_t trace_mem_touch(
    RvState* restrict state, uint64_t block_addr, uint64_t block_value) {
  Tracer* restrict t = state->tracer;
  uint32_t timestamp = t->timestamp++;
  uint32_t block_idx = (uint32_t)(block_addr / WORD_SIZE);
  uint32_t prev_timestamp = t->shadow_memory[block_idx];
  t->shadow_memory[block_idx] = timestamp;
  if (prev_timestamp == 0u) {
    preflight_append_touched(t, AS_MEMORY, block_addr, block_value);
  }
  return prev_timestamp;
}

/* Store-side variant: only read the pre-write block when it is the seed for
 * this segment. Subsequent stores reconstruct their predecessor on device. */
static __attribute__((always_inline)) inline uint32_t trace_mem_store_touch(
    RvState* restrict state, uint64_t block_addr) {
  Tracer* restrict t = state->tracer;
  uint32_t timestamp = t->timestamp++;
  uint32_t block_idx = (uint32_t)(block_addr / WORD_SIZE);
  uint32_t prev_timestamp = t->shadow_memory[block_idx];
  uint64_t initial_value = 0u;
  if (prev_timestamp == 0u) {
    initial_value = rd_mem_u64(state->memory, block_addr);
  }
  t->shadow_memory[block_idx] = timestamp;
  if (prev_timestamp == 0u) {
    preflight_append_touched(t, AS_MEMORY, block_addr, initial_value);
  }
  return prev_timestamp;
}

/* Touch-only public-values block access for inline REVEAL records. Mirrors
 * the verbose trace_wr_as_u64 timestamp/shadow/touched bookkeeping without
 * appending a MemoryLogEntry. */
static __attribute__((always_inline)) inline uint32_t trace_pv_touch(
    RvState* restrict state, uint64_t block_addr) {
  Tracer* restrict t = state->tracer;
  uint32_t timestamp = t->timestamp++;
  uint32_t block_idx = (uint32_t)(block_addr / WORD_SIZE);
  uint32_t prev_timestamp = t->shadow_public_values[block_idx];
  uint64_t initial_value = 0u;
  if (prev_timestamp == 0u) {
    initial_value = preflight_read_pv_block(t, block_addr);
  }
  t->shadow_public_values[block_idx] = timestamp;
  if (prev_timestamp == 0u) {
    preflight_append_touched(t, AS_PUBLIC_VALUES, block_addr, initial_value);
  }
  return prev_timestamp;
}

/* Touch-only register access for opcodes migrated to inline compact records
 * (R3): advances the timestamp and updates the shadow/touched bookkeeping
 * exactly like the logging `trace_reg_*` helpers, but appends no
 * `MemoryLogEntry` — the compact record carries the aux data instead. The tick
 * model must stay byte-identical to the logging variants. Returns the register
 * block's previous-access timestamp. */
static __attribute__((always_inline)) inline uint32_t trace_reg_touch(
    RvState* restrict state, uint8_t idx) {
  Tracer* restrict t = state->tracer;
  uint32_t timestamp = t->timestamp++;
  uint32_t prev_timestamp = t->shadow_register[idx];
  uint64_t initial_value = 0u;
  if (prev_timestamp == 0u) {
    initial_value = idx == 0 ? 0u : state->regs[idx];
  }
  t->shadow_register[idx] = timestamp;
  if (prev_timestamp == 0u) {
    preflight_append_touched(t, AS_REGISTER, (uint32_t)idx * WORD_SIZE,
                             initial_value);
  }
  return prev_timestamp;
}

static __attribute__((always_inline)) inline void
preflight_set_last_program_write_value(Tracer* restrict t, uint64_t value) {
  if (likely(t->program_log_len != 0u &&
             t->program_log_len <= t->program_log_cap)) {
    ProgramLogEntry* restrict entry = &t->program_log[t->program_log_len - 1u];
    entry->write_value = value;
    entry->pc_and_flags |= PREFLIGHT_PROGRAM_WRITE_COMPLETE;
  }
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

/* Fixed PhantomRecord ABI: pc, three instruction operands, timestamp. The
 * instruction has no memory-bus accesses; the caller performs its one bare
 * timestamp tick after writing the record. */
static __attribute__((always_inline)) inline void preflight_emit_phantom(
    RvState* restrict state, uint32_t chip_idx, uint32_t pc,
    uint32_t timestamp, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t* restrict words = preflight_claim_record(state, chip_idx);
  if (unlikely(words == NULL)) {
    return;
  }
  nt_store_u32(&words[0], pc);
  nt_store_u32(&words[1], a);
  nt_store_u32(&words[2], b);
  nt_store_u32(&words[3], c);
  nt_store_u32(&words[4], timestamp);
}

/* Claim one packed variable-row direct-final record. `len` remains the byte
 * cursor; `core_off`, unused by this layout, counts emitted trace rows so the
 * host can pin the exact unpadded height without parsing record bytes. */
static __attribute__((always_inline)) inline uint8_t*
preflight_claim_variable_record(RvState* restrict state, uint32_t chip_idx,
                                uint32_t record_bytes, uint32_t rows) {
  Tracer* restrict t = state->tracer;
  if (unlikely(t->chip_records == NULL || chip_idx >= t->chip_counts_len)) {
    return NULL;
  }
  ChipRecordBuf* restrict buf = &t->chip_records[chip_idx];
  if (unlikely(buf->base == NULL ||
               (buf->flags & PREFLIGHT_RECORD_VARIABLE_ROWS) == 0u)) {
    return NULL;
  }
  uint32_t off = buf->len;
  uint32_t claim_bytes = record_bytes;
  if ((buf->flags & PREFLIGHT_RECORD_VARIABLE_ROW_STRIDE) != 0u) {
    if (unlikely(rows > UINT32_MAX / buf->stride)) {
      buf->flags |= PREFLIGHT_RECORD_OVERFLOW;
      return NULL;
    }
    claim_bytes = rows * buf->stride;
    if (unlikely(record_bytes > claim_bytes)) {
      buf->flags |= PREFLIGHT_RECORD_OVERFLOW;
      return NULL;
    }
  }
  uint32_t next = off + claim_bytes;
  uint32_t next_rows = buf->core_off + rows;
  if (unlikely(next < off || next > buf->cap || next_rows < buf->core_off)) {
    buf->flags |= PREFLIGHT_RECORD_OVERFLOW;
    return NULL;
  }
  buf->len = next;
  buf->core_off = next_rows;
  return buf->base + off;
}

/* Reserve one basic block's delta records with a single cursor/capacity
 * update. Generated C indexes the returned span at compile-time constants,
 * removing per-instruction claiming and bounds checks from the hot path. */
static __attribute__((always_inline)) inline PreflightDeltaRecord*
preflight_claim_delta_records(RvState* restrict state, uint32_t count) {
  ChipRecordBuf* restrict buf = state->tracer->delta_records;
  if (unlikely(buf == NULL)) {
    return NULL;
  }
  if (unlikely(buf->base == NULL ||
               buf->stride != PREFLIGHT_DELTA_RECORD_SIZE)) {
    buf->flags |= PREFLIGHT_RECORD_OVERFLOW;
    return NULL;
  }
  if (unlikely(count > UINT32_MAX / PREFLIGHT_DELTA_RECORD_SIZE)) {
    buf->flags |= PREFLIGHT_RECORD_OVERFLOW;
    return NULL;
  }
  uint32_t off = buf->len;
  uint32_t bytes = count * PREFLIGHT_DELTA_RECORD_SIZE;
  if (unlikely(off + bytes < off || off + bytes > buf->cap)) {
    buf->flags |= PREFLIGHT_RECORD_OVERFLOW;
    return NULL;
  }
  buf->len = off + bytes;
  return (PreflightDeltaRecord*)(buf->base + off);
}

static __attribute__((always_inline)) inline void preflight_write_delta2(
    PreflightDeltaRecord* restrict record, uint32_t from_pc,
    uint32_t from_timestamp, uint64_t v1, uint64_t v2) {
  if (unlikely(record == NULL)) {
    return;
  }
  *record = (PreflightDeltaRecord){
      .from_pc = from_pc,
      .from_timestamp = from_timestamp,
      .v1 = v1,
      .v2 = v2,
  };
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
  uint64_t block_addr = preflight_block_addr(addr);
  bool compact_residual =
      preflight_compact_residual_memory(state->tracer);
  if (unlikely(compact_residual && block_addr > UINT32_MAX)) {
    state->tracer->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
    return;
  }
  uint32_t timestamp;
  uint64_t initial_value;
  uint32_t prev_timestamp = preflight_touch_seed_from_state(
      state, (uint8_t)addr_space, block_addr, &initial_value, &timestamp);
  preflight_append_memory_record(
      state->tracer, PREFLIGHT_MEMORY_KIND_TOUCH, (uint8_t)addr_space,
      block_addr, 0, 0, initial_value, timestamp, prev_timestamp,
      compact_residual);
}

static __attribute__((always_inline)) inline void trace_mem_access_u64_range(
    RvState* restrict state, uint64_t base_addr, uint32_t num_dwords,
    uint32_t addr_space) {
  bool compact_residual =
      preflight_compact_residual_memory(state->tracer);
  for (uint32_t i = 0; i < num_dwords; i++) {
    uint64_t block_addr = base_addr + i * WORD_SIZE;
    if (unlikely(compact_residual && block_addr > UINT32_MAX)) {
      state->tracer->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
      return;
    }
    uint32_t timestamp;
    uint64_t initial_value;
    uint32_t prev_timestamp = preflight_touch_seed_from_state(
        state, (uint8_t)addr_space, block_addr, &initial_value, &timestamp);
    preflight_append_memory_record(
        state->tracer, PREFLIGHT_MEMORY_KIND_TOUCH, (uint8_t)addr_space,
        block_addr, WORD_SIZE, 0, initial_value, timestamp, prev_timestamp,
        compact_residual);
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
    uint32_t inline_chip_idx, bool delta_inline) {
  Tracer* restrict t = state->tracer;
  if (likely(t->exec_frequencies != NULL &&
             exec_idx < t->exec_frequencies_len)) {
    t->exec_frequencies[exec_idx]++;
  }
  bool direct_final = delta_inline
      ? t->delta_records != NULL &&
            (t->delta_records->flags & PREFLIGHT_RECORD_DIRECT_FINAL) != 0u
      :
      inline_chip_idx < t->chip_counts_len && t->chip_records != NULL &&
      (t->chip_records[inline_chip_idx].flags &
       PREFLIGHT_RECORD_DIRECT_FINAL) != 0u;
  bool arena_direct_final =
      !delta_inline && inline_chip_idx < t->chip_counts_len &&
      t->chip_records != NULL &&
      (t->chip_records[inline_chip_idx].flags &
       PREFLIGHT_RECORD_DIRECT_FINAL) != 0u;
  /* In combined delta + arena-native mode, retain one lightweight chronology
   * entry for each arena-native instruction. Its final record is already in
   * the arena, but the delta decoder must replay its register touches before
   * reconstructing later compact records' previous timestamps. */
  bool residual_memory_chronology =
      arena_direct_final &&
      (t->chip_records[inline_chip_idx].flags &
       PREFLIGHT_RECORD_RESIDUAL_MEMORY_CHRONOLOGY) != 0u;
  bool needs_delta_chronology = t->delta_records != NULL && arena_direct_final &&
                                !residual_memory_chronology;
  if (!direct_final || needs_delta_chronology) {
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
