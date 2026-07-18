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

#ifndef OPENVM_TRACER_PREFLIGHT_COMMON_H
#define OPENVM_TRACER_PREFLIGHT_COMMON_H

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
  /* OpenVM pcs are 4-byte aligned; the low bits are side-band guards. */
  uint32_t pc_and_flags;
  uint64_t write_value;
} ProgramLogEntry;

/* L2: one compact descriptor per executed basic block. Device predecode
 * expands the consecutive pcs into exact program order and frequencies. */
typedef struct ProgramRunEntry {
  uint32_t first_pc;
  uint32_t instruction_count;
  uint32_t chronology_offset;
  uint32_t complete;
} ProgramRunEntry;

/* Oracle-only legacy per-instruction chronology. */
typedef struct DeviceProgramEntry {
  uint32_t pc;
  uint32_t filtered_index;
} DeviceProgramEntry;

/* G2 private wire cursors. Slot order is private producer ABI; Rust publishes
 * the sorted public descriptor table only after every cursor validates. */
typedef struct G2ProducerLaneV1 {
  uint64_t offset;
  uint32_t len;
  uint32_t cap;
  uint32_t expected_len;
  uint32_t reserved;
} G2ProducerLaneV1;

typedef struct G2ProducerV1 {
  uint8_t* base;
  uint64_t capacity;
  G2ProducerLaneV1* lanes;
  uint32_t lane_count;
  uint32_t instruction_count;
  uint32_t overflow;
  uint32_t reserved;
} G2ProducerV1;

static constexpr uint32_t G2_PRODUCER_RUN_SLOT = 0u;
static constexpr uint32_t G2_PRODUCER_RESIDUAL_CTRL_SLOT = 1u;
static constexpr uint32_t G2_PRODUCER_RESIDUAL_TAG_SLOT = 2u;
static constexpr uint32_t G2_PRODUCER_RESIDUAL_VALUE_SLOT = 3u;
static constexpr uint32_t G2_PRODUCER_ADDI_SLOT = 4u;
static constexpr uint32_t G2_PRODUCER_OPAQUE_EVENT_COUNT_SLOT = 58u;
static constexpr uint32_t G2_PRODUCER_LANE_COUNT = 59u;

static constexpr uint32_t PREFLIGHT_PROGRAM_WRITE_COMPLETE = 1u;
static constexpr uint32_t PREFLIGHT_PROGRAM_CROSSING_RESIDUAL = 2u;

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
  uint64_t initial_value;
} TouchedBlock;

typedef struct DeviceAuxReference {
  uint32_t prev_timestamp;
  uint32_t _reserved;
  uint64_t prev_value;
} DeviceAuxReference;

typedef struct DeviceAuxPatch {
  uint64_t target;
  uint32_t event_index;
  uint32_t kind;
  uint64_t expected;
} DeviceAuxPatch;

static constexpr uint32_t PREFLIGHT_DEVICE_AUX_PATCH_U32 = 0u;
static constexpr uint32_t PREFLIGHT_DEVICE_AUX_PATCH_U64 = 1u;

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

static constexpr uint32_t PREFLIGHT_DETAIL_PHASE_TRANSLATE = 0u;
static constexpr uint32_t PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE = 1u;
static constexpr uint32_t PREFLIGHT_DETAIL_PHASE_FIRST_TOUCH = 2u;
static constexpr uint32_t PREFLIGHT_DETAIL_PHASE_RESIDUAL_EMIT = 3u;
static constexpr uint32_t PREFLIGHT_DETAIL_PHASE_DELTA_EMIT = 4u;
static constexpr uint32_t PREFLIGHT_DETAIL_PHASE_ARENA_EMIT = 5u;
static constexpr uint32_t PREFLIGHT_DETAIL_PHASE_CHRONOLOGY = 6u;
static constexpr uint32_t PREFLIGHT_DETAIL_PHASE_COUNT = 7u;
static constexpr uint32_t PREFLIGHT_DETAIL_FAMILY_COUNT = 9u;

/* Profiling-only counters. A normal tracer carries a null pointer and generated
 * code contains no detail hooks unless OPENVM_RVR_NATIVE_DETAIL selected the
 * instrumented preflight project. */
typedef struct RvrNativeDetail {
  uint64_t family_cycles[PREFLIGHT_DETAIL_FAMILY_COUNT];
  uint64_t family_instructions[PREFLIGHT_DETAIL_FAMILY_COUNT];
  uint64_t phase_cycles[PREFLIGHT_DETAIL_PHASE_COUNT];
  uint64_t phase_samples[PREFLIGHT_DETAIL_PHASE_COUNT];
  uint64_t phase_events[PREFLIGHT_DETAIL_PHASE_COUNT];
  uint64_t phase_bytes[PREFLIGHT_DETAIL_PHASE_COUNT];
  uint64_t family_started;
  uint64_t outer_started;
  uint64_t timer_overhead;
  uint32_t sample_state;
  uint32_t sample_countdown;
  uint32_t current_family;
  uint32_t family_active;
} RvrNativeDetail;

static constexpr uint32_t PREFLIGHT_RECORD_DIRECT_FINAL = 1u;
static constexpr uint32_t PREFLIGHT_RECORD_OVERFLOW = 2u;
static constexpr uint32_t PREFLIGHT_RECORD_RESIDUAL_MEMORY_CHRONOLOGY = 4u;
static constexpr uint32_t PREFLIGHT_RECORD_VARIABLE_ROWS = 8u;
static constexpr uint32_t PREFLIGHT_RECORD_VARIABLE_ROW_STRIDE = 16u;
static constexpr uint32_t PREFLIGHT_RECORD_COMPACT_RESIDUAL_MEMORY = 32u;
static constexpr uint32_t PREFLIGHT_RECORD_DEVICE_AUX = 64u;
static constexpr uint32_t PREFLIGHT_RECORD_DEVICE_AUX_ORACLE = 128u;
static constexpr uint32_t PREFLIGHT_RECORD_DEVICE_CHRONOLOGY = 256u;

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
  /* Pooled counter reset metadata. Each index is appended exactly once, on
   * its counter's 0→nonzero transition. The host later scrubs only this
   * written prefix before returning both buffers to the cross-segment pool. */
  uint32_t* chip_counts_touched;
  uint32_t chip_counts_touched_len;
  uint32_t chip_counts_touched_cap;
  uint32_t* exec_frequencies_touched;
  uint32_t exec_frequencies_touched_len;
  uint32_t exec_frequencies_touched_cap;
  RvrNativeDetail* native_detail;
  DeviceAuxPatch* device_aux_patches;
  DeviceAuxReference* device_aux_references;
  uint64_t* dirty_memory_pages;
  uint32_t device_aux_patches_len;
  uint32_t device_aux_patches_cap;
  uint32_t device_aux_references_cap;
  uint32_t dirty_memory_pages_words;
  ProgramRunEntry* program_runs;
  DeviceProgramEntry* device_program_references;
  uint32_t program_runs_len;
  uint32_t program_runs_cap;
  uint32_t program_instruction_len;
  uint32_t device_program_references_len;
  uint32_t device_program_references_cap;
  G2ProducerV1* g2;
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
_Static_assert(sizeof(ProgramRunEntry) == PREFLIGHT_PROGRAM_RUN_ENTRY_SIZE,
               "ProgramRunEntry size drift");
_Static_assert(_Alignof(ProgramRunEntry) == PREFLIGHT_PROGRAM_RUN_ENTRY_ALIGN,
               "ProgramRunEntry align drift");
_Static_assert(sizeof(DeviceProgramEntry) == PREFLIGHT_DEVICE_PROGRAM_ENTRY_SIZE,
               "DeviceProgramEntry size drift");
_Static_assert(_Alignof(DeviceProgramEntry) == PREFLIGHT_DEVICE_PROGRAM_ENTRY_ALIGN,
               "DeviceProgramEntry align drift");
_Static_assert(sizeof(G2ProducerLaneV1) == 24,
               "G2ProducerLaneV1 size drift");
_Static_assert(_Alignof(G2ProducerLaneV1) == 8,
               "G2ProducerLaneV1 align drift");
_Static_assert(sizeof(G2ProducerV1) == 40, "G2ProducerV1 size drift");
_Static_assert(_Alignof(G2ProducerV1) == 8, "G2ProducerV1 align drift");
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
_Static_assert(offsetof(Tracer, chip_counts_touched) == 144,
               "Tracer chip_counts_touched offset drift");
_Static_assert(offsetof(Tracer, chip_counts_touched_len) == 152,
               "Tracer chip_counts_touched_len offset drift");
_Static_assert(offsetof(Tracer, chip_counts_touched_cap) == 156,
               "Tracer chip_counts_touched_cap offset drift");
_Static_assert(offsetof(Tracer, exec_frequencies_touched) == 160,
               "Tracer exec_frequencies_touched offset drift");
_Static_assert(offsetof(Tracer, exec_frequencies_touched_len) == 168,
               "Tracer exec_frequencies_touched_len offset drift");
_Static_assert(offsetof(Tracer, exec_frequencies_touched_cap) == 172,
               "Tracer exec_frequencies_touched_cap offset drift");
_Static_assert(offsetof(Tracer, native_detail) == 176,
               "Tracer native_detail offset drift");
_Static_assert(offsetof(Tracer, device_aux_patches) == 184,
               "Tracer device_aux_patches offset drift");
_Static_assert(offsetof(Tracer, device_aux_references) == 192,
               "Tracer device_aux_references offset drift");
_Static_assert(offsetof(Tracer, dirty_memory_pages) == 200,
               "Tracer dirty_memory_pages offset drift");
_Static_assert(offsetof(Tracer, device_aux_patches_len) == 208,
               "Tracer device_aux_patches_len offset drift");
_Static_assert(offsetof(Tracer, device_aux_patches_cap) == 212,
               "Tracer device_aux_patches_cap offset drift");
_Static_assert(offsetof(Tracer, device_aux_references_cap) == 216,
               "Tracer device_aux_references_cap offset drift");
_Static_assert(offsetof(Tracer, dirty_memory_pages_words) == 220,
               "Tracer dirty_memory_pages_words offset drift");
_Static_assert(offsetof(Tracer, program_runs) == 224,
               "Tracer program_runs offset drift");
_Static_assert(offsetof(Tracer, device_program_references) == 232,
               "Tracer device_program_references offset drift");
_Static_assert(offsetof(Tracer, program_runs_len) == 240,
               "Tracer program_runs_len offset drift");
_Static_assert(offsetof(Tracer, program_runs_cap) == 244,
               "Tracer program_runs_cap offset drift");
_Static_assert(offsetof(Tracer, g2) == 264, "Tracer g2 offset drift");
_Static_assert(offsetof(Tracer, program_instruction_len) == 248,
               "Tracer program_instruction_len offset drift");
_Static_assert(offsetof(Tracer, device_program_references_len) == 252,
               "Tracer device_program_references_len offset drift");
_Static_assert(offsetof(Tracer, device_program_references_cap) == 256,
               "Tracer device_program_references_cap offset drift");
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

#if defined(OPENVM_RVR_PREFLIGHT_NATIVE_DETAIL)
#include "openvm_tracer_preflight_native_detail.h"
#else
static __attribute__((always_inline)) inline uint64_t
preflight_detail_phase_begin(Tracer* restrict t, uint32_t phase,
                             uint64_t bytes) {
  return 0u;
}

static __attribute__((always_inline)) inline void preflight_detail_phase_end(
    Tracer* restrict t, uint32_t phase, uint64_t started) {}

static __attribute__((always_inline)) inline void preflight_detail_family(
    Tracer* restrict t, uint32_t family) {}
#endif

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

static constexpr uint32_t G2_REJECT_DIRTY_PAGE = 2u;
static constexpr uint32_t G2_REJECT_CUSTOM_SCRATCH = 3u;
static constexpr uint32_t G2_REJECT_CUSTOM_EVENT_COUNT = 4u;
static constexpr uint32_t G2_REJECT_PRODUCER_ABI = 5u;
static constexpr uint32_t G2_REJECT_LANE_CAPACITY = 6u;
static constexpr uint32_t G2_REJECT_RUN_COUNT = 7u;
static constexpr uint32_t G2_REJECT_STANDARD_KIND = 8u;
static constexpr uint32_t G2_REJECT_NARROW_VALUE = 9u;
static constexpr uint32_t G2_REJECT_STANDARD_PAIR = 10u;
static constexpr uint32_t G2_REJECT_POINTER = 11u;
static constexpr uint32_t G2_REJECT_LOAD_STORE_KIND = 12u;
static constexpr uint32_t G2_REJECT_LOAD_STORE_PAIR = 13u;
static constexpr uint32_t G2_REJECT_RESIDUAL_SHAPE = 14u;
static constexpr uint32_t G2_REJECT_RESIDUAL_PAIR = 15u;
static constexpr uint32_t G2_REJECT_CUSTOM_SCRATCH_CAPACITY = 16u;
static constexpr uint32_t G2_REJECT_ADDRESS = 17u;
static constexpr uint32_t G2_REJECT_DEVICE_AUX = 18u;

#if defined(OPENVM_RVR_PREFLIGHT_DELTA)
#include "openvm_tracer_preflight_delta.h"
#else
static __attribute__((always_inline)) inline bool
preflight_compact_residual_memory(Tracer* restrict t) {
  return t->g2 != NULL;
}

static __attribute__((always_inline)) inline bool
preflight_device_aux(Tracer* restrict t) {
  return t->g2 != NULL;
}

static __attribute__((always_inline)) inline bool
preflight_device_chronology(Tracer* restrict t) {
  return false;
}

static __attribute__((always_inline)) inline void
preflight_g2_emit_opaque_event_count(Tracer* restrict t,
                                     uint32_t event_count);
static __attribute__((always_inline)) inline uint32_t
preflight_g2_emit_residual_group(Tracer* restrict t,
                                 MemoryLogEntry const* restrict events,
                                 uint32_t event_count);

#include "openvm_g2_emission.h"

static __attribute__((always_inline)) inline bool
preflight_device_aux_oracle(Tracer* restrict t) {
  return t->g2 != NULL && OPENVM_G2_CHECKS_ENABLED;
}

static __attribute__((always_inline)) inline void
preflight_mark_dirty_memory_page(Tracer* restrict t, uint64_t address) {
  if (likely(t->g2 == NULL)) {
    return;
  }
  uint64_t page = address >> 12;
  uint64_t word = page >> 6;
  if (unlikely(word >= t->dirty_memory_pages_words ||
               t->dirty_memory_pages == NULL)) {
    t->g2->overflow = G2_REJECT_DIRTY_PAGE;
    return;
  }
  t->dirty_memory_pages[word] |= UINT64_C(1) << (page & 63u);
}

static constexpr uint32_t PREFLIGHT_DEVICE_AUX_TOKEN = 1u << 31;

static __attribute__((always_inline)) inline bool
preflight_g2_store_device_aux_reference(Tracer* restrict t,
                                        uint32_t event_index,
                                        uint32_t prev_timestamp,
                                        uint64_t prev_value) {
  if (likely(!preflight_device_aux_oracle(t))) {
    return true;
  }
  if (unlikely(t->device_aux_references == NULL ||
               event_index >= t->device_aux_references_cap)) {
    t->g2->overflow = G2_REJECT_DEVICE_AUX;
    return false;
  }
  t->device_aux_references[event_index] = (DeviceAuxReference){
      .prev_timestamp = prev_timestamp,
      ._reserved = 0u,
      .prev_value = prev_value,
  };
  return true;
}

static __attribute__((always_inline)) inline void preflight_append_device_patch(
    Tracer* restrict t, void* target, uint32_t token, uint32_t kind) {
  uint32_t event_index = token & ~PREFLIGHT_DEVICE_AUX_TOKEN;
  uint32_t patch_index = t->device_aux_patches_len++;
  G2ProducerLaneV1 const* restrict residual_lane =
      &t->g2->lanes[G2_PRODUCER_RESIDUAL_VALUE_SLOT];
  if (unlikely((token & PREFLIGHT_DEVICE_AUX_TOKEN) == 0u ||
               target == NULL || t->device_aux_patches == NULL ||
               patch_index >= t->device_aux_patches_cap ||
               event_index >= residual_lane->len ||
               (kind != PREFLIGHT_DEVICE_AUX_PATCH_U32 &&
                kind != PREFLIGHT_DEVICE_AUX_PATCH_U64))) {
    t->g2->overflow = G2_REJECT_DEVICE_AUX;
    return;
  }
  uint64_t expected = 0u;
  if (unlikely(preflight_device_aux_oracle(t))) {
    if (unlikely(t->device_aux_references == NULL ||
                 event_index >= t->device_aux_references_cap)) {
      t->g2->overflow = G2_REJECT_DEVICE_AUX;
      return;
    }
    DeviceAuxReference reference = t->device_aux_references[event_index];
    expected = kind == PREFLIGHT_DEVICE_AUX_PATCH_U32
                   ? (uint64_t)reference.prev_timestamp
                   : reference.prev_value;
  }
  t->device_aux_patches[patch_index] = (DeviceAuxPatch){
      .target = (uint64_t)(uintptr_t)target,
      .event_index = event_index,
      .kind = kind,
      .expected = expected,
  };
}

static __attribute__((always_inline)) inline void
preflight_store_prev_timestamp(Tracer* restrict t, uint32_t* target,
                               uint32_t token) {
  if (likely(!preflight_device_aux(t))) {
    *target = token;
    return;
  }
  preflight_append_device_patch(t, target, token,
                                PREFLIGHT_DEVICE_AUX_PATCH_U32);
  uint32_t event_index = token & ~PREFLIGHT_DEVICE_AUX_TOKEN;
  *target = preflight_device_aux_oracle(t) &&
                    likely(t->device_aux_references != NULL &&
                           event_index < t->device_aux_references_cap)
                ? t->device_aux_references[event_index].prev_timestamp
                : 0u;
}

static __attribute__((always_inline)) inline void preflight_store_prev_value(
    Tracer* restrict t, void* target, uint32_t token,
    uint64_t legacy_value) {
  if (likely(!preflight_device_aux(t))) {
    memcpy(target, &legacy_value, sizeof(legacy_value));
    return;
  }
  preflight_append_device_patch(t, target, token,
                                PREFLIGHT_DEVICE_AUX_PATCH_U64);
  uint32_t event_index = token & ~PREFLIGHT_DEVICE_AUX_TOKEN;
  uint64_t value = preflight_device_aux_oracle(t) &&
                           likely(t->device_aux_references != NULL &&
                                  event_index < t->device_aux_references_cap)
                       ? t->device_aux_references[event_index].prev_value
                       : 0u;
  memcpy(target, &value, sizeof(value));
}

static __attribute__((always_inline)) inline void
preflight_begin_custom_memory_capture(RvState* restrict state) {
  Tracer* restrict t = state->tracer;
  if (t->g2 != NULL) {
    if (unlikely(OPENVM_G2_CHECKS_ENABLED &&
                 (t->custom_memory_scratch == NULL ||
                  t->custom_memory_scratch_cap == 0u))) {
      t->g2->overflow = G2_REJECT_CUSTOM_SCRATCH;
      t->custom_memory_scratch_len = UINT32_MAX;
      return;
    }
    t->custom_memory_scratch_len = 0u;
  }
}

static __attribute__((always_inline)) inline MemoryLogEntry*
preflight_take_custom_memory_events(Tracer* restrict t, uint32_t event_count) {
  if (t->g2 != NULL) {
    uint32_t captured = t->custom_memory_scratch_len;
    t->custom_memory_scratch_len = UINT32_MAX;
    if (unlikely(captured != event_count ||
                 (OPENVM_G2_CHECKS_ENABLED &&
                  event_count > t->custom_memory_scratch_cap))) {
      t->g2->overflow = G2_REJECT_CUSTOM_EVENT_COUNT;
      return NULL;
    }
    uint32_t event_base = preflight_g2_emit_residual_group(
        t, t->custom_memory_scratch, event_count);
    if (unlikely(event_base == UINT32_MAX)) {
      return NULL;
    }
    for (uint32_t i = 0; i < event_count; ++i) {
      MemoryLogEntry* restrict event = &t->custom_memory_scratch[i];
      if (unlikely(!preflight_g2_store_device_aux_reference(
              t, event_base + i, event->prev_timestamp,
              event->prev_value))) {
        return NULL;
      }
      event->prev_timestamp = PREFLIGHT_DEVICE_AUX_TOKEN | (event_base + i);
    }
    preflight_g2_emit_opaque_event_count(t, event_count);
    return t->custom_memory_scratch;
  }
  if (unlikely(t->memory_log_len < event_count ||
               t->memory_log_len > t->memory_log_cap)) {
    return NULL;
  }
  return t->memory_log + t->memory_log_len - event_count;
}

static __attribute__((always_inline)) inline bool preflight_device_trace_pc(
    Tracer* restrict t, uint64_t pc, uint32_t exec_idx) {
  return t->g2 != NULL;
}

static __attribute__((always_inline)) inline void
preflight_g2_emit_opaque_event_count(Tracer* restrict t,
                                     uint32_t event_count) {
  G2ProducerV1* restrict g2 = t->g2;
  if (g2 == NULL) return;
  uint32_t index = preflight_g2_claim(
      g2, G2_PRODUCER_OPAQUE_EVENT_COUNT_SLOT, sizeof(uint32_t));
  if (unlikely(OPENVM_G2_CHECKS_ENABLED &&
               (index == UINT32_MAX || event_count == 0u))) {
    if (event_count == 0u) g2->overflow = G2_REJECT_CUSTOM_EVENT_COUNT;
    return;
  }
  G2ProducerLaneV1 const* restrict lane =
      &g2->lanes[G2_PRODUCER_OPAQUE_EVENT_COUNT_SLOT];
  ((uint32_t*)(g2->base + lane->offset))[index] = event_count;
}

static __attribute__((always_inline)) inline uint32_t
preflight_g2_load_store_slot(uint32_t kind) {
  switch (kind) {
    case 8u: return 5u;
    case 9u: return 7u;
    case 20u: return 9u;
    case 21u: return 11u;
    case 22u: return 13u;
    case 23u: return 15u;
    case 24u: return 17u;
    case 25u: return 19u;
    case 26u: return 21u;
    case 27u: return 23u;
    case 28u: return 25u;
    default: return UINT32_MAX;
  }
}

static __attribute__((always_inline)) inline uint32_t
preflight_g2_standard_slot(uint32_t kind) {
  switch (kind) {
    case 0u: return 27u;
    case 1u: return 29u;
    case 2u: return 31u;
    case 3u: return 33u;
    case 4u: return 35u;
    case 5u: return 37u;
    case 6u: return 39u;
    case 7u: return 41u;
    case 10u: return 43u;
    case 11u: return 45u;
    case 13u: return 57u;
    case 15u: return 47u;
    case 16u: return 49u;
    case 17u: return 51u;
    case 18u: return 53u;
    case 19u: return 55u;
    case 29u: return G2_PRODUCER_ADDI_SLOT;
    default: return UINT32_MAX;
  }
}

static __attribute__((always_inline)) inline void preflight_g2_emit_run(
    RvState* restrict state, uint32_t program_slot,
    uint32_t instruction_count) {
  G2ProducerV1* restrict g2 = state->tracer->g2;
  if (unlikely(g2 == NULL)) {
    return;
  }
  uint32_t index = preflight_g2_claim(g2, G2_PRODUCER_RUN_SLOT,
                                      sizeof(uint32_t));
  uint32_t next = g2->instruction_count + instruction_count;
  if (unlikely(index == UINT32_MAX || next < g2->instruction_count)) {
    g2->overflow = G2_REJECT_RUN_COUNT;
    return;
  }
  G2ProducerLaneV1 const* restrict lane =
      &g2->lanes[G2_PRODUCER_RUN_SLOT];
  ((uint32_t*)(g2->base + lane->offset))[index] = program_slot;
  g2->instruction_count = next;
}

static __attribute__((always_inline)) inline void preflight_g2_emit_addi(
    RvState* restrict state, uint64_t rs1_value) {
  G2ProducerV1* restrict g2 = state->tracer->g2;
  if (unlikely(g2 == NULL)) {
    return;
  }
  uint32_t index = preflight_g2_claim(g2, G2_PRODUCER_ADDI_SLOT,
                                      sizeof(uint64_t));
  if (unlikely(index == UINT32_MAX)) return;
  G2ProducerLaneV1 const* restrict lane =
      &g2->lanes[G2_PRODUCER_ADDI_SLOT];
  ((uint64_t*)(g2->base + lane->offset))[index] = rs1_value;
}

static __attribute__((always_inline)) inline void preflight_g2_emit_standard1(
    RvState* restrict state, uint32_t kind, uint64_t v0) {
  G2ProducerV1* restrict g2 = state->tracer->g2;
  uint32_t slot = preflight_g2_standard_slot(kind);
  if (unlikely(slot == UINT32_MAX || kind == 12u || kind == 14u)) {
    if (g2 != NULL) g2->overflow = G2_REJECT_STANDARD_KIND;
    return;
  }
  uint32_t index = preflight_g2_claim(
      g2, slot, kind == 13u ? sizeof(uint32_t) : sizeof(uint64_t));
  if (unlikely(index == UINT32_MAX)) return;
  G2ProducerLaneV1 const* restrict lane = &g2->lanes[slot];
  if (kind == 13u) {
    if (unlikely(v0 > UINT32_MAX)) {
      g2->overflow = G2_REJECT_NARROW_VALUE;
      return;
    }
    ((uint32_t*)(g2->base + lane->offset))[index] = (uint32_t)v0;
  } else {
    ((uint64_t*)(g2->base + lane->offset))[index] = v0;
  }
}

static __attribute__((always_inline)) inline void preflight_g2_emit_standard2(
    RvState* restrict state, uint32_t kind, uint64_t v0, uint64_t v1) {
  G2ProducerV1* restrict g2 = state->tracer->g2;
  uint32_t slot = preflight_g2_standard_slot(kind);
  if (unlikely(slot == UINT32_MAX || kind == 13u || kind == 29u)) {
    if (g2 != NULL) g2->overflow = G2_REJECT_STANDARD_PAIR;
    return;
  }
  uint32_t v0_index = preflight_g2_claim(g2, slot, sizeof(uint64_t));
  uint32_t v1_index = preflight_g2_claim(g2, slot + 1u, sizeof(uint64_t));
  if (unlikely(v0_index == UINT32_MAX || v1_index == UINT32_MAX)) return;
  G2ProducerLaneV1 const* restrict v0_lane = &g2->lanes[slot];
  G2ProducerLaneV1 const* restrict v1_lane = &g2->lanes[slot + 1u];
  ((uint64_t*)(g2->base + v0_lane->offset))[v0_index] = v0;
  ((uint64_t*)(g2->base + v1_lane->offset))[v1_index] = v1;
}

static __attribute__((always_inline)) inline bool
preflight_g2_validate_pointer(RvState* restrict state, uint64_t pointer) {
  if (unlikely(OPENVM_G2_CHECKS_ENABLED && pointer > UINT32_MAX &&
               state->tracer->g2 != NULL)) {
    state->tracer->g2->overflow = G2_REJECT_POINTER;
    return false;
  }
  return true;
}

/* Production-floor semantic load: reuse the block already emitted to the G2
 * current-value lane. `block1` is nonzero only for the exceptional crossing
 * residual path. */
static __attribute__((always_inline)) inline uint64_t
preflight_g2_floor_load_value(uint64_t block0, uint64_t block1,
                              uint64_t address) {
  uint32_t shift = (uint32_t)(address & (WORD_SIZE - 1u)) * 8u;
  return shift == 0u ? block0
                     : (block0 >> shift) | (block1 << (64u - shift));
}

static __attribute__((always_inline)) inline void
preflight_g2_emit_load_store(RvState* restrict state, uint32_t kind,
                             uint64_t base_pointer, uint64_t block_value) {
  G2ProducerV1* restrict g2 = state->tracer->g2;
  uint32_t slot = preflight_g2_load_store_slot(kind);
  if (unlikely(slot == UINT32_MAX || base_pointer > UINT32_MAX)) {
    if (g2 != NULL) g2->overflow = G2_REJECT_LOAD_STORE_KIND;
    return;
  }
  uint32_t pointer_index = preflight_g2_claim(g2, slot, sizeof(uint32_t));
  uint32_t block_index = preflight_g2_claim(g2, slot + 1u, sizeof(uint64_t));
  if (unlikely(pointer_index == UINT32_MAX || block_index == UINT32_MAX ||
               pointer_index != block_index)) {
    if (g2 != NULL) g2->overflow = G2_REJECT_LOAD_STORE_PAIR;
    return;
  }
  G2ProducerLaneV1 const* restrict pointer_lane = &g2->lanes[slot];
  G2ProducerLaneV1 const* restrict block_lane = &g2->lanes[slot + 1u];
  ((uint32_t*)(g2->base + pointer_lane->offset))[pointer_index] =
      (uint32_t)base_pointer;
  ((uint64_t*)(g2->base + block_lane->offset))[block_index] = block_value;
}

static __attribute__((always_inline)) inline uint8_t preflight_g2_residual_tag(
    uint8_t kind, uint8_t addr_space, uint8_t width) {
  uint8_t as_code = addr_space == AS_REGISTER
                        ? 0u
                        : (addr_space == AS_MEMORY
                               ? 1u
                               : (addr_space == AS_PUBLIC_VALUES ? 2u : 3u));
  uint8_t width_code = width == 0u   ? 0u
                       : width == 1u ? 1u
                       : width == 2u ? 2u
                       : width == 4u ? 3u
                       : width == 8u ? 4u
                                     : 7u;
  if (unlikely(OPENVM_G2_CHECKS_ENABLED &&
               (kind > PREFLIGHT_MEMORY_KIND_TOUCH || as_code == 3u ||
                width_code > 4u ||
                (kind != PREFLIGHT_MEMORY_KIND_TOUCH && width == 0u)))) {
    return UINT8_MAX;
  }
  return kind | (uint8_t)(as_code << 2u) | (uint8_t)(width_code << 4u);
}

static __attribute__((always_inline)) inline uint32_t
preflight_g2_emit_residual(Tracer* restrict t, uint8_t kind,
                           uint8_t addr_space, uint64_t address, uint8_t width,
                           uint64_t value, uint32_t timestamp) {
  G2ProducerV1* restrict g2 = t->g2;
  uint8_t tag = preflight_g2_residual_tag(kind, addr_space, width);
  if (unlikely(OPENVM_G2_CHECKS_ENABLED &&
               (g2 == NULL || address > UINT32_MAX || tag == UINT8_MAX))) {
    if (g2 != NULL) g2->overflow = G2_REJECT_RESIDUAL_SHAPE;
    return UINT32_MAX;
  }
  uint32_t ctrl_index = preflight_g2_claim(
      g2, G2_PRODUCER_RESIDUAL_CTRL_SLOT, sizeof(uint64_t));
  uint32_t tag_index = preflight_g2_claim(
      g2, G2_PRODUCER_RESIDUAL_TAG_SLOT, sizeof(uint8_t));
  uint32_t value_index = preflight_g2_claim(
      g2, G2_PRODUCER_RESIDUAL_VALUE_SLOT, sizeof(uint64_t));
  if (unlikely(OPENVM_G2_CHECKS_ENABLED &&
               (ctrl_index == UINT32_MAX || tag_index != ctrl_index ||
                value_index != ctrl_index))) {
    g2->overflow = G2_REJECT_RESIDUAL_PAIR;
    return UINT32_MAX;
  }
  G2ProducerLaneV1 const* restrict ctrl_lane =
      &g2->lanes[G2_PRODUCER_RESIDUAL_CTRL_SLOT];
  G2ProducerLaneV1 const* restrict tag_lane =
      &g2->lanes[G2_PRODUCER_RESIDUAL_TAG_SLOT];
  G2ProducerLaneV1 const* restrict value_lane =
      &g2->lanes[G2_PRODUCER_RESIDUAL_VALUE_SLOT];
  ((uint64_t*)(g2->base + ctrl_lane->offset))[ctrl_index] =
      (uint64_t)timestamp | (address << 32u);
  (g2->base + tag_lane->offset)[tag_index] = tag;
  ((uint64_t*)(g2->base + value_lane->offset))[value_index] = value;
  return ctrl_index;
}

/* Opaque/custom producers expose their exact event span only after the FFI
 * returns. Claim the three residual lanes once and fill each lane linearly
 * from the predecessor scratch captured in execution order. */
static __attribute__((always_inline)) inline uint32_t
preflight_g2_emit_residual_group(Tracer* restrict t,
                                 MemoryLogEntry const* restrict events,
                                 uint32_t event_count) {
  G2ProducerV1* restrict g2 = t->g2;
  if (unlikely(OPENVM_G2_CHECKS_ENABLED &&
               (g2 == NULL || event_count == 0u))) {
    if (g2 != NULL) g2->overflow = G2_REJECT_CUSTOM_EVENT_COUNT;
    return UINT32_MAX;
  }
  uint32_t ctrl_index = preflight_g2_claim_span(
      g2, G2_PRODUCER_RESIDUAL_CTRL_SLOT, event_count, sizeof(uint64_t));
  uint32_t tag_index = preflight_g2_claim_span(
      g2, G2_PRODUCER_RESIDUAL_TAG_SLOT, event_count, sizeof(uint8_t));
  uint32_t value_index = preflight_g2_claim_span(
      g2, G2_PRODUCER_RESIDUAL_VALUE_SLOT, event_count, sizeof(uint64_t));
  if (unlikely(OPENVM_G2_CHECKS_ENABLED &&
               (ctrl_index == UINT32_MAX || tag_index != ctrl_index ||
                value_index != ctrl_index))) {
    g2->overflow = G2_REJECT_RESIDUAL_PAIR;
    return UINT32_MAX;
  }
  uint64_t* restrict ctrl =
      (uint64_t*)(g2->base +
                  g2->lanes[G2_PRODUCER_RESIDUAL_CTRL_SLOT].offset) +
      ctrl_index;
  uint8_t* restrict tag =
      g2->base + g2->lanes[G2_PRODUCER_RESIDUAL_TAG_SLOT].offset + tag_index;
  uint64_t* restrict value =
      (uint64_t*)(g2->base +
                  g2->lanes[G2_PRODUCER_RESIDUAL_VALUE_SLOT].offset) +
      value_index;
  for (uint32_t i = 0; i < event_count; ++i) {
    MemoryLogEntry const* restrict event = &events[i];
    if (unlikely(OPENVM_G2_CHECKS_ENABLED &&
                 event->address > UINT32_MAX)) {
      g2->overflow = G2_REJECT_RESIDUAL_SHAPE;
    }
    ctrl[i] = (uint64_t)event->timestamp | (event->address << 32u);
  }
  for (uint32_t i = 0; i < event_count; ++i) {
    MemoryLogEntry const* restrict event = &events[i];
    uint8_t encoded = preflight_g2_residual_tag(
        event->kind, event->addr_space, event->width);
    if (unlikely(OPENVM_G2_CHECKS_ENABLED && encoded == UINT8_MAX)) {
      g2->overflow = G2_REJECT_RESIDUAL_SHAPE;
    }
    tag[i] = encoded;
  }
  for (uint32_t i = 0; i < event_count; ++i) {
    value[i] = events[i].value;
  }
  return ctrl_index;
}

/* The shared extension wrapper exports this symbol for every tracer mode.
 * Arena-native preflight never calls it from generated blocks. */
static __attribute__((always_inline)) inline void trace_block(
    RvState* restrict state, uint64_t pc, uint32_t block_insn_count) {}
#endif

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
  uint64_t detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  uint32_t timestamp = t->timestamp++;
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             detail_started);

  if (likely(preflight_device_aux(t) && !preflight_device_aux_oracle(t) &&
             t->g2 == NULL)) {
    *out_timestamp = timestamp;
    return 0u;
  }

  detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_TRANSLATE, 0u);
  uint64_t block_addr = address & ~(uint64_t)(WORD_SIZE - 1u);
  uint32_t block_idx = (uint32_t)(block_addr / WORD_SIZE);
  uint32_t* restrict shadow = preflight_shadow_for(t, addr_space);
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_TRANSLATE,
                             detail_started);

  detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  uint32_t prev_timestamp = shadow[block_idx];
  shadow[block_idx] = timestamp;
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             detail_started);
  if (prev_timestamp == 0u) {
    detail_started = preflight_detail_phase_begin(
        t, PREFLIGHT_DETAIL_PHASE_FIRST_TOUCH, sizeof(TouchedBlock));
    preflight_append_touched(t, addr_space, block_addr, initial_value);
    preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_FIRST_TOUCH,
                               detail_started);
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
  uint64_t detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  uint32_t timestamp = t->timestamp++;
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             detail_started);

  if (likely(preflight_device_aux(t) && !preflight_device_aux_oracle(t) &&
             t->g2 == NULL)) {
    *out_initial_value = 0u;
    *out_timestamp = timestamp;
    return 0u;
  }

  detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_TRANSLATE, 0u);
  uint64_t block_addr = preflight_block_addr(address);
  uint32_t block_idx = (uint32_t)(block_addr / WORD_SIZE);
  uint32_t* restrict shadow = preflight_shadow_for(t, addr_space);
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_TRANSLATE,
                             detail_started);

  detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  uint32_t prev_timestamp = shadow[block_idx];
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             detail_started);
  uint64_t initial_value = 0u;
  if (prev_timestamp == 0u || preflight_device_aux_oracle(t) || t->g2 != NULL) {
    /* First touch needs the boundary seed. The oracle and G2 direct-final
     * custom arenas also retain the current block for repeated TOUCH events. */
    if (prev_timestamp == 0u) {
      detail_started = preflight_detail_phase_begin(
          t, PREFLIGHT_DETAIL_PHASE_FIRST_TOUCH, sizeof(TouchedBlock));
    }
    initial_value =
        preflight_read_block_for_seed(state, addr_space, block_addr);
  }
  uint64_t aux_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  shadow[block_idx] = timestamp;
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             aux_started);
  if (prev_timestamp == 0u) {
    preflight_append_touched(t, addr_space, block_addr, initial_value);
    preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_FIRST_TOUCH,
                               detail_started);
  }

  *out_initial_value = initial_value;
  *out_timestamp = timestamp;
  return prev_timestamp;
}

#if defined(OPENVM_RVR_PREFLIGHT_DELTA)
static __attribute__((always_inline)) inline uint32_t
preflight_append_memory_record(Tracer* restrict t, uint8_t kind,
                               uint8_t addr_space, uint64_t address,
                               uint8_t width, uint64_t value,
                               uint64_t prev_value, uint32_t timestamp,
                               uint32_t prev_timestamp,
                               bool compact_residual) {
  uint64_t detail_bytes = compact_residual ? sizeof(DeltaMemoryLogEntry)
                                           : sizeof(MemoryLogEntry);
  if (unlikely(t->custom_memory_scratch_len != UINT32_MAX)) {
    detail_bytes += sizeof(MemoryLogEntry);
  }
  uint64_t detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_RESIDUAL_EMIT, detail_bytes);
  uint32_t idx = t->memory_log_len++;
  if (likely(idx < t->memory_log_cap)) {
    bool device_aux = preflight_device_aux(t);
    if (unlikely(device_aux && idx >= PREFLIGHT_DEVICE_AUX_TOKEN)) {
      t->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
    }
    if (unlikely(device_aux && preflight_device_aux_oracle(t))) {
      if (likely(t->device_aux_references != NULL &&
                 idx < t->device_aux_references_cap)) {
        t->device_aux_references[idx] = (DeviceAuxReference){
            .prev_timestamp = prev_timestamp,
            ._reserved = 0u,
            .prev_value = prev_value,
        };
      } else {
        t->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
      }
    }
    if (unlikely(t->custom_memory_scratch_len != UINT32_MAX)) {
      uint32_t scratch_idx = t->custom_memory_scratch_len++;
      if (likely(scratch_idx < t->custom_memory_scratch_cap)) {
        MemoryLogEntry scratch_entry = {
            .timestamp = timestamp,
            .prev_timestamp = device_aux && t->g2 == NULL
                                  ? (PREFLIGHT_DEVICE_AUX_TOKEN | idx)
                                  : prev_timestamp,
            .kind = kind,
            .addr_space = addr_space,
            .width = width,
            ._pad0 = 0,
            ._pad1 = device_aux && t->g2 == NULL ? idx + 1u : 0u,
            .address = address,
            .value = value,
            .prev_value = device_aux && t->g2 == NULL ? 0u : prev_value,
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
      preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_RESIDUAL_EMIT,
                                 detail_started);
      return idx;
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
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_RESIDUAL_EMIT,
                             detail_started);
  return idx;
}
#else
static __attribute__((always_inline)) inline uint32_t
preflight_append_memory_record(Tracer* restrict t, uint8_t kind,
                               uint8_t addr_space, uint64_t address,
                               uint8_t width, uint64_t value,
                               uint64_t prev_value, uint32_t timestamp,
                               uint32_t prev_timestamp,
                               bool compact_residual) {
  if (unlikely(compact_residual && t->g2 != NULL)) {
    uint64_t detail_started = preflight_detail_phase_begin(
        t, PREFLIGHT_DETAIL_PHASE_RESIDUAL_EMIT, 17u);
    bool custom_capture = t->custom_memory_scratch_len != UINT32_MAX;
    uint32_t index = custom_capture
                         ? UINT32_MAX
                         : preflight_g2_emit_residual(
                               t, kind, addr_space, address, width, value,
                               timestamp);
    if (likely(!custom_capture && index != UINT32_MAX)) {
      preflight_g2_store_device_aux_reference(t, index, prev_timestamp,
                                              prev_value);
    }
    if (unlikely(custom_capture)) {
      uint32_t scratch_idx = t->custom_memory_scratch_len++;
      if (likely(!OPENVM_G2_CHECKS_ENABLED ||
                 scratch_idx < t->custom_memory_scratch_cap)) {
        t->custom_memory_scratch[scratch_idx] = (MemoryLogEntry){
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
      } else {
        t->g2->overflow = G2_REJECT_CUSTOM_SCRATCH_CAPACITY;
      }
    }
    preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_RESIDUAL_EMIT,
                               detail_started);
    return index;
  }
  uint64_t detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_RESIDUAL_EMIT, sizeof(MemoryLogEntry));
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
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_RESIDUAL_EMIT,
                             detail_started);
  return idx;
}
#endif

/* Append one self-contained memory event. `address` need not be block-aligned;
 * the shadow index and touched-block key are derived from the aligned block.
 * `prev_value` is the block's value before this access (only consumed for
 * writes) and is supplied by the caller, which holds the store pointer. */
static __attribute__((always_inline)) inline uint32_t preflight_append_memory(
    Tracer* restrict t, uint8_t kind, uint8_t addr_space, uint64_t address,
    uint8_t width, uint64_t value, uint64_t prev_value) {
  bool compact_residual = preflight_compact_residual_memory(t);
  if (unlikely(compact_residual && address > UINT32_MAX)) {
    if (t->g2 != NULL) {
      t->g2->overflow = G2_REJECT_ADDRESS;
    } else if (t->delta_records != NULL) {
      t->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
    }
    return 0;
  }
  uint32_t timestamp;
  uint32_t prev_timestamp =
      preflight_touch(t, addr_space, address, prev_value, &timestamp);
  if (kind == PREFLIGHT_MEMORY_KIND_WRITE && addr_space == AS_MEMORY) {
    preflight_mark_dirty_memory_page(t, address);
  }
  bool g2_custom_capture =
      t->g2 != NULL && t->custom_memory_scratch_len != UINT32_MAX;
  uint32_t event_index = preflight_append_memory_record(
      t, kind, addr_space, address, width, value, prev_value, timestamp,
      prev_timestamp, compact_residual);
#if defined(OPENVM_RVR_PREFLIGHT_DELTA)
  return preflight_device_aux(t) && t->g2 == NULL
             ? (PREFLIGHT_DEVICE_AUX_TOKEN | event_index)
             : prev_timestamp;
#else
  return preflight_device_aux(t) && !g2_custom_capture
             ? (PREFLIGHT_DEVICE_AUX_TOKEN | event_index)
             : prev_timestamp;
#endif
}

static __attribute__((always_inline)) inline void trace_timestamp(
    RvState* restrict state) {
  state->tracer->timestamp++;
}

/* G2 standard rows are reconstructed from their value lanes on-device, so
 * their accesses must advance the host predecessor shadow without being
 * duplicated in the residual lanes. Custom and HintStore instructions keep
 * using the trace_* helpers below because their events are residual-native. */
static __attribute__((always_inline)) inline void preflight_g2_shadow_touch(
    Tracer* restrict t, uint8_t addr_space, uint64_t address,
    uint64_t current_value) {
  uint32_t timestamp;
  preflight_touch(t, addr_space, address, current_value, &timestamp);
}

static __attribute__((always_inline)) inline void
preflight_g2_shadow_reg_touch(RvState* restrict state, uint8_t idx) {
  uint64_t value = idx == 0 ? 0u : state->regs[idx];
  preflight_g2_shadow_touch(state->tracer, AS_REGISTER,
                            (uint32_t)idx * WORD_SIZE, value);
}

static __attribute__((always_inline)) inline void
preflight_g2_shadow_mem_touch(RvState* restrict state, uint64_t block_addr,
                              uint64_t block_value) {
  preflight_g2_shadow_touch(state->tracer, AS_MEMORY, block_addr, block_value);
}

static __attribute__((always_inline)) inline void
preflight_g2_shadow_mem_store_touch(RvState* restrict state,
                                    uint64_t block_addr) {
  preflight_g2_shadow_touch(state->tracer, AS_MEMORY, block_addr,
                            preflight_read_mem_block(state, block_addr));
  preflight_mark_dirty_memory_page(state->tracer, block_addr);
}

static __attribute__((always_inline)) inline void
preflight_g2_shadow_pv_touch(RvState* restrict state, uint64_t block_addr) {
  preflight_g2_shadow_touch(state->tracer, AS_PUBLIC_VALUES, block_addr,
                            preflight_read_pv_block(state->tracer, block_addr));
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
  uint64_t prev_value =
      preflight_device_aux(state->tracer) &&
              !preflight_device_aux_oracle(state->tracer) &&
              state->tracer->g2 == NULL
          ? 0u
          : state->regs[idx];
  return preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE,
                                 AS_REGISTER, (uint32_t)idx * WORD_SIZE,
                                 WORD_SIZE, new_val, prev_value);
}

/* Touch-only main-memory block access for migrated load/store opcodes (R3):
 * identical timestamp/shadow/touched bookkeeping to the logging trace_rd/wr
 * helpers, no MemoryLogEntry. `block_addr` must be block-aligned. Returns the
 * block's previous-access timestamp. */
static __attribute__((always_inline)) inline uint32_t trace_mem_touch(
    RvState* restrict state, uint64_t block_addr, uint64_t block_value) {
  Tracer* restrict t = state->tracer;
  if (likely(preflight_device_aux(t) && !preflight_device_aux_oracle(t) &&
             t->g2 == NULL)) {
    t->timestamp++;
    return 0u;
  }
  uint64_t detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  uint32_t timestamp = t->timestamp++;
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             detail_started);
  detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_TRANSLATE, 0u);
  uint32_t block_idx = (uint32_t)(block_addr / WORD_SIZE);
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_TRANSLATE,
                             detail_started);
  detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  uint32_t prev_timestamp = t->shadow_memory[block_idx];
  t->shadow_memory[block_idx] = timestamp;
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             detail_started);
  if (prev_timestamp == 0u) {
    detail_started = preflight_detail_phase_begin(
        t, PREFLIGHT_DETAIL_PHASE_FIRST_TOUCH, sizeof(TouchedBlock));
    preflight_append_touched(t, AS_MEMORY, block_addr, block_value);
    preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_FIRST_TOUCH,
                               detail_started);
  }
  return prev_timestamp;
}

/* Store-side variant: only read the pre-write block when it is the seed for
 * this segment. Subsequent stores reconstruct their predecessor on device. */
static __attribute__((always_inline)) inline uint32_t trace_mem_store_touch(
    RvState* restrict state, uint64_t block_addr) {
  Tracer* restrict t = state->tracer;
  if (likely(preflight_device_aux(t) && !preflight_device_aux_oracle(t) &&
             t->g2 == NULL)) {
    t->timestamp++;
    preflight_mark_dirty_memory_page(t, block_addr);
    return 0u;
  }
  uint64_t detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  uint32_t timestamp = t->timestamp++;
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             detail_started);
  detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_TRANSLATE, 0u);
  uint32_t block_idx = (uint32_t)(block_addr / WORD_SIZE);
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_TRANSLATE,
                             detail_started);
  detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  uint32_t prev_timestamp = t->shadow_memory[block_idx];
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             detail_started);
  uint64_t initial_value = 0u;
  if (prev_timestamp == 0u) {
    detail_started = preflight_detail_phase_begin(
        t, PREFLIGHT_DETAIL_PHASE_FIRST_TOUCH, sizeof(TouchedBlock));
    initial_value = rd_mem_u64(state->memory, block_addr);
  }
  uint64_t aux_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  t->shadow_memory[block_idx] = timestamp;
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             aux_started);
  if (prev_timestamp == 0u) {
    preflight_append_touched(t, AS_MEMORY, block_addr, initial_value);
    preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_FIRST_TOUCH,
                               detail_started);
  }
  preflight_mark_dirty_memory_page(t, block_addr);
  return prev_timestamp;
}

/* Touch-only public-values block access for inline REVEAL records. Mirrors
 * the verbose trace_wr_as_u64 timestamp/shadow/touched bookkeeping without
 * appending a MemoryLogEntry. */
static __attribute__((always_inline)) inline uint32_t trace_pv_touch(
    RvState* restrict state, uint64_t block_addr) {
  Tracer* restrict t = state->tracer;
  if (likely(preflight_device_aux(t) && !preflight_device_aux_oracle(t) &&
             t->g2 == NULL)) {
    t->timestamp++;
    return 0u;
  }
  uint64_t detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  uint32_t timestamp = t->timestamp++;
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             detail_started);
  detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_TRANSLATE, 0u);
  uint32_t block_idx = (uint32_t)(block_addr / WORD_SIZE);
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_TRANSLATE,
                             detail_started);
  detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  uint32_t prev_timestamp = t->shadow_public_values[block_idx];
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             detail_started);
  uint64_t initial_value = 0u;
  if (prev_timestamp == 0u) {
    detail_started = preflight_detail_phase_begin(
        t, PREFLIGHT_DETAIL_PHASE_FIRST_TOUCH, sizeof(TouchedBlock));
    initial_value = preflight_read_pv_block(t, block_addr);
  }
  uint64_t aux_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  t->shadow_public_values[block_idx] = timestamp;
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             aux_started);
  if (prev_timestamp == 0u) {
    preflight_append_touched(t, AS_PUBLIC_VALUES, block_addr, initial_value);
    preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_FIRST_TOUCH,
                               detail_started);
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
  if (likely(preflight_device_aux(t) && !preflight_device_aux_oracle(t) &&
             t->g2 == NULL)) {
    t->timestamp++;
    return 0u;
  }
  uint64_t detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  uint32_t timestamp = t->timestamp++;
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             detail_started);
  detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_TRANSLATE, 0u);
  uint32_t block_idx = idx;
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_TRANSLATE,
                             detail_started);
  detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  uint32_t prev_timestamp = t->shadow_register[block_idx];
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             detail_started);
  uint64_t initial_value = 0u;
  if (prev_timestamp == 0u) {
    detail_started = preflight_detail_phase_begin(
        t, PREFLIGHT_DETAIL_PHASE_FIRST_TOUCH, sizeof(TouchedBlock));
    initial_value = idx == 0 ? 0u : state->regs[idx];
  }
  uint64_t aux_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE, 0u);
  t->shadow_register[block_idx] = timestamp;
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_AUX_CAPTURE,
                             aux_started);
  if (prev_timestamp == 0u) {
    preflight_append_touched(t, AS_REGISTER, (uint32_t)idx * WORD_SIZE,
                             initial_value);
    preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_FIRST_TOUCH,
                               detail_started);
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

static __attribute__((always_inline)) inline void
preflight_mark_program_crossing_residual(Tracer* restrict t, uint32_t pc,
                                         uint32_t timestamp) {
  bool has_entry = t->program_log_len != 0u &&
                   t->program_log_len <= t->program_log_cap;
  if (!has_entry) {
    preflight_append_program(t, pc);
  } else {
    ProgramLogEntry* restrict last = &t->program_log[t->program_log_len - 1u];
    uint32_t last_pc = last->pc_and_flags & ~3u;
    if (last->timestamp != timestamp || last_pc != pc) {
      preflight_append_program(t, pc);
    }
  }
  if (likely(t->program_log_len != 0u &&
             t->program_log_len <= t->program_log_cap)) {
    ProgramLogEntry* restrict entry = &t->program_log[t->program_log_len - 1u];
    entry->timestamp = timestamp;
    entry->pc_and_flags |= PREFLIGHT_PROGRAM_CROSSING_RESIDUAL;
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
  uint32_t* record = (uint32_t*)(buf->base + off);
  return record;
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
  uint64_t detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_ARENA_EMIT, claim_bytes);
  buf->len = next;
  buf->core_off = next_rows;
  uint8_t* record = buf->base + off;
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_ARENA_EMIT,
                             detail_started);
  return record;
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

static __attribute__((always_inline)) inline bool
preflight_crosses_block(uint64_t addr, uint8_t width) {
  return (addr & (WORD_SIZE - 1u)) + width > WORD_SIZE;
}

static __attribute__((always_inline)) inline void preflight_patch_two_blocks(
    uint64_t prev0, uint64_t prev1, uint64_t addr, uint8_t width,
    uint64_t value, uint64_t* restrict next0, uint64_t* restrict next1) {
  uint8_t bytes[2 * WORD_SIZE];
  memcpy(bytes, &prev0, WORD_SIZE);
  memcpy(bytes + WORD_SIZE, &prev1, WORD_SIZE);
  memcpy(bytes + (addr & (WORD_SIZE - 1u)), &value, width);
  memcpy(next0, bytes, WORD_SIZE);
  memcpy(next1, bytes + WORD_SIZE, WORD_SIZE);
}

/* Rare crossing access: retain both complete block events in the residual
 * stream. The first predecessor is returned only to keep the ordinary compact
 * wire structurally complete; consumers select the residual row instead. */
static __attribute__((always_inline)) inline uint32_t
trace_crossing_mem_read_blocks(RvState* restrict state, uint64_t block_addr,
                               uint64_t block0, uint64_t* restrict block1,
                               uint32_t* restrict prev_timestamp1) {
  *block1 = preflight_read_mem_block(state, block_addr + WORD_SIZE);
  uint32_t prev0 = preflight_append_memory(
      state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY, block_addr,
      WORD_SIZE, block0, block0);
  uint32_t prev1 = preflight_append_memory(
      state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
      block_addr + WORD_SIZE, WORD_SIZE, *block1, *block1);
  if (prev_timestamp1 != NULL) {
    *prev_timestamp1 = prev1;
  }
  return prev0;
}

static __attribute__((always_inline)) inline uint32_t
trace_crossing_store_blocks(RvState* restrict state, uint64_t addr,
                            uint8_t width, uint64_t value, uint8_t addr_space,
                            uint64_t* restrict prev0,
                            uint64_t* restrict prev1,
                            uint32_t* restrict prev_timestamp1) {
  uint64_t block_addr = preflight_block_addr(addr);
  if (addr_space == AS_PUBLIC_VALUES) {
    *prev0 = preflight_read_pv_block(state->tracer, block_addr);
    *prev1 = preflight_read_pv_block(state->tracer, block_addr + WORD_SIZE);
  } else {
    *prev0 = preflight_read_mem_block(state, block_addr);
    *prev1 = preflight_read_mem_block(state, block_addr + WORD_SIZE);
  }
  uint64_t next0;
  uint64_t next1;
  preflight_patch_two_blocks(*prev0, *prev1, addr, width, value, &next0,
                             &next1);
  uint32_t first = preflight_append_memory(
      state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, addr_space, block_addr,
      WORD_SIZE, next0, *prev0);
  uint32_t write_prev1 = preflight_append_memory(
      state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, addr_space,
      block_addr + WORD_SIZE, WORD_SIZE, next1, *prev1);
  if (prev_timestamp1 != NULL) {
    *prev_timestamp1 = write_prev1;
  }
  return first;
}

/* Production-floor crossing residuals carry only current values. Device
 * replay reconstructs predecessor timestamps/values and touched-memory from
 * the segment-start state, so these helpers neither touch host shadows nor
 * advance the timestamp. */
static __attribute__((always_inline)) inline void
preflight_g2_floor_crossing_load(RvState* restrict state, uint64_t block_addr,
                                 uint64_t block0,
                                 uint64_t* restrict block1) {
  Tracer* restrict t = state->tracer;
  *block1 = preflight_read_mem_block(state, block_addr + WORD_SIZE);
  uint32_t first_timestamp = t->timestamp + 1u;
  preflight_append_memory_record(t, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                                 block_addr, WORD_SIZE, block0, block0,
                                 first_timestamp, 0u, true);
  preflight_append_memory_record(t, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
                                 block_addr + WORD_SIZE, WORD_SIZE, *block1,
                                 *block1, first_timestamp + 1u, 0u, true);
}

static __attribute__((always_inline)) inline void
preflight_g2_floor_crossing_store(RvState* restrict state, uint64_t addr,
                                  uint8_t width, uint64_t value,
                                  uint8_t addr_space, uint64_t block0) {
  Tracer* restrict t = state->tracer;
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t block1 = addr_space == AS_PUBLIC_VALUES
                        ? preflight_read_pv_block(t, block_addr + WORD_SIZE)
                        : preflight_read_mem_block(state,
                                                   block_addr + WORD_SIZE);
  uint64_t next0;
  uint64_t next1;
  preflight_patch_two_blocks(block0, block1, addr, width, value, &next0,
                             &next1);
  uint32_t first_timestamp = t->timestamp + 2u;
  preflight_append_memory_record(t, PREFLIGHT_MEMORY_KIND_WRITE, addr_space,
                                 block_addr, WORD_SIZE, next0, block0,
                                 first_timestamp, 0u, true);
  preflight_append_memory_record(t, PREFLIGHT_MEMORY_KIND_WRITE, addr_space,
                                 block_addr + WORD_SIZE, WORD_SIZE, next1,
                                 block1, first_timestamp + 1u, 0u, true);
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
  if (unlikely(preflight_crosses_block(addr, 2))) {
    uint64_t block1;
    trace_crossing_mem_read_blocks(state, block_addr, block, &block1, NULL);
  } else {
    preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ,
                            AS_MEMORY, block_addr, WORD_SIZE, block, block);
  }
}
static __attribute__((always_inline)) inline void trace_rd_mem_i16(
    RvState* restrict state, uint64_t addr, int16_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t block = preflight_read_mem_block(state, block_addr);
  if (unlikely(preflight_crosses_block(addr, 2))) {
    uint64_t block1;
    trace_crossing_mem_read_blocks(state, block_addr, block, &block1, NULL);
  } else {
    preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ,
                            AS_MEMORY, block_addr, WORD_SIZE, block, block);
  }
}
static __attribute__((always_inline)) inline void trace_rd_mem_u32(
    RvState* restrict state, uint64_t addr, uint32_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t block = preflight_read_mem_block(state, block_addr);
  if (unlikely(preflight_crosses_block(addr, 4))) {
    uint64_t block1;
    trace_crossing_mem_read_blocks(state, block_addr, block, &block1, NULL);
  } else {
    preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ,
                            AS_MEMORY, block_addr, WORD_SIZE, block, block);
  }
}
static __attribute__((always_inline)) inline void trace_rd_mem_i32(
    RvState* restrict state, uint64_t addr, int32_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t block = preflight_read_mem_block(state, block_addr);
  if (unlikely(preflight_crosses_block(addr, 4))) {
    uint64_t block1;
    trace_crossing_mem_read_blocks(state, block_addr, block, &block1, NULL);
  } else {
    preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ,
                            AS_MEMORY, block_addr, WORD_SIZE, block, block);
  }
}
static __attribute__((always_inline)) inline void trace_rd_mem_u64(
    RvState* restrict state, uint64_t addr, uint64_t val) {
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t block = preflight_device_aux(state->tracer) &&
                           !preflight_device_aux_oracle(state->tracer) &&
                           addr == block_addr
                       ? val
                       : preflight_read_mem_block(state, block_addr);
  if (unlikely(preflight_crosses_block(addr, 8))) {
    uint64_t block1;
    trace_crossing_mem_read_blocks(state, block_addr, block, &block1, NULL);
  } else {
    preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_READ,
                            AS_MEMORY, block_addr, WORD_SIZE, block, block);
  }
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
  if (unlikely(preflight_crosses_block(addr, 2))) {
    uint64_t prev0;
    uint64_t prev1;
    trace_crossing_store_blocks(state, addr, 2, new_val, AS_MEMORY, &prev0,
                                &prev1, NULL);
    return;
  }
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t prev_block = preflight_read_mem_block(state, block_addr);
  uint64_t block =
      preflight_patch_mem_block(prev_block, addr, 2, (uint64_t)new_val);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, AS_MEMORY,
                          block_addr, WORD_SIZE, block, prev_block);
}
static __attribute__((always_inline)) inline void trace_wr_mem_u32(
    RvState* restrict state, uint64_t addr, uint32_t new_val) {
  if (unlikely(preflight_crosses_block(addr, 4))) {
    uint64_t prev0;
    uint64_t prev1;
    trace_crossing_store_blocks(state, addr, 4, new_val, AS_MEMORY, &prev0,
                                &prev1, NULL);
    return;
  }
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t prev_block = preflight_read_mem_block(state, block_addr);
  uint64_t block =
      preflight_patch_mem_block(prev_block, addr, 4, (uint64_t)new_val);
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, AS_MEMORY,
                          block_addr, WORD_SIZE, block, prev_block);
}
static __attribute__((always_inline)) inline void trace_wr_mem_u64(
    RvState* restrict state, uint64_t addr, uint64_t new_val) {
  if (unlikely(preflight_crosses_block(addr, 8))) {
    uint64_t prev0;
    uint64_t prev1;
    trace_crossing_store_blocks(state, addr, 8, new_val, AS_MEMORY, &prev0,
                                &prev1, NULL);
    return;
  }
  uint64_t block_addr = preflight_block_addr(addr);
  uint64_t prev_block =
      preflight_device_aux(state->tracer) &&
              !preflight_device_aux_oracle(state->tracer) &&
              state->tracer->g2 == NULL
          ? 0u
          : preflight_read_mem_block(state, block_addr);
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
    uint64_t prev_block =
        preflight_device_aux(state->tracer) &&
                !preflight_device_aux_oracle(state->tracer) &&
                state->tracer->g2 == NULL
            ? 0u
            : preflight_read_mem_block(state, block_addr);
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
    if (state->tracer->g2 != NULL) {
      state->tracer->g2->overflow = G2_REJECT_ADDRESS;
    } else if (state->tracer->delta_records != NULL) {
      state->tracer->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
    }
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
      if (state->tracer->g2 != NULL) {
        state->tracer->g2->overflow = G2_REJECT_ADDRESS;
      } else if (state->tracer->delta_records != NULL) {
        state->tracer->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
      }
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
  if (unlikely(preflight_crosses_block(addr, WORD_SIZE))) {
    uint64_t prev0;
    uint64_t prev1;
    trace_crossing_store_blocks(state, addr, WORD_SIZE, new_val,
                                (uint8_t)addr_space, &prev0, &prev1, NULL);
    return;
  }
  uint64_t block_addr = preflight_block_addr(addr);
  /* Public-values reveal writes are traced before the store, so the previous
   * value is still in the aliased public-values buffer. */
  uint64_t prev_block =
      addr_space == AS_PUBLIC_VALUES &&
              !(preflight_device_aux(state->tracer) &&
                !preflight_device_aux_oracle(state->tracer) &&
                state->tracer->g2 == NULL)
          ? preflight_read_pv_block(state->tracer, block_addr)
          : 0u;
  preflight_append_memory(state->tracer, PREFLIGHT_MEMORY_KIND_WRITE,
                          (uint8_t)addr_space, block_addr, WORD_SIZE, new_val,
                          prev_block);
  trace_timestamp(state);
}

static __attribute__((always_inline)) inline void trace_wr_as(
    RvState* restrict state, uint64_t addr, uint64_t new_val, uint32_t width,
    uint32_t addr_space) {
  if (unlikely(width > 1u && preflight_crosses_block(addr, (uint8_t)width))) {
    uint64_t prev0;
    uint64_t prev1;
    trace_crossing_store_blocks(state, addr, (uint8_t)width, new_val,
                                (uint8_t)addr_space, &prev0, &prev1, NULL);
    return;
  }
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
  if (width > 1u) {
    trace_timestamp(state);
  }
}

static __attribute__((always_inline)) inline void trace_pc(
    RvState* restrict state, uint64_t pc) {
  preflight_append_program(state->tracer, pc);
}

static __attribute__((always_inline)) inline void preflight_add_counter(
    uint32_t* restrict values, uint32_t values_len,
    uint32_t* restrict touched, uint32_t* restrict touched_len,
    uint32_t touched_cap, uint32_t index, uint32_t count) {
  if (unlikely(values == NULL || index >= values_len)) {
    return;
  }
  uint32_t* restrict value = &values[index];
  if (*value == 0u && count != 0u && touched != NULL) {
    uint32_t len = *touched_len;
    if (unlikely(len >= touched_cap)) {
      /* UINT32_MAX is an impossible valid cursor and makes the Rust capacity
       * check fail closed before any pooled counter can escape. */
      *touched_len = UINT32_MAX;
      return;
    }
    touched[len] = index;
    *touched_len = len + 1u;
  }
  *value += count;
}

/* ZG2 single-pass dispatch accounting. Every instruction increments its
 * compile-time filtered program index. An inline instruction whose chip
 * target is DIRECT_FINAL already carries its pc/timestamp in the final wire
 * stream, so the duplicate ProgramLogEntry is suppressed. Pooled/unstaged
 * compact records keep the log because host expansion still consumes it. */
static __attribute__((always_inline)) inline void trace_pc_indexed(
    RvState* restrict state, uint64_t pc, uint32_t exec_idx,
    uint32_t inline_chip_idx, bool delta_inline, uint32_t detail_family) {
  Tracer* restrict t = state->tracer;
  if (unlikely(preflight_device_trace_pc(t, pc, exec_idx))) {
    return;
  }
  preflight_detail_family(t, detail_family);
  uint64_t detail_started = preflight_detail_phase_begin(
      t, PREFLIGHT_DETAIL_PHASE_CHRONOLOGY, 0u);
  if (likely(t->exec_frequencies != NULL &&
             exec_idx < t->exec_frequencies_len)) {
    preflight_add_counter(
        t->exec_frequencies, t->exec_frequencies_len,
        t->exec_frequencies_touched, &t->exec_frequencies_touched_len,
        t->exec_frequencies_touched_cap, exec_idx, 1u);
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
  preflight_detail_phase_end(t, PREFLIGHT_DETAIL_PHASE_CHRONOLOGY,
                             detail_started);
}

static __attribute__((always_inline)) inline void trace_chip(
    RvState* restrict state, uint32_t chip_idx, uint32_t count) {
  Tracer* restrict t = state->tracer;
  preflight_add_counter(
      t->chip_counts, t->chip_counts_len, t->chip_counts_touched,
      &t->chip_counts_touched_len, t->chip_counts_touched_cap, chip_idx,
      count);
}

#endif /* OPENVM_TRACER_PREFLIGHT_COMMON_H */
