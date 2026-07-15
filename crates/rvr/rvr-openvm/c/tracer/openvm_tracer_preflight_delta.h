/* Delta/device-only preflight helpers. This header is selected by the C
 * project generator only for the delta/device route. */

#ifndef OPENVM_TRACER_PREFLIGHT_DELTA_H
#define OPENVM_TRACER_PREFLIGHT_DELTA_H

/* Delta-only residual-memory wire. The chronological decoder reconstructs
 * prev_timestamp and prev_value. */
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

_Static_assert(sizeof(DeltaMemoryLogEntry) ==
                   PREFLIGHT_DELTA_MEMORY_LOG_ENTRY_SIZE,
               "DeltaMemoryLogEntry size drift");
_Static_assert(_Alignof(DeltaMemoryLogEntry) ==
                   PREFLIGHT_DELTA_MEMORY_LOG_ENTRY_ALIGN,
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
_Static_assert(sizeof(PreflightDeltaRecord) == PREFLIGHT_DELTA_RECORD_SIZE,
               "PreflightDeltaRecord size drift");
_Static_assert(_Alignof(PreflightDeltaRecord) == 8,
               "PreflightDeltaRecord align drift");

static __attribute__((always_inline)) inline bool
preflight_compact_residual_memory(Tracer* restrict t) {
  return t->delta_records != NULL &&
         (t->delta_records->flags &
          PREFLIGHT_RECORD_COMPACT_RESIDUAL_MEMORY) != 0u;
}

static constexpr uint32_t PREFLIGHT_DEVICE_AUX_TOKEN = 1u << 31;

static __attribute__((always_inline)) inline bool
preflight_device_aux(Tracer* restrict t) {
  return t->delta_records != NULL &&
         (t->delta_records->flags & PREFLIGHT_RECORD_DEVICE_AUX) != 0u;
}

static __attribute__((always_inline)) inline bool
preflight_device_aux_oracle(Tracer* restrict t) {
  return t->delta_records != NULL &&
         (t->delta_records->flags & PREFLIGHT_RECORD_DEVICE_AUX_ORACLE) != 0u;
}

static __attribute__((always_inline)) inline bool
preflight_device_chronology(Tracer* restrict t) {
  return t->delta_records != NULL &&
         (t->delta_records->flags & PREFLIGHT_RECORD_DEVICE_CHRONOLOGY) != 0u;
}

static __attribute__((always_inline)) inline void
preflight_mark_dirty_memory_page(Tracer* restrict t, uint64_t address) {
  if (likely(!preflight_device_aux(t))) {
    return;
  }
  uint64_t page = address >> 12;
  uint64_t word = page >> 6;
  if (unlikely(word >= t->dirty_memory_pages_words ||
               t->dirty_memory_pages == NULL)) {
    t->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
    return;
  }
  t->dirty_memory_pages[word] |= UINT64_C(1) << (page & 63u);
}

static __attribute__((always_inline)) inline void preflight_append_device_patch(
    Tracer* restrict t, void* target, uint32_t token, uint32_t kind) {
  if (unlikely((token & PREFLIGHT_DEVICE_AUX_TOKEN) == 0u)) {
    t->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
    return;
  }
  uint32_t event_index = token & ~PREFLIGHT_DEVICE_AUX_TOKEN;
  uint32_t patch_index = t->device_aux_patches_len++;
  if (unlikely(t->device_aux_patches == NULL ||
               patch_index >= t->device_aux_patches_cap ||
               event_index >= t->memory_log_len)) {
    t->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
    return;
  }
  uint64_t expected = 0u;
  if (unlikely(preflight_device_aux_oracle(t))) {
    if (unlikely(t->device_aux_references == NULL ||
                 event_index >= t->device_aux_references_cap)) {
      t->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
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

/* Custom direct-final emitters use these stores for every predecessor field.
 * Production records receive a placeholder and an explicit patch descriptor;
 * oracle records receive the legacy-host value first, then the same device
 * value overwrites it before trace generation. */
static __attribute__((always_inline)) inline void
preflight_store_prev_timestamp(Tracer* restrict t, uint32_t* target,
                               uint32_t token_or_value) {
  if (likely(!preflight_device_aux(t))) {
    *target = token_or_value;
    return;
  }
  preflight_append_device_patch(t, target, token_or_value,
                                PREFLIGHT_DEVICE_AUX_PATCH_U32);
  uint32_t event_index = token_or_value & ~PREFLIGHT_DEVICE_AUX_TOKEN;
  *target = preflight_device_aux_oracle(t) &&
                    likely(t->device_aux_references != NULL &&
                           event_index < t->device_aux_references_cap)
                ? t->device_aux_references[event_index].prev_timestamp
                : 0u;
}

static __attribute__((always_inline)) inline void preflight_store_prev_value(
    Tracer* restrict t, void* target, uint32_t token_or_timestamp,
    uint64_t legacy_value) {
  if (likely(!preflight_device_aux(t))) {
    memcpy(target, &legacy_value, sizeof(legacy_value));
    return;
  }
  preflight_append_device_patch(t, target, token_or_timestamp,
                                PREFLIGHT_DEVICE_AUX_PATCH_U64);
  uint32_t event_index = token_or_timestamp & ~PREFLIGHT_DEVICE_AUX_TOKEN;
  uint64_t value = preflight_device_aux_oracle(t) &&
                           likely(t->device_aux_references != NULL &&
                                  event_index < t->device_aux_references_cap)
                       ? t->device_aux_references[event_index].prev_value
                       : 0u;
  memcpy(target, &value, sizeof(value));
}

/* Custom VecHeap emitters run after their traced accesses, so compact residual
 * entries no longer contain the predecessor fields they need. Capture only
 * that instruction's events in bounded scratch. */
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

/* Reserve one basic block's delta records with a single cursor/capacity
 * update. Generated C indexes the returned span at compile-time constants. */
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
  uint64_t detail_started = preflight_detail_phase_begin(
      state->tracer, PREFLIGHT_DETAIL_PHASE_DELTA_EMIT, 0u);
  buf->len = off + bytes;
  PreflightDeltaRecord* records = (PreflightDeltaRecord*)(buf->base + off);
  preflight_detail_phase_end(state->tracer,
                             PREFLIGHT_DETAIL_PHASE_DELTA_EMIT,
                             detail_started);
  return records;
}

static __attribute__((always_inline)) inline void preflight_write_delta2(
    RvState* restrict state, PreflightDeltaRecord* restrict record,
    uint32_t from_pc, uint32_t from_timestamp, uint64_t v1, uint64_t v2) {
  if (unlikely(record == NULL)) {
    return;
  }
  uint64_t detail_started = preflight_detail_phase_begin(
      state->tracer, PREFLIGHT_DETAIL_PHASE_DELTA_EMIT,
      sizeof(PreflightDeltaRecord));
  *record = (PreflightDeltaRecord){
      .from_pc = from_pc,
      .from_timestamp = from_timestamp,
      .v1 = v1,
      .v2 = v2,
  };
  preflight_detail_phase_end(state->tracer,
                             PREFLIGHT_DETAIL_PHASE_DELTA_EMIT,
                             detail_started);
}

static __attribute__((always_inline)) inline bool preflight_device_trace_pc(
    Tracer* restrict t, uint64_t pc, uint32_t exec_idx) {
  if (likely(!preflight_device_chronology(t))) {
    return false;
  }
  if (likely(!preflight_device_aux_oracle(t))) {
    return true;
  }
  uint32_t reference_idx = t->device_program_references_len++;
  if (unlikely(t->device_program_references == NULL ||
               reference_idx >= t->device_program_references_cap ||
               pc > UINT32_MAX)) {
    t->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
    return true;
  }
  t->device_program_references[reference_idx] = (DeviceProgramEntry){
      .pc = (uint32_t)pc,
      .filtered_index = exec_idx,
  };
  return false;
}

static __attribute__((always_inline)) inline void trace_block(
    RvState* restrict state, uint64_t pc, uint32_t block_insn_count) {
  Tracer* restrict t = state->tracer;
  if (likely(!preflight_device_chronology(t))) {
    return;
  }
  uint32_t run_idx = t->program_runs_len++;
  uint32_t chronology_offset = t->program_instruction_len;
  uint32_t next = chronology_offset + block_insn_count;
  if (unlikely(t->program_runs == NULL || run_idx >= t->program_runs_cap ||
               next < chronology_offset || pc > UINT32_MAX)) {
    t->delta_records->flags |= PREFLIGHT_RECORD_OVERFLOW;
    return;
  }
  t->program_runs[run_idx] = (ProgramRunEntry){
      .first_pc = (uint32_t)pc,
      .instruction_count = block_insn_count,
      .chronology_offset = chronology_offset,
      .complete = 1u,
  };
  t->program_instruction_len = next;
}

#endif /* OPENVM_TRACER_PREFLIGHT_DELTA_H */
