/*
 * Dispatch table and forwarding stubs for the Rv64Io operations: the
 * hint-store consumers (HINT_STOREW, HINT_BUFFER) and REVEAL.
 */

#include "rv64io_callbacks.h"

#include "openvm.h"
#include "openvm_io.h"

static constexpr uint32_t RVR_HINT_MAX_BUFFER_DWORDS = 1023u;

#ifdef OPENVM_TRACER_PREFLIGHT_H
typedef struct {
  uint32_t prev_timestamp;
} Rv64HintReadAux;

typedef struct {
  uint32_t num_words;
  uint32_t from_pc;
  uint32_t timestamp;
  uint32_t mem_ptr_ptr;
  uint32_t mem_ptr;
  Rv64HintReadAux mem_ptr_aux_record;
  uint32_t num_words_ptr;
  Rv64HintReadAux num_words_read;
} Rv64HintStoreRecordHeader;

typedef struct {
  uint32_t prev_timestamp;
  uint8_t prev_data[WORD_SIZE];
  uint8_t data[WORD_SIZE];
} Rv64HintStoreVar;

static_assert(sizeof(Rv64HintStoreRecordHeader) == 32,
              "HintStore header ABI drift");
static_assert(sizeof(Rv64HintStoreVar) == 20, "HintStore row ABI drift");

static inline Rv64HintStoreRecordHeader* rvr_claim_hintstore_record(
    RvState* state, uint32_t chip_idx, uint32_t num_words,
    Rv64HintStoreVar** vars) {
  if (unlikely(chip_idx == UINT32_MAX || num_words == 0u ||
               num_words > RVR_HINT_MAX_BUFFER_DWORDS)) {
    if (chip_idx != UINT32_MAX && state->tracer->chip_records != NULL &&
        chip_idx < state->tracer->chip_counts_len) {
      state->tracer->chip_records[chip_idx].flags |= PREFLIGHT_RECORD_OVERFLOW;
    }
    *vars = NULL;
    return NULL;
  }
  uint32_t record_bytes =
      (uint32_t)sizeof(Rv64HintStoreRecordHeader) +
      num_words * (uint32_t)sizeof(Rv64HintStoreVar);
  uint8_t* record = preflight_claim_variable_record(
      state, chip_idx, record_bytes, num_words);
  if (unlikely(record == NULL)) {
    *vars = NULL;
    return NULL;
  }
  *vars = (Rv64HintStoreVar*)(record + sizeof(Rv64HintStoreRecordHeader));
  return (Rv64HintStoreRecordHeader*)record;
}

static inline void rvr_hintstore_trace_word(RvState* state,
                                            uint64_t dest_addr,
                                            uint64_t prev_data, uint64_t word,
                                            Rv64HintStoreVar* var) {
  uint64_t block_addr = preflight_block_addr(dest_addr);
  uint32_t prev_timestamp = preflight_append_memory(
      state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, AS_MEMORY, block_addr,
      WORD_SIZE, word, prev_data);
  if (likely(var != NULL)) {
    preflight_store_prev_timestamp(state->tracer, &var->prev_timestamp,
                                   prev_timestamp);
    preflight_store_prev_value(state->tracer, var->prev_data, prev_timestamp,
                               prev_data);
    memcpy(var->data, &word, WORD_SIZE);
  }
}
#endif

static thread_local Rv64IoHostCallbacks g_rv64io_host_callbacks;

void register_rv64io_host_callbacks(const Rv64IoHostCallbacks* cb) {
  g_rv64io_host_callbacks = *cb;
}

bool openvm_hint_storew(void* state, uint64_t dest_addr, uint32_t from_pc,
                        uint32_t from_timestamp, uint32_t mem_ptr_ptr,
                        uint32_t mem_ptr_prev_timestamp, uint32_t chip_idx) {
#ifdef OPENVM_TRACER_PREFLIGHT_H
  if (unlikely(dest_addr > OPENVM_MEM_SIZE - WORD_SIZE)) {
    return false;
  }
  RvState* rv_state = (RvState*)state;
  bool reconstruct_prev = chip_idx != UINT32_MAX &&
                          preflight_device_aux(rv_state->tracer) &&
                          !preflight_device_aux_oracle(rv_state->tracer);
  uint64_t prev_data =
      reconstruct_prev ? 0u : read_mem_u64(rv_state->memory, dest_addr);
#endif
  bool ok =
      g_rv64io_host_callbacks.hint_storew(openvm_get_io_ctx(), dest_addr);
  if (unlikely(!ok)) {
    return false;
  }
  uint64_t word = read_mem_u64(((RvState*)state)->memory, dest_addr);
#ifdef OPENVM_TRACER_PREFLIGHT_H
  bool emit_direct = chip_idx != UINT32_MAX;
  Rv64HintStoreVar* vars = NULL;
  Rv64HintStoreRecordHeader* record = emit_direct
      ? rvr_claim_hintstore_record(rv_state, chip_idx, 1u, &vars)
      : NULL;
  if (likely(record != NULL)) {
    *record = (Rv64HintStoreRecordHeader){
        .num_words = 1u,
        .from_pc = from_pc,
        .timestamp = from_timestamp,
        .mem_ptr_ptr = mem_ptr_ptr,
        .mem_ptr = (uint32_t)dest_addr,
        .mem_ptr_aux_record = {.prev_timestamp = 0u},
        .num_words_ptr = UINT32_MAX,
        .num_words_read = {.prev_timestamp = 0u},
    };
    preflight_store_prev_timestamp(
        rv_state->tracer, &record->mem_ptr_aux_record.prev_timestamp,
        mem_ptr_prev_timestamp);
  }
  rvr_hintstore_trace_word(rv_state, dest_addr, prev_data, word, vars);
#else
  trace_wr_mem_u64((RvState*)state, dest_addr, word);
#endif
  return true;
}

bool openvm_hint_buffer(void* state, uint64_t dest_addr, uint16_t num_words,
                        uint32_t from_pc, uint32_t from_timestamp,
                        uint32_t mem_ptr_ptr,
                        uint32_t mem_ptr_prev_timestamp,
                        uint32_t num_words_ptr,
                        uint32_t num_words_prev_timestamp,
                        uint32_t chip_idx) {
#ifdef OPENVM_TRACER_PREFLIGHT_H
  size_t num_bytes = (size_t)num_words * WORD_SIZE;
  if (unlikely(num_words == 0u || num_words > RVR_HINT_MAX_BUFFER_DWORDS ||
               dest_addr > OPENVM_MEM_SIZE ||
               num_bytes > OPENVM_MEM_SIZE - dest_addr)) {
    return false;
  }
  RvState* rv_state = (RvState*)state;
  bool reconstruct_prev = chip_idx != UINT32_MAX &&
                          preflight_device_aux(rv_state->tracer) &&
                          !preflight_device_aux_oracle(rv_state->tracer);
  uint64_t prev_words[RVR_HINT_MAX_BUFFER_DWORDS];
  for (uint32_t i = 0; i < num_words; i++) {
    prev_words[i] = reconstruct_prev
                        ? 0u
                        : read_mem_u64(rv_state->memory,
                                       dest_addr + (uint64_t)i * WORD_SIZE);
  }
#endif
  bool ok = g_rv64io_host_callbacks.hint_buffer(openvm_get_io_ctx(), dest_addr,
                                                num_words);
  if (unlikely(!ok)) {
    return false;
  }
#ifdef OPENVM_TRACER_PREFLIGHT_H
  bool emit_direct = chip_idx != UINT32_MAX;
  Rv64HintStoreVar* vars = NULL;
  Rv64HintStoreRecordHeader* record = emit_direct
      ? rvr_claim_hintstore_record(rv_state, chip_idx, num_words, &vars)
      : NULL;
  if (likely(record != NULL)) {
    *record = (Rv64HintStoreRecordHeader){
        .num_words = num_words,
        .from_pc = from_pc,
        .timestamp = from_timestamp,
        .mem_ptr_ptr = mem_ptr_ptr,
        .mem_ptr = (uint32_t)dest_addr,
        .mem_ptr_aux_record = {.prev_timestamp = 0u},
        .num_words_ptr = num_words_ptr,
        .num_words_read = {.prev_timestamp = 0u},
    };
    preflight_store_prev_timestamp(
        rv_state->tracer, &record->mem_ptr_aux_record.prev_timestamp,
        mem_ptr_prev_timestamp);
    preflight_store_prev_timestamp(
        rv_state->tracer, &record->num_words_read.prev_timestamp,
        num_words_prev_timestamp);
  }
#endif
  for (uint32_t i = 0; i < num_words; i++) {
    if (i != 0) {
      trace_timestamp((RvState*)state);
      trace_timestamp((RvState*)state);
    }
    uint64_t addr = dest_addr + (uint64_t)i * WORD_SIZE;
    uint64_t word = read_mem_u64(((RvState*)state)->memory, addr);
#ifdef OPENVM_TRACER_PREFLIGHT_H
    rvr_hintstore_trace_word(rv_state, addr, prev_words[i], word,
                             vars == NULL ? NULL : &vars[i]);
#else
    trace_wr_mem_u64((RvState*)state, addr, word);
#endif
  }
  return true;
}

bool openvm_reveal(uint64_t src_val, uint64_t addr, uint8_t width) {
  return g_rv64io_host_callbacks.reveal(openvm_get_io_ctx(), src_val, addr,
                                        width);
}
