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

static inline void rvr_hintstore_write_word(RvState* state, uint64_t dest_addr,
                                            uint64_t word, bool emit_direct,
                                            Rv64HintStoreVar* var) {
  if (likely(emit_direct)) {
    uint64_t block_addr = preflight_block_addr(dest_addr);
    uint64_t prev_data =
        preflight_device_aux(state->tracer) &&
                !preflight_device_aux_oracle(state->tracer)
            ? 0u
            : preflight_read_mem_block(state, block_addr);
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
    wr_mem_u64(state->memory, dest_addr, word);
  } else {
    wr_mem_u64_traced(state, dest_addr, word);
  }
}
#endif

typedef struct {
  uint64_t (*hint_storew)(void* ctx, uint64_t dest_addr);
  uint64_t (*hint_buffer)(void* ctx, uint64_t dest_addr, uint32_t num_words,
                          uint32_t word_index);
  void (*reveal)(void* ctx, uint64_t src_val, uint64_t ptr, uint32_t offset,
                 uint32_t width);
} Rv64IoHostCallbacks;

static thread_local Rv64IoHostCallbacks g_rv64io_callbacks;

void register_rv64io_callbacks(const Rv64IoHostCallbacks* cb) { g_rv64io_callbacks = *cb; }

void openvm_hint_storew(void* state, uint64_t dest_addr, uint32_t from_pc,
                        uint32_t from_timestamp, uint32_t mem_ptr_ptr,
                        uint32_t mem_ptr_prev_timestamp, uint32_t chip_idx) {
  uint64_t word = g_rv64io_callbacks.hint_storew(openvm_get_io_ctx(), dest_addr);
#ifdef OPENVM_TRACER_PREFLIGHT_H
  RvState* rv_state = (RvState*)state;
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
  rvr_hintstore_write_word(rv_state, dest_addr, word, emit_direct, vars);
#else
  wr_mem_u64_traced((RvState*)state, dest_addr, word);
#endif
}

void openvm_hint_buffer(void* state, uint64_t dest_addr, uint32_t num_words,
                        uint32_t from_pc, uint32_t from_timestamp,
                        uint32_t mem_ptr_ptr,
                        uint32_t mem_ptr_prev_timestamp,
                        uint32_t num_words_ptr,
                        uint32_t num_words_prev_timestamp,
                        uint32_t chip_idx) {
#ifdef OPENVM_TRACER_PREFLIGHT_H
  RvState* rv_state = (RvState*)state;
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
    uint64_t word =
        g_rv64io_callbacks.hint_buffer(openvm_get_io_ctx(), dest_addr, num_words, i);
#ifdef OPENVM_TRACER_PREFLIGHT_H
    rvr_hintstore_write_word(rv_state, dest_addr + i * WORD_SIZE, word,
                             emit_direct, vars == NULL ? NULL : &vars[i]);
#else
    wr_mem_u64_traced((RvState*)state, dest_addr + i * WORD_SIZE, word);
#endif
  }
}

void openvm_reveal(uint64_t src_val, uint64_t ptr, uint32_t offset,
                   uint32_t width) {
  g_rv64io_callbacks.reveal(openvm_get_io_ctx(), src_val, ptr, offset, width);
}
