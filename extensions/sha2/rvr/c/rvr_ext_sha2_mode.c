#include "openvm.h"
#include "rvr_ext_sha2.h"

#include <string.h>

static constexpr uint32_t RVR_SHA256_STATE_WORDS = 4u;
static constexpr uint32_t RVR_SHA256_BLOCK_WORDS = 8u;

extern void rvr_ext_sha256_fallback(RvState* state, uint64_t dst_ptr,
                                    uint64_t state_ptr, uint64_t input_ptr);
extern void rvr_sha256_compress_words(uint64_t state_words[RVR_SHA256_STATE_WORDS],
                                      const uint64_t block_words[RVR_SHA256_BLOCK_WORDS]);

#ifdef OPENVM_TRACER_PREFLIGHT_H
typedef struct RvrSha256ReadAux {
  uint32_t prev_timestamp;
} RvrSha256ReadAux;

typedef struct RvrSha256WriteAux {
  uint32_t prev_timestamp;
  uint8_t prev_data[WORD_SIZE];
} RvrSha256WriteAux;

typedef struct RvrSha256DirectRecord {
  uint32_t variant;
  uint32_t from_pc;
  uint32_t timestamp;
  uint32_t dst_reg_ptr;
  uint32_t state_reg_ptr;
  uint32_t input_reg_ptr;
  uint32_t dst_ptr;
  uint32_t state_ptr;
  uint32_t input_ptr;
  RvrSha256ReadAux register_reads_aux[3];
  uint8_t message_bytes[64];
  uint8_t prev_state[32];
  uint8_t new_state[32];
  RvrSha256ReadAux input_reads_aux[RVR_SHA256_BLOCK_WORDS];
  RvrSha256ReadAux state_reads_aux[RVR_SHA256_STATE_WORDS];
  RvrSha256WriteAux write_aux[RVR_SHA256_STATE_WORDS];
} RvrSha256DirectRecord;

static_assert(sizeof(RvrSha256ReadAux) == 4,
              "SHA-256 read aux ABI drift");
static_assert(sizeof(RvrSha256WriteAux) == 12,
              "SHA-256 write aux ABI drift");
static_assert(sizeof(RvrSha256DirectRecord) == 272,
              "SHA-256 direct record ABI drift");
#endif

uint8_t rvr_ext_sha2_is_preflight(void) {
#ifdef OPENVM_TRACER_PREFLIGHT_H
  return 1;
#else
  return 0;
#endif
}

__attribute__((preserve_most)) void rvr_ext_sha256(
    RvState* restrict state, uint64_t dst_ptr, uint64_t state_ptr,
    uint64_t input_ptr, uint32_t from_pc, uint32_t from_timestamp,
    uint32_t dst_reg_ptr, uint32_t state_reg_ptr, uint32_t input_reg_ptr,
    uint32_t dst_prev_timestamp, uint32_t state_prev_timestamp,
    uint32_t input_prev_timestamp, uint32_t main_chip_idx) {
#ifdef OPENVM_TRACER_PREFLIGHT_H
  if (likely(main_chip_idx != UINT32_MAX)) {
    uint64_t state_words[RVR_SHA256_STATE_WORDS];
    uint64_t block_words[RVR_SHA256_BLOCK_WORDS];
    read_mem_u64_range_raw(state, input_ptr, block_words,
                           RVR_SHA256_BLOCK_WORDS);
    read_mem_u64_range_raw(state, state_ptr, state_words,
                           RVR_SHA256_STATE_WORDS);

    RvrSha256DirectRecord* restrict record =
        (RvrSha256DirectRecord*)preflight_claim_record(state, main_chip_idx);
    if (likely(record != NULL)) {
      /* Every byte is filled below. Scalar stores avoid lowering a mostly
       * zero aggregate initializer into a 272-byte clear followed by the
       * actual record writes. */
      record->variant = 0u;
      record->from_pc = from_pc;
      record->timestamp = from_timestamp;
      record->dst_reg_ptr = dst_reg_ptr;
      record->state_reg_ptr = state_reg_ptr;
      record->input_reg_ptr = input_reg_ptr;
      record->dst_ptr = (uint32_t)dst_ptr;
      record->state_ptr = (uint32_t)state_ptr;
      record->input_ptr = (uint32_t)input_ptr;
      preflight_store_prev_timestamp(
          state->tracer, &record->register_reads_aux[0].prev_timestamp,
          dst_prev_timestamp);
      preflight_store_prev_timestamp(
          state->tracer, &record->register_reads_aux[1].prev_timestamp,
          state_prev_timestamp);
      preflight_store_prev_timestamp(
          state->tracer, &record->register_reads_aux[2].prev_timestamp,
          input_prev_timestamp);
      memcpy(record->message_bytes, block_words, sizeof(block_words));
      memcpy(record->prev_state, state_words, sizeof(state_words));
    }

    for (uint32_t i = 0; i < RVR_SHA256_BLOCK_WORDS; i++) {
      uint32_t prev_timestamp = preflight_append_memory(
          state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
          input_ptr + i * WORD_SIZE, WORD_SIZE, block_words[i], block_words[i]);
      if (likely(record != NULL)) {
        preflight_store_prev_timestamp(
            state->tracer, &record->input_reads_aux[i].prev_timestamp,
            prev_timestamp);
      }
    }
    for (uint32_t i = 0; i < RVR_SHA256_STATE_WORDS; i++) {
      uint32_t prev_timestamp = preflight_append_memory(
          state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
          state_ptr + i * WORD_SIZE, WORD_SIZE, state_words[i], state_words[i]);
      if (likely(record != NULL)) {
        preflight_store_prev_timestamp(
            state->tracer, &record->state_reads_aux[i].prev_timestamp,
            prev_timestamp);
      }
    }

    rvr_sha256_compress_words(state_words, block_words);
    if (likely(record != NULL)) {
      memcpy(record->new_state, state_words, sizeof(state_words));
    }
    for (uint32_t i = 0; i < RVR_SHA256_STATE_WORDS; i++) {
      uint64_t address = dst_ptr + i * WORD_SIZE;
      uint64_t prev_data =
          preflight_device_aux(state->tracer) &&
                  !preflight_device_aux_oracle(state->tracer)
              ? 0u
              : preflight_read_mem_block(state, address);
      uint32_t prev_timestamp = preflight_append_memory(
          state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, AS_MEMORY, address,
          WORD_SIZE, state_words[i], prev_data);
      if (likely(record != NULL)) {
        preflight_store_prev_timestamp(
            state->tracer, &record->write_aux[i].prev_timestamp,
            prev_timestamp);
        preflight_store_prev_value(state->tracer, record->write_aux[i].prev_data,
                                   prev_timestamp, prev_data);
      }
    }
    write_mem_u64_range_raw(state, dst_ptr, state_words,
                            RVR_SHA256_STATE_WORDS);
    return;
  }
#endif
  rvr_ext_sha256_fallback(state, dst_ptr, state_ptr, input_ptr);
}
