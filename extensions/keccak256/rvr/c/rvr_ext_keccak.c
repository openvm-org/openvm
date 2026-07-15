/* Keccak extension FFI: memory access + dispatch into the keccak-ffi
 * staticlib's `rvr_keccak_f1600` (asm backend selected at FFI-crate build
 * time). Compiled into the generated native project so clang can inline the
 * tracer helpers. */

#include "openvm.h"
#include "rvr_ext_keccak.h"

static constexpr uint32_t KECCAK_WIDTH_BYTES = 200;
static constexpr uint32_t KECCAK_WIDTH_WORDS = KECCAK_WIDTH_BYTES / WORD_SIZE;
static constexpr uint32_t KECCAK_RATE_BYTES = 136;
static constexpr uint32_t KECCAK_RATE_WORDS = KECCAK_RATE_BYTES / WORD_SIZE;

extern void rvr_keccak_f1600(uint64_t state[static const KECCAK_WIDTH_WORDS]);

#ifdef OPENVM_TRACER_PREFLIGHT_H
typedef struct KeccakfDirectRecord {
  uint32_t pc;
  uint32_t timestamp;
  uint32_t rd_ptr;
  uint32_t buffer_ptr;
  uint32_t rd_prev_timestamp;
  uint32_t buffer_word_prev_timestamps[KECCAK_WIDTH_WORDS];
  uint8_t preimage_buffer_bytes[KECCAK_WIDTH_BYTES];
} KeccakfDirectRecord;

static_assert(sizeof(KeccakfDirectRecord) == 320,
              "Keccakf direct record size drift");

typedef struct XorinWriteAux {
  uint32_t prev_timestamp;
  uint8_t prev_data[WORD_SIZE];
} XorinWriteAux;

typedef struct XorinDirectRecord {
  uint32_t from_pc;
  uint32_t timestamp;
  uint32_t rd_ptr;
  uint32_t rs1_ptr;
  uint32_t rs2_ptr;
  uint32_t buffer;
  uint32_t input;
  uint32_t len;
  uint8_t buffer_limbs[KECCAK_RATE_BYTES];
  uint8_t input_limbs[KECCAK_RATE_BYTES];
  uint32_t register_prev_timestamps[3];
  uint32_t input_read_prev_timestamps[KECCAK_RATE_WORDS];
  uint32_t buffer_read_prev_timestamps[KECCAK_RATE_WORDS];
  XorinWriteAux buffer_write_aux[KECCAK_RATE_WORDS];
} XorinDirectRecord;

static_assert(sizeof(XorinWriteAux) == 12, "Xorin write aux size drift");
static_assert(sizeof(XorinDirectRecord) == 656,
              "Xorin direct record size drift");
#endif

__attribute__((preserve_most)) void rvr_ext_keccakf(RvState* restrict state,
                                                    uint64_t buffer_ptr,
                                                    uint32_t from_pc,
                                                    uint32_t from_timestamp,
                                                    uint32_t rd_ptr,
                                                    uint32_t rd_prev_timestamp,
                                                    uint32_t op_chip_idx) {
  uint64_t st[KECCAK_WIDTH_WORDS];
  static_assert(sizeof(st) == KECCAK_WIDTH_BYTES, "keccak state size mismatch");

#ifdef OPENVM_TRACER_PREFLIGHT_H
  bool emit_direct = op_chip_idx != UINT32_MAX;
  uint64_t preimage[KECCAK_WIDTH_WORDS];
  KeccakfDirectRecord* restrict record = NULL;
  if (likely(emit_direct)) {
    /* KeccakfOp models the preimage as each write's previous data, not as
     * separate memory reads. Keep the load needed by the native permutation,
     * but do not consume preflight timestamps for it. */
    read_mem_u64_range_raw(state, buffer_ptr, preimage, KECCAK_WIDTH_WORDS);
    memcpy(st, preimage, sizeof(st));
    record =
        (KeccakfDirectRecord*)preflight_claim_record(state, op_chip_idx);
    if (likely(record != NULL)) {
      record->pc = from_pc;
      record->timestamp = from_timestamp;
      record->rd_ptr = rd_ptr;
      record->buffer_ptr = (uint32_t)buffer_ptr;
      preflight_store_prev_timestamp(state->tracer,
                                     &record->rd_prev_timestamp,
                                     rd_prev_timestamp);
      memcpy(record->preimage_buffer_bytes, preimage, sizeof(preimage));
    }
  } else {
    read_mem_u64_range_raw(state, buffer_ptr, st, KECCAK_WIDTH_WORDS);
  }
#else
  read_mem_u64_range_raw(state, buffer_ptr, st, KECCAK_WIDTH_WORDS);
#endif
  rvr_keccak_f1600(st);
#ifdef OPENVM_TRACER_PREFLIGHT_H
  if (likely(emit_direct)) {
    for (uint32_t i = 0; i < KECCAK_WIDTH_WORDS; i++) {
      uint32_t prev_timestamp = preflight_append_memory(
          state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, AS_MEMORY,
          buffer_ptr + i * WORD_SIZE, WORD_SIZE, st[i], preimage[i]);
      if (likely(record != NULL)) {
        preflight_store_prev_timestamp(
            state->tracer, &record->buffer_word_prev_timestamps[i],
            prev_timestamp);
      }
    }
    write_mem_u64_range_raw(state, buffer_ptr, st, KECCAK_WIDTH_WORDS);
  } else {
    write_mem_u64_range(state, buffer_ptr, st, KECCAK_WIDTH_WORDS);
  }
#else
  write_mem_u64_range(state, buffer_ptr, st, KECCAK_WIDTH_WORDS);
#endif
}

__attribute__((preserve_most)) bool rvr_ext_xorin(RvState* restrict state,
                                                  uint64_t buffer_ptr,
                                                  uint64_t input_ptr,
                                                  uint64_t len,
                                                  uint32_t from_pc,
                                                  uint32_t from_timestamp,
                                                  uint32_t rd_ptr,
                                                  uint32_t rs1_ptr,
                                                  uint32_t rs2_ptr,
                                                  uint32_t rd_prev_timestamp,
                                                  uint32_t rs1_prev_timestamp,
                                                  uint32_t rs2_prev_timestamp,
                                                  uint32_t chip_idx) {
  if (unlikely(len > KECCAK_RATE_BYTES)) {
    return false;
  }
  /* len is at most 136, so this is at most 17. uint32_t matches the range
   * helper. */
  uint32_t num_words = (uint32_t)((len + WORD_SIZE - 1) / WORD_SIZE);
  assume(num_words <= KECCAK_RATE_WORDS);

  uint64_t buffer[KECCAK_RATE_WORDS];
  uint64_t input[KECCAK_RATE_WORDS];

#ifdef OPENVM_TRACER_PREFLIGHT_H
  bool emit_direct = chip_idx != UINT32_MAX;
  XorinDirectRecord* restrict record = emit_direct
      ? (XorinDirectRecord*)preflight_claim_record(state, chip_idx)
      : NULL;
  if (likely(record != NULL)) {
    /* Compact fallback buffers may be recycled with dirty bytes. Direct-final
     * arena staging instead guarantees zero-filled slots, including recycled
     * pinned backings, so the measured path avoids a second 656-byte clear. */
    if ((state->tracer->chip_records[chip_idx].flags &
         PREFLIGHT_RECORD_DIRECT_FINAL) == 0u) {
      memset(record, 0, sizeof(*record));
    }
    record->from_pc = from_pc;
    record->timestamp = from_timestamp;
    record->rd_ptr = rd_ptr;
    record->rs1_ptr = rs1_ptr;
    record->rs2_ptr = rs2_ptr;
    record->buffer = (uint32_t)buffer_ptr;
    record->input = (uint32_t)input_ptr;
    record->len = (uint32_t)len;
    preflight_store_prev_timestamp(
        state->tracer, &record->register_prev_timestamps[0],
        rd_prev_timestamp);
    preflight_store_prev_timestamp(
        state->tracer, &record->register_prev_timestamps[1],
        rs1_prev_timestamp);
    preflight_store_prev_timestamp(
        state->tracer, &record->register_prev_timestamps[2],
        rs2_prev_timestamp);
  }
  if (likely(num_words != 0) && likely(emit_direct)) {
    read_mem_u64_range_raw(state, buffer_ptr, buffer, num_words);
    for (uint32_t i = 0; i < num_words; i++) {
      uint32_t prev_timestamp = preflight_append_memory(
          state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
          buffer_ptr + i * WORD_SIZE, WORD_SIZE, buffer[i], buffer[i]);
      if (likely(record != NULL)) {
        preflight_store_prev_timestamp(
            state->tracer, &record->buffer_read_prev_timestamps[i],
            prev_timestamp);
      }
    }
    read_mem_u64_range_raw(state, input_ptr, input, num_words);
    for (uint32_t i = 0; i < num_words; i++) {
      uint32_t prev_timestamp = preflight_append_memory(
          state->tracer, PREFLIGHT_MEMORY_KIND_READ, AS_MEMORY,
          input_ptr + i * WORD_SIZE, WORD_SIZE, input[i], input[i]);
      if (likely(record != NULL)) {
        preflight_store_prev_timestamp(
            state->tracer, &record->input_read_prev_timestamps[i],
            prev_timestamp);
      }
    }
    if (likely(record != NULL)) {
      memcpy(record->buffer_limbs, buffer, num_words * WORD_SIZE);
      memcpy(record->input_limbs, input, num_words * WORD_SIZE);
    }
  } else if (likely(num_words != 0)) {
    read_mem_u64_range(state, buffer_ptr, buffer, num_words);
    read_mem_u64_range(state, input_ptr, input, num_words);
  }
#else
  if (unlikely(num_words == 0)) {
    return true;
  }
  read_mem_u64_range(state, buffer_ptr, buffer, num_words);
  read_mem_u64_range(state, input_ptr, input, num_words);
#endif
  for (uint32_t i = 0; i < num_words; i++) {
    buffer[i] ^= input[i];
  }
#ifdef OPENVM_TRACER_PREFLIGHT_H
  if (likely(emit_direct)) {
    for (uint32_t i = 0; i < num_words; i++) {
      uint64_t address = buffer_ptr + i * WORD_SIZE;
      /* `buffer` was loaded before the XOR, so undoing the just-applied XOR
       * recovers the exact pre-write word without rereading guest memory. */
      uint64_t prev_data =
          preflight_device_aux(state->tracer) &&
                  !preflight_device_aux_oracle(state->tracer)
              ? 0u
              : buffer[i] ^ input[i];
      uint32_t prev_timestamp = preflight_append_memory(
          state->tracer, PREFLIGHT_MEMORY_KIND_WRITE, AS_MEMORY, address,
          WORD_SIZE, buffer[i], prev_data);
      if (likely(record != NULL)) {
        preflight_store_prev_timestamp(
            state->tracer, &record->buffer_write_aux[i].prev_timestamp,
            prev_timestamp);
        preflight_store_prev_value(
            state->tracer, record->buffer_write_aux[i].prev_data,
            prev_timestamp, prev_data);
      }
    }
    write_mem_u64_range_raw(state, buffer_ptr, buffer, num_words);
  } else {
    write_mem_u64_range(state, buffer_ptr, buffer, num_words);
  }
#else
  write_mem_u64_range(state, buffer_ptr, buffer, num_words);
#endif
  return true;
}
