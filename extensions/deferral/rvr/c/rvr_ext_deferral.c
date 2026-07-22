/*
 * rvr_ext_deferral.c — Deferral extension entry points (CALL + OUTPUT).
 *
 * Compiled into the generated native project so the tracer helpers inline.
 * Each entry point owns the guest-memory I/O and tracing, then dispatches the
 * deferral-map lookup + accumulator update to a Rust host callback registered
 * via register_deferral_callbacks(). The callbacks work on plain byte buffers
 * (u8<->u32 packing stays on the Rust side) and never touch guest memory.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "openvm.h"
#include "rvr_ext_deferral.h"

/* Byte-layout constants derived from the generated digest and word sizes. */
// TODO(rvr): update to better formula/solution instead of defining field element size here
static constexpr uint32_t F_NUM_BYTES = 4;
static constexpr uint32_t DEFERRAL_COMMIT_NUM_BYTES =
    DEFERRAL_DIGEST_SIZE * F_NUM_BYTES;
/* output_key layout: commit ++ u64 output length. */
static constexpr uint32_t DEFERRAL_OUTPUT_KEY_BYTES =
    DEFERRAL_COMMIT_NUM_BYTES + (uint32_t)sizeof(uint64_t);
static constexpr uint32_t COMMIT_WORDS = DEFERRAL_COMMIT_NUM_BYTES / WORD_SIZE;
static constexpr uint32_t OUTPUT_KEY_WORDS =
    DEFERRAL_OUTPUT_KEY_BYTES / WORD_SIZE;
static constexpr uint32_t DIGEST_MEMORY_OPS = DEFERRAL_DIGEST_SIZE / WORD_SIZE;
static constexpr uint64_t MAIN_MEMORY_PAGE_BYTES =
    1ull << (TRACER_BYTE_SPACE_PTRS_PER_LEAF_BITS + TRACER_PAGE_BITS);
/* Leave room for an unaligned chunk to touch one extra page. */
static constexpr uint64_t OUTPUT_ROWS_PER_PAGE_BUFFER =
    ((uint64_t)TRACER_MEM_PAGE_BUF_CAP - 1ull) * MAIN_MEMORY_PAGE_BYTES /
    DEFERRAL_DIGEST_SIZE;
static_assert(OUTPUT_ROWS_PER_PAGE_BUFFER > 0u,
              "main-memory page buffer must hold one deferral output row");

static thread_local DeferralHostCallbacks g_deferral;

void register_deferral_callbacks(const DeferralHostCallbacks* cb) {
  g_deferral = *cb;
}

/* Deferral CALL: read input_commit, look up output_key, write it. */
void rvr_ext_deferral_call(RvState* restrict state, uint64_t output_ptr,
                           uint64_t input_ptr, uint32_t def_idx) {
  /* Read input_commit (COMMIT_WORDS words) from guest memory. */
  uint64_t commit_words[COMMIT_WORDS];
  read_mem_u64_range(state, input_ptr, commit_words, COMMIT_WORDS);

  /* Trace DEFERRAL_AS reads (old input_acc + old output_acc). Addresses and
   * span sizes are both expressed in field-element units. */
  uint64_t input_acc_ptr = 2u * def_idx * DEFERRAL_DIGEST_SIZE;
  uint64_t output_acc_ptr = input_acc_ptr + DEFERRAL_DIGEST_SIZE;
  trace_page_access_u64_range(state, input_acc_ptr, DIGEST_MEMORY_OPS,
                              AS_DEFERRAL);
  trace_page_access_u64_range(state, output_acc_ptr, DIGEST_MEMORY_OPS,
                              AS_DEFERRAL);

  /* Look up output_key + update accumulators (Rust side, on byte buffers). */
  uint64_t key_words[OUTPUT_KEY_WORDS];
  g_deferral.call_lookup(g_deferral.ctx, openvm_get_io_ctx(), def_idx,
                         (const uint8_t*)commit_words, (uint8_t*)key_words);

  /* Write output_key (OUTPUT_KEY_WORDS words) to guest memory. */
  write_mem_u64_range(state, output_ptr, key_words, OUTPUT_KEY_WORDS);

  /* Trace DEFERRAL_AS writes (new input_acc + new output_acc). */
  trace_page_access_u64_range(state, input_acc_ptr, DIGEST_MEMORY_OPS,
                              AS_DEFERRAL);
  trace_page_access_u64_range(state, output_acc_ptr, DIGEST_MEMORY_OPS,
                              AS_DEFERRAL);
}

/* Deferral OUTPUT: read output_key, look up raw output, write it. */
uint32_t rvr_ext_deferral_output(RvState* restrict state, uint64_t output_ptr,
                                 uint64_t input_ptr, uint32_t def_idx) {
  /* Read output_key (OUTPUT_KEY_WORDS words) from guest memory. */
  uint64_t key_words[OUTPUT_KEY_WORDS];
  read_mem_u64_range(state, input_ptr, key_words, OUTPUT_KEY_WORDS);

  /* output_len is the u64 LE stored right after the commit. */
  /* The AIR stores the length in the low four bytes. The upper bytes are
   * unconstrained, so ignore them as the reference executor does. */
  uint32_t output_len = (uint32_t)key_words[COMMIT_WORDS];

  /* Look up the raw output into a heap buffer sized to output_len. The
   * output_commit is the leading DEFERRAL_COMMIT_NUM_BYTES of output_key. */
  uint8_t empty_output;
  uint8_t* output_raw = &empty_output;
  if (output_len > 0) {
    output_raw = (uint8_t*)malloc((size_t)output_len);
    if (!output_raw) abort();
  }
  g_deferral.output_lookup(g_deferral.ctx, openvm_get_io_ctx(), def_idx,
                           (const uint8_t*)key_words, output_raw,
                           (uint32_t)output_len);

  /* The key read above is covered by the extension's fixed page bound. Drain
   * it before the variable-size output so each output chunk starts empty. */
  flush_main_memory_page_buffer(state);

  /* Write raw output to guest memory in DEFERRAL_DIGEST_SIZE-byte rows. Each
   * chunk spans at most TRACER_MEM_PAGE_BUF_CAP pages at arbitrary alignment,
   * then drains without creating an instruction or segment checkpoint. */
  uint32_t num_data_rows = output_len / DEFERRAL_DIGEST_SIZE;
  uint64_t row_words[DIGEST_MEMORY_OPS];
  for (uint64_t chunk_start = 0; chunk_start < num_data_rows;
       chunk_start += OUTPUT_ROWS_PER_PAGE_BUFFER) {
    uint64_t chunk_end = chunk_start + OUTPUT_ROWS_PER_PAGE_BUFFER;
    if (chunk_end > num_data_rows) {
      chunk_end = num_data_rows;
    }
    for (uint64_t row_idx = chunk_start; row_idx < chunk_end; row_idx++) {
      uint64_t row_byte_base = row_idx * DEFERRAL_DIGEST_SIZE;
      /* row_idx is less than output_len / DEFERRAL_DIGEST_SIZE, so this slice
       * stays within output_raw. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
      memcpy(row_words, output_raw + row_byte_base, DEFERRAL_DIGEST_SIZE);
#pragma clang diagnostic pop
      write_mem_u64_range(state, output_ptr + row_byte_base, row_words,
                          DIGEST_MEMORY_OPS);
    }
    flush_main_memory_page_buffer(state);
  }
  if (output_len > 0) free(output_raw);

  uint32_t num_rows = num_data_rows + 1;
  return num_rows;
}
