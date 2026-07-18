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

static thread_local DeferralHostCallbacks g_deferral;

void register_deferral_callbacks(const DeferralHostCallbacks* cb) {
  g_deferral = *cb;
}

/* Deferral CALL: read input_commit, look up output_key, write it. */
void rvr_ext_deferral_call(RvState* restrict state, uint64_t output_ptr,
                           uint64_t input_ptr, uint32_t def_idx) {
  /* Read input_commit (COMMIT_WORDS words) from guest memory. */
  uint64_t commit_words[COMMIT_WORDS];
  rd_mem_u64_range_traced(state, input_ptr, commit_words, COMMIT_WORDS);

  /* Trace DEFERRAL_AS reads (old input_acc + old output_acc). Slot offsets
   * are in field-element units.
   * TODO: deferral address space elements are field elements, not u32.
   * Page tracking currently assumes u32 cells — verify that the page and
   * chunk geometry is correct for field-element-typed cells. */
  uint64_t input_acc_ptr = 2u * def_idx * DEFERRAL_DIGEST_SIZE;
  uint64_t output_acc_ptr = input_acc_ptr + DEFERRAL_DIGEST_SIZE;
  trace_mem_access_u64_range(state, input_acc_ptr, DIGEST_MEMORY_OPS,
                             AS_DEFERRAL);
  trace_mem_access_u64_range(state, output_acc_ptr, DIGEST_MEMORY_OPS,
                             AS_DEFERRAL);

  /* Look up output_key + update accumulators (Rust side, on byte buffers). */
  uint64_t key_words[OUTPUT_KEY_WORDS];
  g_deferral.call_lookup(g_deferral.ctx, openvm_get_io_ctx(), def_idx,
                         (const uint8_t*)commit_words, (uint8_t*)key_words);

  /* Write output_key (OUTPUT_KEY_WORDS words) to guest memory. */
  wr_mem_u64_range_traced(state, output_ptr, key_words, OUTPUT_KEY_WORDS);

  /* Trace DEFERRAL_AS writes (new input_acc + new output_acc). */
  trace_mem_access_u64_range(state, input_acc_ptr, DIGEST_MEMORY_OPS,
                             AS_DEFERRAL);
  trace_mem_access_u64_range(state, output_acc_ptr, DIGEST_MEMORY_OPS,
                             AS_DEFERRAL);
}

/* Deferral OUTPUT: read output_key, look up raw output, write it. */
uint32_t rvr_ext_deferral_output(RvState* restrict state, uint64_t output_ptr,
                                 uint64_t input_ptr, uint32_t def_idx) {
  /* Read output_key (OUTPUT_KEY_WORDS words) from guest memory. */
  uint64_t key_words[OUTPUT_KEY_WORDS];
  rd_mem_u64_range_traced(state, input_ptr, key_words, OUTPUT_KEY_WORDS);

  /* output_len is the u64 LE stored right after the commit. */
  /* The output AIR encodes the length in its low four bytes. Ignore the
   * unconstrained upper bytes exactly as the reference executor does. */
  uint32_t output_len = (uint32_t)key_words[COMMIT_WORDS];

  /* Look up the raw output into a heap buffer sized to output_len. The
   * output_commit is the leading DEFERRAL_COMMIT_NUM_BYTES of output_key. */
  uint8_t* output_raw = (uint8_t*)malloc((size_t)output_len);
  if (!output_raw && output_len > 0) abort();
  g_deferral.output_lookup(g_deferral.ctx, openvm_get_io_ctx(), def_idx,
                           (const uint8_t*)key_words, output_raw,
                           (uint32_t)output_len);

  /* Write raw output to guest memory in DEFERRAL_DIGEST_SIZE-byte rows. Each
   * row is an independent batched write (the trace API is per-row). */
  uint32_t num_data_rows = output_len / DEFERRAL_DIGEST_SIZE;
  uint64_t row_words[DIGEST_MEMORY_OPS];
  for (uint64_t row_idx = 0; row_idx < num_data_rows; row_idx++) {
    uint64_t row_byte_base = row_idx * DEFERRAL_DIGEST_SIZE;
    /* output_raw has output_len bytes and row_idx is bounded by
     * output_len / DEFERRAL_DIGEST_SIZE. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
    memcpy(row_words, output_raw + row_byte_base, DEFERRAL_DIGEST_SIZE);
#pragma clang diagnostic pop
    wr_mem_u64_range_traced(state, output_ptr + row_byte_base, row_words,
                            DIGEST_MEMORY_OPS);
  }
  free(output_raw);

  uint32_t num_rows = num_data_rows + 1;
  return num_rows;
}
