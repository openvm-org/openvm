/* Keccak extension FFI: traced memory ops + dispatch into the keccak-ffi
 * staticlib's `rvr_keccak_f1600` (asm backend selected at FFI-crate build
 * time). Compiled into the generated native project so clang can inline the
 * tracer helpers. */

#include "openvm.h"

static constexpr uint32_t KECCAK_WIDTH_BYTES = 200;
static constexpr uint32_t KECCAK_WIDTH_WORDS = KECCAK_WIDTH_BYTES / WORD_SIZE;
static constexpr uint64_t KECCAK_RATE_BYTES = 136;
static constexpr uint32_t KECCAK_NUM_ROUNDS = 24;

extern void rvr_keccak_f1600(uint64_t state[static KECCAK_WIDTH_WORDS]);

__attribute__((preserve_most)) void rvr_ext_keccakf(RvState* restrict state,
                                                    uint64_t buffer_ptr,
                                                    uint32_t /* op_chip_idx */,
                                                    uint32_t perm_chip_idx) {
  uint64_t st[KECCAK_WIDTH_WORDS];
  static_assert(sizeof(st) == KECCAK_WIDTH_BYTES, "keccak state size mismatch");

  rd_mem_u64_range_traced(state, buffer_ptr, st, KECCAK_WIDTH_WORDS);
  rvr_keccak_f1600(st);
  wr_mem_u64_range_traced(state, buffer_ptr, st, KECCAK_WIDTH_WORDS);

  /* KeccakfOp cost (1 row) is covered by the per-block chip update emitted
   * at block entry; only the additional KeccakfPerm rows (24 per
   * permutation) need trace_chip. */
  trace_chip(state, perm_chip_idx, KECCAK_NUM_ROUNDS);
}

__attribute__((preserve_most)) void rvr_ext_xorin(RvState* restrict state,
                                                  uint64_t buffer_ptr,
                                                  uint64_t input_ptr,
                                                  uint64_t len,
                                                  uint32_t /* chip_idx */) {
  assume(len <= KECCAK_RATE_BYTES);
  /* The bound limits this to 17; uint32_t matches the traced range-helper ABI. */
  uint32_t num_words = (uint32_t)((len + WORD_SIZE - 1) / WORD_SIZE);
  if (unlikely(num_words == 0)) {
    return;
  }

  uint64_t buffer[KECCAK_WIDTH_WORDS];
  uint64_t input[KECCAK_WIDTH_WORDS];
  assume(num_words <= KECCAK_WIDTH_WORDS);

  rd_mem_u64_range_traced(state, buffer_ptr, buffer, num_words);
  rd_mem_u64_range_traced(state, input_ptr, input, num_words);
  /* Both local arrays have KECCAK_WIDTH_WORDS elements and num_words is
   * bounded above; Clang's C unsafe-buffer heuristic cannot express that. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
  for (uint32_t i = 0; i < num_words; i++) {
    buffer[i] ^= input[i];
  }
#pragma clang diagnostic pop
  wr_mem_u64_range_traced(state, buffer_ptr, buffer, num_words);
}
