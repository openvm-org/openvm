/* Keccak extension FFI: memory access + dispatch into the keccak-ffi
 * staticlib's `rvr_keccak_f1600` (asm backend selected at FFI-crate build
 * time). Compiled into the generated native project so clang can inline the
 * tracer helpers. */

#include "openvm.h"
#include "rvr_ext_keccak.h"

static constexpr uint32_t KECCAK_WIDTH_BYTES = 200;
static constexpr uint32_t KECCAK_WIDTH_WORDS = KECCAK_WIDTH_BYTES / WORD_SIZE;
static constexpr uint64_t KECCAK_RATE_BYTES = 136;

extern void rvr_keccak_f1600(uint64_t state[static const KECCAK_WIDTH_WORDS]);

__attribute__((preserve_most)) void rvr_ext_keccakf(RvState* restrict state,
                                                    uint64_t buffer_ptr) {
  uint64_t st[KECCAK_WIDTH_WORDS];
  static_assert(sizeof(st) == KECCAK_WIDTH_BYTES, "keccak state size mismatch");

  read_mem_u64_range(state, buffer_ptr, st, KECCAK_WIDTH_WORDS);
  rvr_keccak_f1600(st);
  write_mem_u64_range(state, buffer_ptr, st, KECCAK_WIDTH_WORDS);
}

__attribute__((preserve_most)) bool rvr_ext_xorin(RvState* restrict state,
                                                  uint64_t buffer_ptr,
                                                  uint64_t input_ptr,
                                                  uint64_t len) {
  if (unlikely(len > KECCAK_RATE_BYTES)) {
    return false;
  }
  /* len is at most 136, so this is at most 17. uint32_t matches the range
   * helper. */
  uint32_t num_words = (uint32_t)((len + WORD_SIZE - 1) / WORD_SIZE);
  if (unlikely(num_words == 0)) {
    return true;
  }

  uint64_t buffer[KECCAK_WIDTH_WORDS];
  uint64_t input[KECCAK_WIDTH_WORDS];
  assume(num_words <= KECCAK_WIDTH_WORDS);

  read_mem_u64_range(state, buffer_ptr, buffer, num_words);
  read_mem_u64_range(state, input_ptr, input, num_words);
  /* Both arrays have KECCAK_WIDTH_WORDS elements, and the check above bounds
   * num_words. Clang cannot prove these bounds. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
  for (uint32_t i = 0; i < num_words; i++) {
    buffer[i] ^= input[i];
  }
#pragma clang diagnostic pop
  write_mem_u64_range(state, buffer_ptr, buffer, num_words);
  return true;
}
