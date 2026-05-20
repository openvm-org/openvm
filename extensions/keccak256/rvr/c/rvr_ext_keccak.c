/* Keccak extension FFI: traced memory ops + dispatch into the keccak-ffi
 * staticlib's `rvr_keccak_f1600` (asm backend selected at FFI-crate build
 * time). Compiled into the generated native project so clang can inline the
 * tracer helpers. */

#include "openvm.h"
#include <string.h>

extern void rvr_keccak_f1600(uint64_t state[25]);

static constexpr uint32_t KECCAK_WIDTH_BYTES = 200;
static constexpr uint32_t KECCAK_WIDTH_WORDS = KECCAK_WIDTH_BYTES / WORD_SIZE;
static constexpr uint32_t KECCAK_NUM_ROUNDS  = 24;

__attribute__((preserve_most)) void rvr_ext_keccakf(RvState* restrict state, uint32_t buffer_ptr,
                                                    uint32_t _op_chip_idx,
                                                    uint32_t perm_chip_idx) {
    /* u64-aligned scratch; the u32 view is zero-copy on LE. */
    uint64_t st[KECCAK_WIDTH_WORDS / 2];
    static_assert(sizeof(st) == KECCAK_WIDTH_BYTES, "keccak state size mismatch");

    uint32_t words[KECCAK_WIDTH_WORDS];
    rd_mem_u32_range_traced(state, buffer_ptr, words, KECCAK_WIDTH_WORDS);
    memcpy(st, words, KECCAK_WIDTH_BYTES);

    rvr_keccak_f1600(st);

    memcpy(words, st, KECCAK_WIDTH_BYTES);
    wr_mem_u32_range_traced(state, buffer_ptr, words, KECCAK_WIDTH_WORDS);

    /* KeccakfOp cost (1 row) is covered by the per-block chip update emitted
     * at block entry; only the additional KeccakfPerm rows (24 per
     * permutation) need trace_chip. */
    trace_chip(state, perm_chip_idx, KECCAK_NUM_ROUNDS);
}

__attribute__((preserve_most)) void rvr_ext_xorin(RvState* restrict state, uint32_t buffer_ptr,
                                                  uint32_t input_ptr, uint32_t len,
                                                  uint32_t _chip_idx) {
    uint32_t num_words = (len + WORD_SIZE - 1) / WORD_SIZE;
    if (unlikely(num_words == 0)) {
        return;
    }

    uint32_t buffer[KECCAK_WIDTH_WORDS];
    uint32_t input[KECCAK_WIDTH_WORDS];
    assume(num_words <= KECCAK_WIDTH_WORDS);

    rd_mem_u32_range_traced(state, buffer_ptr, buffer, num_words);
    rd_mem_u32_range_traced(state, input_ptr, input, num_words);
    for (uint32_t i = 0; i < num_words; i++) {
        buffer[i] ^= input[i];
    }
    wr_mem_u32_range_traced(state, buffer_ptr, buffer, num_words);
}
