/*
 * Dispatch table and forwarding stubs for the Rv64Io operations: the
 * hint-store consumers (HINT_STOREW, HINT_BUFFER) and REVEAL.
 */

#include "rv64io_callbacks.h"

#include "openvm.h"
#include "openvm_io.h"

typedef struct {
  uint64_t (*hint_storew)(void* ctx, uint64_t dest_addr);
  uint64_t (*hint_buffer)(void* ctx, uint64_t dest_addr, uint32_t num_words,
                          uint32_t word_index);
  void (*reveal)(void* ctx, uint64_t src_val, uint64_t ptr, uint32_t offset,
                 uint32_t width);
} Rv64IoHostCallbacks;

static thread_local Rv64IoHostCallbacks g_rv64io_callbacks;

void register_rv64io_callbacks(const Rv64IoHostCallbacks* cb) { g_rv64io_callbacks = *cb; }

void openvm_hint_storew(void* state, uint64_t dest_addr) {
  uint64_t word = g_rv64io_callbacks.hint_storew(openvm_get_io_ctx(), dest_addr);
  wr_mem_u64_traced((RvState*)state, dest_addr, word);
}

void openvm_hint_buffer(void* state, uint64_t dest_addr, uint32_t num_words) {
  for (uint32_t i = 0; i < num_words; i++) {
    if (i != 0) {
      trace_timestamp((RvState*)state);
      trace_timestamp((RvState*)state);
    }
    uint64_t word =
        g_rv64io_callbacks.hint_buffer(openvm_get_io_ctx(), dest_addr, num_words, i);
    wr_mem_u64_traced((RvState*)state, dest_addr + i * WORD_SIZE, word);
  }
}

void openvm_reveal(uint64_t src_val, uint64_t ptr, uint32_t offset,
                   uint32_t width) {
  g_rv64io_callbacks.reveal(openvm_get_io_ctx(), src_val, ptr, offset, width);
}
