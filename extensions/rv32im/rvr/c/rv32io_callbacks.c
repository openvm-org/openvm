/*
 * Dispatch table and forwarding stubs for the Rv32Io operations: the
 * hint-store consumers (HINT_STOREW, HINT_BUFFER) and REVEAL.
 */

#include "rv32io_callbacks.h"

#include "openvm_io.h"

typedef struct {
  void (*hint_storew)(void* ctx, uint32_t dest_addr);
  void (*hint_buffer)(void* ctx, uint32_t dest_addr, uint32_t num_words);
  void (*reveal)(void* ctx, uint32_t src_val, uint32_t ptr, uint32_t offset);
} Rv32IoHostCallbacks;

/* NOT thread-safe: installs one process-global callback table. */
static Rv32IoHostCallbacks g_rv32io_callbacks;

void register_rv32io_callbacks(const Rv32IoHostCallbacks* cb) { g_rv32io_callbacks = *cb; }

void openvm_hint_storew(uint32_t dest_addr) {
  g_rv32io_callbacks.hint_storew(openvm_get_io_ctx(), dest_addr);
}

void openvm_hint_buffer(uint32_t dest_addr, uint32_t num_words) {
  g_rv32io_callbacks.hint_buffer(openvm_get_io_ctx(), dest_addr, num_words);
}

void openvm_reveal(uint32_t src_val, uint32_t ptr, uint32_t offset) {
  g_rv32io_callbacks.reveal(openvm_get_io_ctx(), src_val, ptr, offset);
}
