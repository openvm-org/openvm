/*
 * Dispatch table and forwarding stubs for the Rv64Io operations: the
 * hint-store consumers (HINT_STOREW, HINT_BUFFER) and public-values stores.
 */

#include "rv64io_callbacks.h"

#include "openvm_io.h"

typedef struct {
  void (*hint_storew)(void* ctx, uint32_t dest_addr);
  void (*hint_buffer)(void* ctx, uint32_t dest_addr, uint32_t num_words);
  void (*reveal)(void* ctx, uint64_t src_val, uint32_t ptr, uint32_t offset, uint32_t width);
} Rv64IoHostCallbacks;

/* NOT thread-safe: installs one process-global callback table. */
static Rv64IoHostCallbacks g_rv64io_callbacks;

void register_rv64io_callbacks(const Rv64IoHostCallbacks* cb) { g_rv64io_callbacks = *cb; }

void openvm_hint_storew(uint32_t dest_addr) {
  g_rv64io_callbacks.hint_storew(openvm_get_io_ctx(), dest_addr);
}

void openvm_hint_buffer(uint32_t dest_addr, uint32_t num_words) {
  g_rv64io_callbacks.hint_buffer(openvm_get_io_ctx(), dest_addr, num_words);
}

void openvm_reveal(uint64_t src_val, uint32_t ptr, uint32_t offset, uint32_t width) {
  g_rv64io_callbacks.reveal(openvm_get_io_ctx(), src_val, ptr, offset, width);
}
