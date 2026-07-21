/*
 * Dispatch table and forwarding stubs for the Rv64Io operations: the
 * hint-store consumers (HINT_STOREW, HINT_BUFFER) and public-values stores
 * routed through openvm_reveal.
 */

#include "rv64io_callbacks.h"

#include "openvm_io.h"

static thread_local Rv64IoHostCallbacks g_rv64io_host_callbacks;

void register_rv64io_host_callbacks(const Rv64IoHostCallbacks* cb) {
  g_rv64io_host_callbacks = *cb;
}

bool openvm_hint_storew(uint64_t dest_addr) {
  return g_rv64io_host_callbacks.hint_storew(openvm_get_io_ctx(), dest_addr);
}

bool openvm_hint_buffer(uint64_t dest_addr, uint16_t num_words) {
  return g_rv64io_host_callbacks.hint_buffer(openvm_get_io_ctx(), dest_addr, num_words);
}

bool openvm_reveal(uint64_t src_val, uint64_t addr, uint8_t width) {
  return g_rv64io_host_callbacks.reveal(openvm_get_io_ctx(), src_val, addr, width);
}
