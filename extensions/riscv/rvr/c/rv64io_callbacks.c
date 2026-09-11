/*
 * Dispatch table and forwarding stubs for the Rv64Io operations: the
 * hint-store consumers (HINT_STOREW, HINT_BUFFER) and REVEAL.
 */

/* openvm.h exposes the mode-specific static trace helpers. Clang cannot prove
 * the fixed-capacity bounds maintained by the Rust-owned preflight buffers. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#include "openvm.h"
#pragma clang diagnostic pop
#include "rv64io_callbacks.h"

static thread_local Rv64IoHostCallbacks g_rv64io_host_callbacks;

void register_rv64io_host_callbacks(const Rv64IoHostCallbacks* cb) {
  g_rv64io_host_callbacks = *cb;
}

bool openvm_hint_prepare(uint64_t dest_addr, uint32_t num_words) {
  return g_rv64io_host_callbacks.hint_prepare(openvm_get_io_ctx(), dest_addr,
                                               num_words);
}

void openvm_hint_read_words(uint64_t* words, uint32_t num_words) {
  g_rv64io_host_callbacks.hint_read_words(openvm_get_io_ctx(), words, num_words);
}

bool openvm_hint_storew(uint64_t dest_addr) {
  return g_rv64io_host_callbacks.hint_storew(openvm_get_io_ctx(), dest_addr);
}

bool openvm_hint_buffer(uint64_t dest_addr, uint32_t num_words) {
  return g_rv64io_host_callbacks.hint_buffer(openvm_get_io_ctx(), dest_addr,
                                             num_words);
}

bool openvm_reveal(RvState* state, uint64_t src_val, uint64_t base_addr,
                   uint64_t effective_addr) {
  void* ctx = openvm_get_io_ctx();
  if (unlikely(!g_rv64io_host_callbacks.reveal(
          ctx, src_val, base_addr, effective_addr))) {
    return false;
  }

  trace_write_public_values_u64(state, (uint32_t)effective_addr);
  return true;
}
