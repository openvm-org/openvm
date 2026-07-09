/*
 * Dispatch table and forwarding stubs for the Rv64Io operations: the
 * hint-store consumers (HINT_STOREW, HINT_BUFFER) and REVEAL.
 */

#include "rv64io_callbacks.h"

#include "openvm.h"
#include "openvm_io.h"

static thread_local Rv64IoHostCallbacks g_rv64io_host_callbacks;

void register_rv64io_host_callbacks(const Rv64IoHostCallbacks* cb) {
  g_rv64io_host_callbacks = *cb;
}

bool openvm_hint_storew(void* state, uint64_t dest_addr) {
  bool ok =
      g_rv64io_host_callbacks.hint_storew(openvm_get_io_ctx(), dest_addr);
  if (likely(ok)) {
    trace_wr_mem_u64((RvState*)state, dest_addr,
                     read_mem_u64(((RvState*)state)->memory, dest_addr));
  }
  return ok;
}

bool openvm_hint_buffer(void* state, uint64_t dest_addr, uint16_t num_words) {
  bool ok = g_rv64io_host_callbacks.hint_buffer(openvm_get_io_ctx(), dest_addr,
                                                num_words);
  if (likely(ok)) {
    for (uint32_t i = 0; i < num_words; i++) {
      if (i != 0) {
        trace_timestamp((RvState*)state);
        trace_timestamp((RvState*)state);
      }
      uint64_t addr = dest_addr + i * WORD_SIZE;
      trace_wr_mem_u64((RvState*)state, addr,
                       read_mem_u64(((RvState*)state)->memory, addr));
    }
  }
  return ok;
}

bool openvm_reveal(uint64_t src_val, uint64_t addr, uint8_t width) {
  return g_rv64io_host_callbacks.reveal(openvm_get_io_ctx(), src_val, addr, width);
}
