/*
 * Dispatch table and forwarding stubs for the Rv64I base IO operations: the
 * HINT_INPUT, PRINT_STR, and HINT_RANDOM phantoms.
 */

#include "rv64i_phantom_callbacks.h"

#include "openvm_io.h"

static thread_local Rv64IHostCallbacks g_rv64i_host_callbacks;

void register_rv64i_host_callbacks(const Rv64IHostCallbacks* cb) {
  g_rv64i_host_callbacks = *cb;
}

void openvm_hint_input(void) { g_rv64i_host_callbacks.hint_input(openvm_get_io_ctx()); }

bool openvm_print_str(uint64_t ptr, uint64_t len) {
  return g_rv64i_host_callbacks.print_str(openvm_get_io_ctx(), ptr, len);
}

bool openvm_hint_random(uint64_t num_words) {
  return g_rv64i_host_callbacks.hint_random(openvm_get_io_ctx(), num_words);
}
