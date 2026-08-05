/*
 * Dispatch table and forwarding stubs for the RV64 HINT_INPUT, PRINT_STR, and
 * HINT_RANDOM phantoms.
 */

#include "riscv_phantom_callbacks.h"

#include "openvm_io.h"

static thread_local RiscvPhantomHostCallbacks g_riscv_phantom_host_callbacks;

void register_riscv_phantom_host_callbacks(const RiscvPhantomHostCallbacks* cb) {
  g_riscv_phantom_host_callbacks = *cb;
}

bool openvm_hint_input(void) {
  return g_riscv_phantom_host_callbacks.hint_input(openvm_get_io_ctx());
}

bool openvm_print_str(uint64_t ptr, uint64_t len) {
  return g_riscv_phantom_host_callbacks.print_str(openvm_get_io_ctx(), ptr, len);
}

bool openvm_hint_random(uint64_t num_words) {
  return g_riscv_phantom_host_callbacks.hint_random(openvm_get_io_ctx(),
                                                   num_words);
}
