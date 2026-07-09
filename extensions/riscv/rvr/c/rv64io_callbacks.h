#ifndef RV64IO_CALLBACKS_H
#define RV64IO_CALLBACKS_H

#include <stdint.h>

typedef struct {
  bool (*hint_storew)(void* ctx, uint64_t dest_addr);
  bool (*hint_buffer)(void* ctx, uint64_t dest_addr, uint16_t num_words);
  bool (*reveal)(void* ctx, uint64_t src_val, uint64_t addr, uint8_t width);
} Rv64IoHostCallbacks;

/* Callbacks and forwarding stubs return false for invalid guest operands. */

void register_rv64io_host_callbacks(const Rv64IoHostCallbacks* cb);

/* Forwarding stubs owned by the RISC-V IO (Rv64Io) extension: the hint-store
 * consumers (HINT_STOREW, HINT_BUFFER) and public-values stores routed through
 * openvm_reveal. Backed by a thread-local dispatch table installed at execution
 * time by `Rv64IoRuntimeHooks`. */
bool openvm_hint_storew(void* state, uint64_t dest_addr);
bool openvm_hint_buffer(void* state, uint64_t dest_addr, uint16_t num_words);
bool openvm_reveal(uint64_t src_val, uint64_t addr, uint8_t width);

#endif /* RV64IO_CALLBACKS_H */
