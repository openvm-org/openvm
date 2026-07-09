#ifndef RV64IO_CALLBACKS_H
#define RV64IO_CALLBACKS_H

#include <stdint.h>

/* Forwarding stubs owned by the RISC-V IO (Rv64Io) extension: the hint-store
 * consumers (HINT_STOREW, HINT_BUFFER) and REVEAL. Backed by a dispatch table
 * installed by `Rv64IoExtension::register_host_callbacks`. */
void openvm_hint_storew(void* state, uint64_t dest_addr);
void openvm_hint_buffer(void* state, uint64_t dest_addr, uint32_t num_words);
void openvm_reveal(uint64_t src_val, uint64_t ptr, uint32_t offset);

#endif /* RV64IO_CALLBACKS_H */
