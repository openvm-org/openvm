#ifndef RV64_PHANTOM_CALLBACKS_H
#define RV64_PHANTOM_CALLBACKS_H

#include <stdint.h>

typedef struct {
  bool (*hint_input)(void* ctx);
  bool (*print_str)(void* ctx, uint64_t ptr, uint64_t len);
  bool (*hint_random)(void* ctx, uint64_t num_words);
} Rv64PhantomHostCallbacks;

/* Status-returning callbacks and stubs return false for invalid guest operands. */

void register_rv64_phantom_host_callbacks(const Rv64PhantomHostCallbacks* cb);

/* Forwarding stubs for the RV64 HINT_INPUT, PRINT_STR, and HINT_RANDOM
 * phantoms. A runtime hook installs the thread-local callback table. */
bool openvm_hint_input(void);
bool openvm_print_str(uint64_t ptr, uint64_t len);
bool openvm_hint_random(uint64_t num_words);

#endif /* RV64_PHANTOM_CALLBACKS_H */
