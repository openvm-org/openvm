#ifndef RV64I_PHANTOM_CALLBACKS_H
#define RV64I_PHANTOM_CALLBACKS_H

#include <stdint.h>

/* Forwarding stubs owned by the RISC-V base (Rv64I) extension: the IO phantoms
 * (HINT_INPUT, PRINT_STR, HINT_RANDOM). Backed by a thread-local dispatch table
 * installed at execution time by `Rv64IRuntimeHooks`. */
void openvm_hint_input(void);
void openvm_print_str(uint64_t ptr, uint32_t len);
void openvm_hint_random(uint32_t num_words);

#endif /* RV64I_PHANTOM_CALLBACKS_H */
