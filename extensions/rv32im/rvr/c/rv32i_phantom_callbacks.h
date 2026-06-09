#ifndef RV32I_PHANTOM_CALLBACKS_H
#define RV32I_PHANTOM_CALLBACKS_H

#include <stdint.h>

/* Forwarding stubs owned by the rv32im base (Rv32I) extension: the IO phantoms
 * (HINT_INPUT, PRINT_STR, HINT_RANDOM) and the extension hint-stream setter.
 * Backed by a dispatch table installed by
 * `Rv32IExtension::register_host_callbacks`. */
void openvm_hint_input(void);
void openvm_print_str(uint32_t ptr, uint32_t len);
void openvm_hint_random(uint32_t num_words);

/* Extension hint stream access (called by extension FFI staticlibs). */
void ext_hint_stream_set(const uint8_t* data, uint32_t len);

#endif /* RV32I_PHANTOM_CALLBACKS_H */
