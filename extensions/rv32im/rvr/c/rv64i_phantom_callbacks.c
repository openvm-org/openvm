/*
 * Dispatch table and forwarding stubs for the Rv64I base IO operations: the
 * HINT_INPUT, PRINT_STR, HINT_RANDOM phantoms and the extension hint-stream
 * setter.
 */

#include "rv64i_phantom_callbacks.h"

#include "openvm_io.h"

typedef struct {
  void (*hint_input)(void* ctx);
  void (*print_str)(void* ctx, uint32_t ptr, uint32_t len);
  void (*hint_random)(void* ctx, uint32_t num_words);
  void (*hint_stream_set)(void* ctx, const uint8_t* data, uint32_t len);
} Rv64IPhantomCallbacks;

/* NOT thread-safe: installs one process-global callback table. */
static Rv64IPhantomCallbacks g_rv64i_phantom_callbacks;

void register_rv64i_phantom_callbacks(const Rv64IPhantomCallbacks* cb) {
  g_rv64i_phantom_callbacks = *cb;
}

void openvm_hint_input(void) { g_rv64i_phantom_callbacks.hint_input(openvm_get_io_ctx()); }

void openvm_print_str(uint32_t ptr, uint32_t len) {
  g_rv64i_phantom_callbacks.print_str(openvm_get_io_ctx(), ptr, len);
}

void openvm_hint_random(uint32_t num_words) {
  g_rv64i_phantom_callbacks.hint_random(openvm_get_io_ctx(), num_words);
}

void ext_hint_stream_set(const uint8_t* data, uint32_t len) {
  g_rv64i_phantom_callbacks.hint_stream_set(openvm_get_io_ctx(), data, len);
}
