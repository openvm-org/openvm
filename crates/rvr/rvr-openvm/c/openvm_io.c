/*
 * IO ctx slot. The host installs a pointer via `register_openvm_io_ctx`;
 * extensions read it back via `openvm_get_io_ctx`.
 */

#include "openvm_io.h"

/* NOT thread-safe: installs one process-global IO ctx pointer. */
static void* g_io_ctx;

void register_openvm_io_ctx(void* ctx) { g_io_ctx = ctx; }

void* openvm_get_io_ctx(void) { return g_io_ctx; }

/* NOT thread-safe: installs one process-global hint-stream writer. */
static void (*g_hint_stream_set_fn)(void* ctx, const uint8_t* data,
                                    uint32_t len);

void register_hint_stream_set_fn(void (*fn)(void*, const uint8_t*, uint32_t)) {
  g_hint_stream_set_fn = fn;
}

void ext_hint_stream_set(const uint8_t* data, uint32_t len) {
  g_hint_stream_set_fn(g_io_ctx, data, len);
}
