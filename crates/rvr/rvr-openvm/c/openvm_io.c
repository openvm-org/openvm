/*
 * IO ctx slot. The host installs a pointer via `register_openvm_io_ctx`;
 * extensions read it back via `openvm_get_io_ctx`.
 */

#include "openvm_io.h"

static thread_local void* g_io_ctx;

void register_openvm_io_ctx(void* ctx) { g_io_ctx = ctx; }

void* openvm_get_io_ctx(void) { return g_io_ctx; }

static thread_local HintStreamSetFn g_hint_stream_set_fn;

void register_hint_stream_set_fn(HintStreamSetFn fn) {
  g_hint_stream_set_fn = fn;
}

void ext_hint_stream_set(const uint8_t* data, uint64_t len) {
  g_hint_stream_set_fn(g_io_ctx, data, len);
}
