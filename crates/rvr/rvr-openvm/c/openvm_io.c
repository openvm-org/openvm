/*
 * openvm_io.c — Host callback bridge for rvr
 *
 * Thin stubs that forward openvm_* calls through function pointers
 * to Rust callback implementations registered via register_openvm_callbacks().
 */

#include "openvm_io.h"

/* ── Callback struct ──────────────────────────────────────────────── */

typedef struct {
  void* ctx;
  void (*hint_input)(void* ctx);
  void (*print_str)(void* ctx, uint32_t ptr, uint32_t len);
  void (*hint_random)(void* ctx, uint32_t num_words);
  void (*hint_storew)(void* ctx, uint32_t dest_addr);
  void (*hint_buffer)(void* ctx, uint32_t dest_addr, uint32_t num_words);
  void (*reveal)(void* ctx, uint32_t src_val, uint32_t ptr, uint32_t offset);
  void (*hint_stream_set)(void* ctx, const uint8_t* data, uint32_t len);
} OpenVmHostCallbacks;

/* NOT thread-safe: installs one process-global host callback set. */
static OpenVmHostCallbacks g_host;

void register_openvm_callbacks(const OpenVmHostCallbacks* cb) { g_host = *cb; }

void* openvm_get_io_ctx(void) { return g_host.ctx; }

/* ── Forwarding stubs ─────────────────────────────────────────────── */

void openvm_hint_input(void) { g_host.hint_input(g_host.ctx); }

void openvm_print_str(uint32_t ptr, uint32_t len) { g_host.print_str(g_host.ctx, ptr, len); }

void openvm_hint_random(uint32_t num_words) { g_host.hint_random(g_host.ctx, num_words); }

void openvm_hint_storew(uint32_t dest_addr) { g_host.hint_storew(g_host.ctx, dest_addr); }

void openvm_hint_buffer(uint32_t dest_addr, uint32_t num_words) {
  g_host.hint_buffer(g_host.ctx, dest_addr, num_words);
}

void openvm_reveal(uint32_t src_val, uint32_t ptr, uint32_t offset) {
  g_host.reveal(g_host.ctx, src_val, ptr, offset);
}

void ext_hint_stream_set(const uint8_t* data, uint32_t len) {
  g_host.hint_stream_set(g_host.ctx, data, len);
}
