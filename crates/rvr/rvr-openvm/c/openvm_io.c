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
  int (*deferral_call_lookup)(void* ctx, uint32_t def_idx, const uint8_t* input_commit,
                              uint8_t* output_key_out);
  int (*deferral_output_lookup)(void* ctx, const uint8_t* output_commit, uint8_t* output_raw_out,
                                uint32_t expected_len);
} OpenVmHostCallbacks;

/* NOT thread-safe: installs one process-global host callback set. */
static OpenVmHostCallbacks g_host;

void register_openvm_callbacks(const OpenVmHostCallbacks* cb) { g_host = *cb; }

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

int ext_deferral_call_lookup(uint32_t def_idx, const uint8_t* input_commit,
                             uint8_t* output_key_out) {
  return g_host.deferral_call_lookup(g_host.ctx, def_idx, input_commit, output_key_out);
}

int ext_deferral_output_lookup(const uint8_t* output_commit, uint8_t* output_raw_out,
                               uint32_t expected_len) {
  return g_host.deferral_output_lookup(g_host.ctx, output_commit, output_raw_out, expected_len);
}
