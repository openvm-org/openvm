/*
 * ext_deferral_lookup.c — Host callback bridge for the deferral extension.
 *
 * Thin stubs that forward ext_deferral_*_lookup calls through function
 * pointers to Rust handlers registered via register_deferral_callbacks().
 */

#include <stdint.h>

#include "openvm_io.h"

typedef struct {
  void* ctx;
  int (*call_lookup)(void* d_ctx, void* io_ctx, uint32_t def_idx, const uint8_t* input_commit,
                     uint8_t* output_key_out);
  int (*output_lookup)(void* d_ctx, void* io_ctx, uint32_t def_idx, const uint8_t* output_commit,
                       uint8_t* output_raw_out, uint32_t expected_len);
} DeferralHostCallbacks;

/* NOT thread-safe: installs one process-global deferral callback set. */
static DeferralHostCallbacks g_deferral;

void register_deferral_callbacks(const DeferralHostCallbacks* cb) { g_deferral = *cb; }

int ext_deferral_call_lookup(uint32_t def_idx, const uint8_t* input_commit,
                             uint8_t* output_key_out) {
  return g_deferral.call_lookup(g_deferral.ctx, openvm_get_io_ctx(), def_idx, input_commit,
                                output_key_out);
}

int ext_deferral_output_lookup(uint32_t def_idx, const uint8_t* output_commit,
                               uint8_t* output_raw_out, uint32_t expected_len) {
  return g_deferral.output_lookup(g_deferral.ctx, openvm_get_io_ctx(), def_idx, output_commit,
                                  output_raw_out, expected_len);
}
