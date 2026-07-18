#ifndef RVR_EXT_DEFERRAL_H
#define RVR_EXT_DEFERRAL_H

#include <stdint.h>

struct RvState;

typedef struct DeferralHostCallbacks {
  void* ctx;
  void (*call_lookup)(void* d_ctx, void* io_ctx, uint32_t def_idx,
                      const uint8_t* input_commit, uint8_t* output_key_out);
  void (*output_lookup)(void* d_ctx, void* io_ctx, uint32_t def_idx,
                        const uint8_t* output_commit, uint8_t* output_raw_out,
                        uint32_t expected_len);
} DeferralHostCallbacks;

void register_deferral_callbacks(const DeferralHostCallbacks* cb);

/* Deferral CALL extension entry point (defined in rvr_ext_deferral.c). */
extern void rvr_ext_deferral_call(RvState* state, uint64_t output_ptr,
                                  uint64_t input_ptr, uint32_t def_idx);

/* Deferral OUTPUT extension entry point (defined in rvr_ext_deferral.c). */
extern uint32_t rvr_ext_deferral_output(RvState* state, uint64_t output_ptr,
                                        uint64_t input_ptr, uint32_t def_idx);

#endif /* RVR_EXT_DEFERRAL_H */
