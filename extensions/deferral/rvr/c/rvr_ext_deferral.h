#ifndef RVR_EXT_DEFERRAL_H
#define RVR_EXT_DEFERRAL_H

#include <stdbool.h>
#include <stdint.h>

typedef struct RvState RvState;

typedef struct DeferralHostCallbacks {
  void* ctx;
  bool (*call_lookup)(void* d_ctx, void* io_ctx, uint32_t def_idx,
                      const uint8_t* input_commit, uint8_t* output_key_out,
                      uint64_t* accumulators_out);
  bool (*output_lookup)(void* d_ctx, void* io_ctx, uint32_t def_idx,
                        const uint8_t* output_commit, uint8_t* output_raw_out,
                        uint32_t expected_len);
} DeferralHostCallbacks;

void register_deferral_callbacks(const DeferralHostCallbacks* cb);

/* Deferral CALL extension entry point (defined in rvr_ext_deferral.c). */
extern bool rvr_ext_deferral_call(RvState* state, uint64_t output_ptr,
                                  uint64_t input_ptr, uint32_t def_idx,
                                  uint64_t* replay_out);

/* Deferral OUTPUT extension entry point (defined in rvr_ext_deferral.c). */
extern bool rvr_ext_deferral_output(RvState* state, uint64_t output_ptr,
                                    uint64_t input_ptr, uint32_t def_idx,
                                    uint32_t* num_rows_out);

#endif /* RVR_EXT_DEFERRAL_H */
