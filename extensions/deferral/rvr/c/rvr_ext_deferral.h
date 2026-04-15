#ifndef RVR_EXT_DEFERRAL_H
#define RVR_EXT_DEFERRAL_H

#include <stdint.h>

struct RvState;

/* Deferral CALL extension FFI entry point (implemented in Rust). */
extern void rvr_ext_deferral_call(RvState* state, uint32_t output_ptr, uint32_t input_ptr,
                                  uint32_t def_idx, uint32_t poseidon2_chip_idx);

/* Deferral OUTPUT extension FFI entry point (implemented in Rust). */
extern void rvr_ext_deferral_output(RvState* state, uint32_t output_ptr, uint32_t input_ptr,
                                    uint32_t def_idx, uint32_t output_chip_idx,
                                    uint32_t poseidon2_chip_idx);

#endif /* RVR_EXT_DEFERRAL_H */
