#ifndef RVR_EXT_SHA2_H
#define RVR_EXT_SHA2_H

#include <stdint.h>

struct RvState;

/* SHA-256 compress extension FFI entry point (implemented in Rust). */
extern void rvr_ext_sha256(RvState* state, uint32_t dst_ptr, uint32_t state_ptr, uint32_t input_ptr,
                           uint32_t main_chip_idx, uint32_t block_hasher_chip_idx);

/* SHA-512 compress extension FFI entry point (implemented in Rust). */
extern void rvr_ext_sha512(RvState* state, uint32_t dst_ptr, uint32_t state_ptr, uint32_t input_ptr,
                           uint32_t main_chip_idx, uint32_t block_hasher_chip_idx);

#endif /* RVR_EXT_SHA2_H */
