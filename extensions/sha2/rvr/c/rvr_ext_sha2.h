#ifndef RVR_EXT_SHA2_H
#define RVR_EXT_SHA2_H

#include <stdint.h>

struct RvState;

/* Compile-time-selected tracer mode, exposed to the Rust staticlib shim. */
extern uint8_t rvr_ext_sha2_is_preflight(void);

/* SHA-256 compress extension FFI entry point (implemented in Rust). */
extern void rvr_ext_sha256(RvState* state, uint64_t dst_ptr, uint64_t state_ptr, uint64_t input_ptr,
                           uint32_t main_chip_idx, uint32_t block_hasher_chip_idx);

/* SHA-512 compress extension FFI entry point (implemented in Rust). */
extern void rvr_ext_sha512(RvState* state, uint64_t dst_ptr, uint64_t state_ptr, uint64_t input_ptr,
                           uint32_t main_chip_idx, uint32_t block_hasher_chip_idx);

#endif /* RVR_EXT_SHA2_H */
