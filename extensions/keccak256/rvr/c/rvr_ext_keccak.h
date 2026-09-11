#ifndef RVR_EXT_KECCAK_H
#define RVR_EXT_KECCAK_H

#include <stdint.h>

typedef struct RvState RvState;

/* Cold FFI entry points use the preserve_most calling convention so that
 * hot block callers keep their live state in registers across the call.
 * See crates/rvr-openvm/src/emit/project.rs for the matching preserve_none
 * caller convention. */

extern __attribute__((preserve_most)) void rvr_ext_keccakf(
    RvState* state, uint64_t buffer_ptr);

extern __attribute__((preserve_most)) bool rvr_ext_xorin(RvState* state,
                                                         uint64_t buffer_ptr,
                                                         uint64_t input_ptr,
                                                         uint64_t len);

#endif /* RVR_EXT_KECCAK_H */
