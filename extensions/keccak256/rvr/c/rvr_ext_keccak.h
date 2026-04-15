#ifndef RVR_EXT_KECCAK_H
#define RVR_EXT_KECCAK_H

#include <stdint.h>

struct RvState;

/* Cold FFI entry points use the preserve_most calling convention so that
 * hot block callers keep their live state in registers across the call.
 * See crates/rvr-openvm/src/emit/project.rs for the matching preserve_none
 * caller convention. */

extern __attribute__((preserve_most)) void rvr_ext_keccakf(RvState* state, uint32_t buffer_ptr,
                                                           uint32_t op_chip_idx,
                                                           uint32_t perm_chip_idx);

extern __attribute__((preserve_most)) void rvr_ext_xorin(RvState* state, uint32_t buffer_ptr,
                                                         uint32_t input_ptr, uint32_t len,
                                                         uint32_t chip_idx);

#endif /* RVR_EXT_KECCAK_H */
