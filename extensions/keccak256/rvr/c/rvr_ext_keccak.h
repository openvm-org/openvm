#ifndef RVR_EXT_KECCAK_H
#define RVR_EXT_KECCAK_H

#include <stdint.h>

typedef struct RvState RvState;

/* Cold FFI entry points use the preserve_most calling convention so that
 * hot block callers keep their live state in registers across the call.
 * See crates/rvr-openvm/src/emit/project.rs for the matching preserve_none
 * caller convention. */

extern __attribute__((preserve_most)) void rvr_ext_keccakf(
    RvState* state, uint64_t buffer_ptr, uint32_t from_pc,
    uint32_t from_timestamp, uint32_t rd_ptr, uint32_t rd_prev_timestamp,
    uint32_t op_chip_idx);

extern __attribute__((preserve_most)) bool rvr_ext_xorin(RvState* state,
                                                         uint64_t buffer_ptr,
                                                         uint64_t input_ptr,
                                                         uint64_t len,
                                                         uint32_t from_pc,
                                                         uint32_t from_timestamp,
                                                         uint32_t rd_ptr,
                                                         uint32_t rs1_ptr,
                                                         uint32_t rs2_ptr,
                                                         uint32_t rd_prev_timestamp,
                                                         uint32_t rs1_prev_timestamp,
                                                         uint32_t rs2_prev_timestamp,
                                                         uint32_t chip_idx);

#endif /* RVR_EXT_KECCAK_H */
