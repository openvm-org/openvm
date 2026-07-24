#ifndef RVR_EXT_ECC_H
#define RVR_EXT_ECC_H

#include <stdint.h>

typedef struct RvState RvState;

/* k256 add_proj/double_proj are C-implemented (in the modular staticlib, via
 * libsecp256k1) and use preserve_most so the hot preserve_none block callers
 * keep their live state in registers across the call. The remaining entry
 * points are Rust-implemented and use the standard C ABI. Setup entry points
 * return a bool indicating whether the on-chip setup values matched. */

extern __attribute__((preserve_most)) void rvr_ext_ec_add_proj_k256(RvState*, uint64_t rd_ptr,
                                                                    uint64_t rs1_ptr,
                                                                    uint64_t rs2_ptr);
extern bool rvr_ext_setup_ec_add_proj_k256(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                           uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_ec_double_proj_k256(RvState*, uint64_t rd_ptr,
                                                                       uint64_t rs1_ptr);
extern bool rvr_ext_setup_ec_double_proj_k256(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr);

extern void rvr_ext_ec_add_proj_p256(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern bool rvr_ext_setup_ec_add_proj_p256(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                           uint64_t rs2_ptr);
extern void rvr_ext_ec_double_proj_p256(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr);
extern bool rvr_ext_setup_ec_double_proj_p256(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr);

extern void rvr_ext_ec_add_proj_bn254(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern bool rvr_ext_setup_ec_add_proj_bn254(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                            uint64_t rs2_ptr);
extern void rvr_ext_ec_double_proj_bn254(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr);
extern bool rvr_ext_setup_ec_double_proj_bn254(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr);

extern void rvr_ext_ec_add_proj_bls12_381(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                          uint64_t rs2_ptr);
extern bool rvr_ext_setup_ec_add_proj_bls12_381(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                                uint64_t rs2_ptr);
extern void rvr_ext_ec_double_proj_bls12_381(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr);
extern bool rvr_ext_setup_ec_double_proj_bls12_381(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr);

#endif /* RVR_EXT_ECC_H */
