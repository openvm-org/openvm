#ifndef RVR_EXT_ECC_H
#define RVR_EXT_ECC_H

#include <stdint.h>

struct RvState;

/* k256 add_ne/double are C-implemented (rvr_ext_k256.c + rvr_ext_k256_ec.h)
 * and use preserve_most so the hot preserve_none block callers keep their
 * live state in registers across the call. The remaining entry points are
 * Rust-implemented and use the standard C ABI. */

extern __attribute__((preserve_most)) void rvr_ext_ec_add_ne_k256(RvState*, uint32_t rd_ptr,
                                                                  uint32_t rs1_ptr,
                                                                  uint32_t rs2_ptr);
extern void rvr_ext_setup_ec_add_ne_k256(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                         uint32_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_ec_double_k256(RvState*, uint32_t rd_ptr,
                                                                  uint32_t rs1_ptr);
extern void rvr_ext_setup_ec_double_k256(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr);

extern void rvr_ext_ec_add_ne_p256(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_setup_ec_add_ne_p256(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                         uint32_t rs2_ptr);
extern void rvr_ext_ec_double_p256(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr);
extern void rvr_ext_setup_ec_double_p256(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr);

extern void rvr_ext_ec_add_ne_bn254(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_setup_ec_add_ne_bn254(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                          uint32_t rs2_ptr);
extern void rvr_ext_ec_double_bn254(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr);
extern void rvr_ext_setup_ec_double_bn254(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr);

extern void rvr_ext_ec_add_ne_bls12_381(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                        uint32_t rs2_ptr);
extern void rvr_ext_setup_ec_add_ne_bls12_381(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                              uint32_t rs2_ptr);
extern void rvr_ext_ec_double_bls12_381(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr);
extern void rvr_ext_setup_ec_double_bls12_381(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr);

#endif /* RVR_EXT_ECC_H */
