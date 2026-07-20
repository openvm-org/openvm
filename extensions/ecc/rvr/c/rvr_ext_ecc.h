#ifndef RVR_EXT_ECC_H
#define RVR_EXT_ECC_H

#include <stdint.h>

struct RvState;

/* Direct C functions use preserve_most so generated block functions keep live
 * values in registers across calls. Rust functions use the standard C ABI. */

extern __attribute__((preserve_most)) void rvr_ext_ec_add_ne_k256(RvState*, uint64_t rd_ptr,
                                                                  uint64_t rs1_ptr,
                                                                  uint64_t rs2_ptr);
extern void rvr_ext_setup_ec_add_ne_k256(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                         uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_ec_double_k256(RvState*, uint64_t rd_ptr,
                                                                  uint64_t rs1_ptr);
extern void rvr_ext_setup_ec_double_k256(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr);

extern void rvr_ext_ec_add_ne_p256(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern void rvr_ext_setup_ec_add_ne_p256(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                         uint64_t rs2_ptr);
extern void rvr_ext_ec_double_p256(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr);
extern void rvr_ext_setup_ec_double_p256(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr);

extern void rvr_ext_ec_add_ne_bn254(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern void rvr_ext_setup_ec_add_ne_bn254(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                          uint64_t rs2_ptr);
extern void rvr_ext_ec_double_bn254(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr);
extern void rvr_ext_setup_ec_double_bn254(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr);

extern __attribute__((preserve_most)) void rvr_ext_ec_add_ne_bls12_381(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern void rvr_ext_setup_ec_add_ne_bls12_381(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                              uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_ec_double_bls12_381(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr);
extern void rvr_ext_setup_ec_double_bls12_381(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr);

#endif /* RVR_EXT_ECC_H */
