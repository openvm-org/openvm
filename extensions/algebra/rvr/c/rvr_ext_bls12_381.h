#ifndef RVR_EXT_BLS12_381_H
#define RVR_EXT_BLS12_381_H

#include <stdint.h>

typedef struct RvState RvState;

extern __attribute__((preserve_most)) void rvr_ext_mod_add_bls12_381_fq(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_sub_bls12_381_fq(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_mul_bls12_381_fq(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_div_bls12_381_fq(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern __attribute__((preserve_most)) uint32_t rvr_ext_mod_iseq_bls12_381_fq(
    RvState*, uint64_t rs1_ptr, uint64_t rs2_ptr);

extern __attribute__((preserve_most)) void rvr_ext_mod_add_bls12_381_fr(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_sub_bls12_381_fr(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_mul_bls12_381_fr(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_div_bls12_381_fr(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern __attribute__((preserve_most)) uint32_t rvr_ext_mod_iseq_bls12_381_fr(
    RvState*, uint64_t rs1_ptr, uint64_t rs2_ptr);

extern __attribute__((preserve_most)) void rvr_ext_fp2_add_bls12_381(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_fp2_sub_bls12_381(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_fp2_mul_bls12_381(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_fp2_div_bls12_381(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);

extern __attribute__((preserve_most)) void rvr_ext_ec_add_ne_bls12_381(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_ec_double_bls12_381(
    RvState*, uint64_t rd_ptr, uint64_t rs1_ptr);

#endif /* RVR_EXT_BLS12_381_H */
