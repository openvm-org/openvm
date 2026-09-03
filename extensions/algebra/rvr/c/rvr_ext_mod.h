#ifndef RVR_EXT_MOD_H
#define RVR_EXT_MOD_H

#include <stdint.h>

typedef struct RvState RvState;

/* Direct C functions use preserve_most so generated block functions keep live
 * values in registers across calls. Rust functions use the standard C ABI. */

/* ── Generic fallback FFI (for unknown moduli, uses BigUint) ───────────── */

/* Modular arithmetic fallback FFI for unknown moduli.
 * One entry point per opcode; modulus is still provided at runtime. */
extern void rvr_ext_mod_add(RvState* state, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr,
                            uint32_t num_limbs, const uint8_t* modulus);
extern void rvr_ext_mod_sub(RvState* state, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr,
                            uint32_t num_limbs, const uint8_t* modulus);
extern void rvr_ext_mod_mul(RvState* state, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr,
                            uint32_t num_limbs, const uint8_t* modulus);
extern void rvr_ext_mod_div(RvState* state, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr,
                            uint32_t num_limbs, const uint8_t* modulus);

/* Modular IS_EQ function implemented in Rust. */
extern bool rvr_ext_mod_iseq(RvState* state, uint64_t rs1_ptr, uint64_t rs2_ptr,
                             uint32_t num_limbs, const uint8_t* modulus);

/* Modular SETUP extension FFI entry point (implemented in Rust). */
extern bool rvr_ext_mod_setup(RvState* state, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr,
                              uint32_t num_limbs, const uint8_t* modulus);
extern uint8_t rvr_ext_mod_setup_iseq(RvState* state, uint64_t rs1_ptr, uint64_t rs2_ptr,
                                      uint32_t num_limbs, const uint8_t* modulus);

/* HintSqrt phantom: computes sqrt hint and sets hint stream (implemented in
 * Rust). */
extern void rvr_ext_algebra_hint_sqrt(RvState* state, uint64_t rs1_ptr, uint32_t num_limbs,
                                      const uint8_t* modulus, const uint8_t* non_qr);

/* ── Specialized per-curve-per-op FFI (native Montgomery arithmetic) ─── */

/* k256 coord/scalar: C-implemented with preserve_most. */
extern __attribute__((preserve_most)) void rvr_ext_mod_add_k256_coord(RvState*, uint64_t rd_ptr,
                                                                      uint64_t rs1_ptr,
                                                                      uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_sub_k256_coord(RvState*, uint64_t rd_ptr,
                                                                      uint64_t rs1_ptr,
                                                                      uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_mul_k256_coord(RvState*, uint64_t rd_ptr,
                                                                      uint64_t rs1_ptr,
                                                                      uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_div_k256_coord(RvState*, uint64_t rd_ptr,
                                                                      uint64_t rs1_ptr,
                                                                      uint64_t rs2_ptr);
extern __attribute__((preserve_most)) bool rvr_ext_mod_iseq_k256_coord(RvState*,
                                                                       uint64_t rs1_ptr,
                                                                       uint64_t rs2_ptr);

extern __attribute__((preserve_most)) void rvr_ext_mod_add_k256_scalar(RvState*, uint64_t rd_ptr,
                                                                       uint64_t rs1_ptr,
                                                                       uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_sub_k256_scalar(RvState*, uint64_t rd_ptr,
                                                                       uint64_t rs1_ptr,
                                                                       uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_mul_k256_scalar(RvState*, uint64_t rd_ptr,
                                                                       uint64_t rs1_ptr,
                                                                       uint64_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_div_k256_scalar(RvState*, uint64_t rd_ptr,
                                                                       uint64_t rs1_ptr,
                                                                       uint64_t rs2_ptr);
extern __attribute__((preserve_most)) bool rvr_ext_mod_iseq_k256_scalar(RvState*,
                                                                        uint64_t rs1_ptr,
                                                                        uint64_t rs2_ptr);

/* P-256 and BN254 use Rust functions with the standard C ABI. */
extern void rvr_ext_mod_add_p256_coord(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                       uint64_t rs2_ptr);
extern void rvr_ext_mod_sub_p256_coord(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                       uint64_t rs2_ptr);
extern void rvr_ext_mod_mul_p256_coord(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                       uint64_t rs2_ptr);
extern void rvr_ext_mod_div_p256_coord(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                       uint64_t rs2_ptr);
extern bool rvr_ext_mod_iseq_p256_coord(RvState*, uint64_t rs1_ptr, uint64_t rs2_ptr);

extern void rvr_ext_mod_add_p256_scalar(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                        uint64_t rs2_ptr);
extern void rvr_ext_mod_sub_p256_scalar(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                        uint64_t rs2_ptr);
extern void rvr_ext_mod_mul_p256_scalar(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                        uint64_t rs2_ptr);
extern void rvr_ext_mod_div_p256_scalar(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr,
                                        uint64_t rs2_ptr);
extern bool rvr_ext_mod_iseq_p256_scalar(RvState*, uint64_t rs1_ptr, uint64_t rs2_ptr);

extern void rvr_ext_mod_add_bn254_fq(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern void rvr_ext_mod_sub_bn254_fq(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern void rvr_ext_mod_mul_bn254_fq(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern void rvr_ext_mod_div_bn254_fq(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern bool rvr_ext_mod_iseq_bn254_fq(RvState*, uint64_t rs1_ptr, uint64_t rs2_ptr);

extern void rvr_ext_mod_add_bn254_fr(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern void rvr_ext_mod_sub_bn254_fr(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern void rvr_ext_mod_mul_bn254_fr(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern void rvr_ext_mod_div_bn254_fr(RvState*, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr);
extern bool rvr_ext_mod_iseq_bn254_fr(RvState*, uint64_t rs1_ptr, uint64_t rs2_ptr);

#endif /* RVR_EXT_MOD_H */
