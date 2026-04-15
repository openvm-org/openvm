#ifndef RVR_EXT_ALGEBRA_H
#define RVR_EXT_ALGEBRA_H

#include <stdint.h>

struct RvState;

/* k256 coord/scalar ops are C-implemented (rvr_ext_k256.c) and use
 * preserve_most so the hot preserve_none block callers keep their live state
 * in registers across the call. All other entry points are Rust-implemented
 * and use the standard C ABI. */

/* ── Generic fallback FFI (for unknown moduli, uses BigUint) ───────────── */

/* Modular arithmetic fallback FFI for unknown moduli.
 * One entry point per opcode; modulus is still provided at runtime. */
extern void rvr_ext_mod_add(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr,
                            uint32_t num_limbs, const uint8_t* modulus);
extern void rvr_ext_mod_sub(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr,
                            uint32_t num_limbs, const uint8_t* modulus);
extern void rvr_ext_mod_mul(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr,
                            uint32_t num_limbs, const uint8_t* modulus);
extern void rvr_ext_mod_div(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr,
                            uint32_t num_limbs, const uint8_t* modulus);

/* Modular IS_EQ extension FFI entry point (implemented in Rust).
 * Returns 1 if equal, 0 otherwise. */
extern uint32_t rvr_ext_mod_iseq(RvState* state, uint32_t rs1_ptr, uint32_t rs2_ptr,
                                 uint32_t num_limbs, const uint8_t* modulus);

/* Modular SETUP extension FFI entry point (implemented in Rust). */
extern void rvr_ext_mod_setup(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr,
                              uint32_t num_limbs);

/* Fp2 arithmetic fallback FFI for unknown base fields.
 * One entry point per opcode; modulus is still provided at runtime. */
extern void rvr_ext_fp2_add(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr,
                            uint32_t num_limbs, const uint8_t* modulus);
extern void rvr_ext_fp2_sub(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr,
                            uint32_t num_limbs, const uint8_t* modulus);
extern void rvr_ext_fp2_mul(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr,
                            uint32_t num_limbs, const uint8_t* modulus);
extern void rvr_ext_fp2_div(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr,
                            uint32_t num_limbs, const uint8_t* modulus);

/* Fp2 SETUP extension FFI entry point (implemented in Rust). */
extern void rvr_ext_fp2_setup(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr,
                              uint32_t num_limbs);

/* HintSqrt phantom: computes sqrt hint and sets hint stream (implemented in
 * Rust). */
extern void rvr_ext_algebra_hint_sqrt(RvState* state, uint32_t rs1_ptr, uint32_t num_limbs,
                                      const uint8_t* modulus, const uint8_t* non_qr);

/* ── Specialized per-curve-per-op FFI (native Montgomery arithmetic) ─── */

/* k256 coord/scalar: C-implemented with preserve_most. */
extern __attribute__((preserve_most)) void rvr_ext_mod_add_k256_coord(RvState*, uint32_t rd_ptr,
                                                                      uint32_t rs1_ptr,
                                                                      uint32_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_sub_k256_coord(RvState*, uint32_t rd_ptr,
                                                                      uint32_t rs1_ptr,
                                                                      uint32_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_mul_k256_coord(RvState*, uint32_t rd_ptr,
                                                                      uint32_t rs1_ptr,
                                                                      uint32_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_div_k256_coord(RvState*, uint32_t rd_ptr,
                                                                      uint32_t rs1_ptr,
                                                                      uint32_t rs2_ptr);
extern __attribute__((preserve_most)) uint32_t rvr_ext_mod_iseq_k256_coord(RvState*,
                                                                           uint32_t rs1_ptr,
                                                                           uint32_t rs2_ptr);

extern __attribute__((preserve_most)) void rvr_ext_mod_add_k256_scalar(RvState*, uint32_t rd_ptr,
                                                                       uint32_t rs1_ptr,
                                                                       uint32_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_sub_k256_scalar(RvState*, uint32_t rd_ptr,
                                                                       uint32_t rs1_ptr,
                                                                       uint32_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_mul_k256_scalar(RvState*, uint32_t rd_ptr,
                                                                       uint32_t rs1_ptr,
                                                                       uint32_t rs2_ptr);
extern __attribute__((preserve_most)) void rvr_ext_mod_div_k256_scalar(RvState*, uint32_t rd_ptr,
                                                                       uint32_t rs1_ptr,
                                                                       uint32_t rs2_ptr);
extern __attribute__((preserve_most)) uint32_t rvr_ext_mod_iseq_k256_scalar(RvState*,
                                                                            uint32_t rs1_ptr,
                                                                            uint32_t rs2_ptr);

/* Remaining curves: Rust-implemented, standard C ABI. */
extern void rvr_ext_mod_add_p256_coord(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                       uint32_t rs2_ptr);
extern void rvr_ext_mod_sub_p256_coord(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                       uint32_t rs2_ptr);
extern void rvr_ext_mod_mul_p256_coord(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                       uint32_t rs2_ptr);
extern void rvr_ext_mod_div_p256_coord(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                       uint32_t rs2_ptr);
extern uint32_t rvr_ext_mod_iseq_p256_coord(RvState*, uint32_t rs1_ptr, uint32_t rs2_ptr);

extern void rvr_ext_mod_add_p256_scalar(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                        uint32_t rs2_ptr);
extern void rvr_ext_mod_sub_p256_scalar(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                        uint32_t rs2_ptr);
extern void rvr_ext_mod_mul_p256_scalar(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                        uint32_t rs2_ptr);
extern void rvr_ext_mod_div_p256_scalar(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                        uint32_t rs2_ptr);
extern uint32_t rvr_ext_mod_iseq_p256_scalar(RvState*, uint32_t rs1_ptr, uint32_t rs2_ptr);

extern void rvr_ext_mod_add_bn254_fq(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_mod_sub_bn254_fq(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_mod_mul_bn254_fq(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_mod_div_bn254_fq(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern uint32_t rvr_ext_mod_iseq_bn254_fq(RvState*, uint32_t rs1_ptr, uint32_t rs2_ptr);

extern void rvr_ext_mod_add_bn254_fr(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_mod_sub_bn254_fr(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_mod_mul_bn254_fr(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_mod_div_bn254_fr(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern uint32_t rvr_ext_mod_iseq_bn254_fr(RvState*, uint32_t rs1_ptr, uint32_t rs2_ptr);

extern void rvr_ext_mod_add_bls12_381_fq(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                         uint32_t rs2_ptr);
extern void rvr_ext_mod_sub_bls12_381_fq(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                         uint32_t rs2_ptr);
extern void rvr_ext_mod_mul_bls12_381_fq(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                         uint32_t rs2_ptr);
extern void rvr_ext_mod_div_bls12_381_fq(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                         uint32_t rs2_ptr);
extern uint32_t rvr_ext_mod_iseq_bls12_381_fq(RvState*, uint32_t rs1_ptr, uint32_t rs2_ptr);

extern void rvr_ext_mod_add_bls12_381_fr(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                         uint32_t rs2_ptr);
extern void rvr_ext_mod_sub_bls12_381_fr(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                         uint32_t rs2_ptr);
extern void rvr_ext_mod_mul_bls12_381_fr(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                         uint32_t rs2_ptr);
extern void rvr_ext_mod_div_bls12_381_fr(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                         uint32_t rs2_ptr);
extern uint32_t rvr_ext_mod_iseq_bls12_381_fr(RvState*, uint32_t rs1_ptr, uint32_t rs2_ptr);

extern void rvr_ext_fp2_add_bn254(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_fp2_sub_bn254(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_fp2_mul_bn254(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_fp2_div_bn254(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);

extern void rvr_ext_fp2_add_bls12_381(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                      uint32_t rs2_ptr);
extern void rvr_ext_fp2_sub_bls12_381(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                      uint32_t rs2_ptr);
extern void rvr_ext_fp2_mul_bls12_381(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                      uint32_t rs2_ptr);
extern void rvr_ext_fp2_div_bls12_381(RvState*, uint32_t rd_ptr, uint32_t rs1_ptr,
                                      uint32_t rs2_ptr);

#endif /* RVR_EXT_ALGEBRA_H */
