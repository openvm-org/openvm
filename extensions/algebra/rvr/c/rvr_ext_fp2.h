#ifndef RVR_EXT_FP2_H
#define RVR_EXT_FP2_H

#include <stdint.h>

struct RvState;

/* ── Generic fallback FFI (for unknown moduli, uses BigUint) ───────────── */

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

/* ── Specialized per-curve fp2 FFI ─────────────────────────────────────── */

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

#endif /* RVR_EXT_FP2_H */
