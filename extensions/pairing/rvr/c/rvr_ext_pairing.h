#ifndef RVR_EXT_PAIRING_H
#define RVR_EXT_PAIRING_H

#include <stdint.h>

struct RvState;

/* HintFinalExp phantom: computes pairing final exponentiation hint and
 * sets the hint stream (implemented in Rust). */
extern void rvr_ext_pairing_hint_final_exp_bn254(RvState* state, uint32_t rs1_val,
                                                 uint32_t rs2_val);

extern void rvr_ext_pairing_hint_final_exp_bls12_381(RvState* state, uint32_t rs1_val,
                                                     uint32_t rs2_val);

#endif /* RVR_EXT_PAIRING_H */
