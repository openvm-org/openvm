#ifndef RVR_EXT_BIGINT_H
#define RVR_EXT_BIGINT_H

#include <stdint.h>

struct RvState;

/* 256-bit ALU operations (implemented in Rust). One specialized FFI entry
 * point per opcode so the C compiler does not see a runtime `op` switch. */
extern void rvr_ext_int256_add(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_int256_sub(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_int256_xor(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_int256_or(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_int256_and(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_int256_sll(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_int256_srl(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_int256_sra(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_int256_slt(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern void rvr_ext_int256_sltu(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr,
                                uint32_t rs2_ptr);
extern void rvr_ext_int256_mul(RvState* state, uint32_t rd_ptr, uint32_t rs1_ptr, uint32_t rs2_ptr);

/* 256-bit branch predicates. Each returns 1 if the branch should be taken. */
extern uint32_t rvr_ext_int256_beq(RvState* state, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern uint32_t rvr_ext_int256_bne(RvState* state, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern uint32_t rvr_ext_int256_blt(RvState* state, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern uint32_t rvr_ext_int256_bltu(RvState* state, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern uint32_t rvr_ext_int256_bge(RvState* state, uint32_t rs1_ptr, uint32_t rs2_ptr);
extern uint32_t rvr_ext_int256_bgeu(RvState* state, uint32_t rs1_ptr, uint32_t rs2_ptr);

#endif /* RVR_EXT_BIGINT_H */
