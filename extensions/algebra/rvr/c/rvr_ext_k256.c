/* k256 field (mod p) and scalar (mod n) FFI, backed by libsecp256k1.
 * This is native execution code, so libsecp256k1's variable-time helpers are
 * appropriate here. OpenVM modular-div and ECC opcodes rely on caller
 * preconditions for invertibility / non-degenerate points, so we keep the
 * old Rust debug-assert behavior via `assert(...)` and let release builds
 * stay branch-free with `__builtin_assume(...)`. */

#include "openvm.h"
#include "rvr_ext_ecc.h"
#include <string.h>

#include "secp256k1.c"
#define RVR_EXT_K256_IMPL 1
#include "rvr_ext_k256_fe.h"

/* ── Scalar (mod n) helpers ───────────────────────────────────────────── */

static inline secp256k1_scalar scalar_read(RvState* restrict state, uint32_t ptr) {
    uint32_t words[SECP256K1_ELEM_WORDS];
    rd_mem_u32_range_traced(state, ptr, words, SECP256K1_ELEM_WORDS);
    uint8_t le[SECP256K1_ELEM_BYTES];
    memcpy(le, words, SECP256K1_ELEM_BYTES);
    uint8_t be[SECP256K1_ELEM_BYTES];
    bytes_reverse_32(be, le);
    secp256k1_scalar r;
    secp256k1_scalar_set_b32(&r, be, NULL);
    return r;
}

static inline void scalar_write(RvState* restrict state, uint32_t ptr,
                                const secp256k1_scalar* val) {
    uint8_t be[SECP256K1_ELEM_BYTES];
    secp256k1_scalar_get_b32(be, val);
    uint8_t le[SECP256K1_ELEM_BYTES];
    bytes_reverse_32(le, be);
    uint32_t words[SECP256K1_ELEM_WORDS];
    memcpy(words, le, SECP256K1_ELEM_BYTES);
    wr_mem_u32_range_traced(state, ptr, words, SECP256K1_ELEM_WORDS);
}

/* ── Modular arithmetic: secp256k1 coordinate field (mod p) ───────────── */

__attribute__((preserve_most)) void rvr_ext_mod_add_k256_coord(RvState* restrict state,
                                                               uint32_t rd_ptr, uint32_t rs1_ptr,
                                                               uint32_t rs2_ptr) {
    secp256k1_fe a = fe_read(state, rs1_ptr);
    secp256k1_fe b = fe_read(state, rs2_ptr);
    secp256k1_fe r = fe_add(a, &b);
    fe_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_sub_k256_coord(RvState* restrict state,
                                                               uint32_t rd_ptr, uint32_t rs1_ptr,
                                                               uint32_t rs2_ptr) {
    secp256k1_fe a = fe_read(state, rs1_ptr);
    secp256k1_fe b = fe_read(state, rs2_ptr);
    secp256k1_fe r = fe_sub(&a, &b);
    fe_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_mul_k256_coord(RvState* restrict state,
                                                               uint32_t rd_ptr, uint32_t rs1_ptr,
                                                               uint32_t rs2_ptr) {
    secp256k1_fe a = fe_read(state, rs1_ptr);
    secp256k1_fe b = fe_read(state, rs2_ptr);
    secp256k1_fe r = fe_mul(&a, &b);
    fe_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_div_k256_coord(RvState* restrict state,
                                                               uint32_t rd_ptr, uint32_t rs1_ptr,
                                                               uint32_t rs2_ptr) {
    secp256k1_fe a = fe_read(state, rs1_ptr);
    secp256k1_fe b = fe_read(state, rs2_ptr);
    debug_assume(!fe_is_zero(&b));
    secp256k1_fe b_inv = fe_inv(&b);
    secp256k1_fe r = fe_mul(&a, &b_inv);
    fe_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) uint32_t rvr_ext_mod_iseq_k256_coord(RvState* restrict state,
                                                                    uint32_t rs1_ptr,
                                                                    uint32_t rs2_ptr) {
    secp256k1_fe a = fe_read(state, rs1_ptr);
    secp256k1_fe b = fe_read(state, rs2_ptr);
    return secp256k1_fe_equal(&a, &b) ? 1u : 0u;
}

/* ── Modular arithmetic: secp256k1 scalar field (mod n) ───────────────── */

__attribute__((preserve_most)) void rvr_ext_mod_add_k256_scalar(RvState* restrict state,
                                                                uint32_t rd_ptr, uint32_t rs1_ptr,
                                                                uint32_t rs2_ptr) {
    secp256k1_scalar a = scalar_read(state, rs1_ptr);
    secp256k1_scalar b = scalar_read(state, rs2_ptr);
    secp256k1_scalar r;
    secp256k1_scalar_add(&r, &a, &b);
    scalar_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_sub_k256_scalar(RvState* restrict state,
                                                                uint32_t rd_ptr, uint32_t rs1_ptr,
                                                                uint32_t rs2_ptr) {
    secp256k1_scalar a = scalar_read(state, rs1_ptr);
    secp256k1_scalar b = scalar_read(state, rs2_ptr);
    secp256k1_scalar neg_b;
    secp256k1_scalar_negate(&neg_b, &b);
    secp256k1_scalar r;
    secp256k1_scalar_add(&r, &a, &neg_b);
    scalar_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_mul_k256_scalar(RvState* restrict state,
                                                                uint32_t rd_ptr, uint32_t rs1_ptr,
                                                                uint32_t rs2_ptr) {
    secp256k1_scalar a = scalar_read(state, rs1_ptr);
    secp256k1_scalar b = scalar_read(state, rs2_ptr);
    secp256k1_scalar r;
    secp256k1_scalar_mul(&r, &a, &b);
    scalar_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_div_k256_scalar(RvState* restrict state,
                                                                uint32_t rd_ptr, uint32_t rs1_ptr,
                                                                uint32_t rs2_ptr) {
    secp256k1_scalar a = scalar_read(state, rs1_ptr);
    secp256k1_scalar b = scalar_read(state, rs2_ptr);
    debug_assume(!secp256k1_scalar_is_zero(&b));
    secp256k1_scalar b_inv;
    secp256k1_scalar_inverse_var(&b_inv, &b);
    secp256k1_scalar r;
    secp256k1_scalar_mul(&r, &a, &b_inv);
    scalar_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) uint32_t rvr_ext_mod_iseq_k256_scalar(RvState* restrict state,
                                                                     uint32_t rs1_ptr,
                                                                     uint32_t rs2_ptr) {
    secp256k1_scalar a = scalar_read(state, rs1_ptr);
    secp256k1_scalar b = scalar_read(state, rs2_ptr);
    return secp256k1_scalar_eq(&a, &b) ? 1u : 0u;
}

/* EC ops owned by the ecc crate; folded in here because libsecp256k1
 * can only be linked from one TU. */
#include "rvr_ext_k256_ec.h"
