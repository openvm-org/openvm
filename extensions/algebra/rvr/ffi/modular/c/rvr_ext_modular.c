/* Single-TU wrapper bundling libsecp256k1 (with ECC modules) plus the
 * rvr-ext FFI shims for k256 modular and EC ops. Compiled at lift time
 * via `ModularRvrExtension::c_sources`.
 */

#include "openvm.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "k256 guest limb codecs require a little-endian host"
#endif

#define RVR_WORD_SIZE 4

/* ── libsecp256k1 amalgamation (ECC modules always enabled) ──────────── */
#include "secp256k1.c"

/* ── k256 field (mod p) helpers ──────────────────────────────────────── */

#define SECP256K1_ELEM_BYTES 32u
#define SECP256K1_ELEM_WORDS (SECP256K1_ELEM_BYTES / RVR_WORD_SIZE)

static inline void bytes_reverse_32(uint8_t out[SECP256K1_ELEM_BYTES],
                                    const uint8_t in[SECP256K1_ELEM_BYTES]) {
    _Static_assert(SECP256K1_ELEM_BYTES == 4 * sizeof(uint64_t), "");
    uint64_t words[4];
    memcpy(words, in, sizeof(words));
    uint64_t reversed[4] = {
        __builtin_bswap64(words[3]),
        __builtin_bswap64(words[2]),
        __builtin_bswap64(words[1]),
        __builtin_bswap64(words[0]),
    };
    memcpy(out, reversed, sizeof(reversed));
}

static inline secp256k1_fe fe_read(RvState *state, uint32_t ptr) {
    uint32_t words[SECP256K1_ELEM_WORDS];
    rd_mem_u32_range_traced(state, ptr, words, SECP256K1_ELEM_WORDS);
    uint8_t le[SECP256K1_ELEM_BYTES];
    memcpy(le, words, SECP256K1_ELEM_BYTES);
    uint8_t be[SECP256K1_ELEM_BYTES];
    bytes_reverse_32(be, le);
    secp256k1_fe r;
    /* Match the Rust path: guest inputs are reduced mod p. */
    secp256k1_fe_set_b32_mod(&r, be);
    return r;
}

static inline void fe_write(RvState *state, uint32_t ptr, const secp256k1_fe *val) {
    secp256k1_fe t = *val;
    secp256k1_fe_normalize_var(&t);
    uint8_t be[SECP256K1_ELEM_BYTES];
    secp256k1_fe_get_b32(be, &t);
    uint8_t le[SECP256K1_ELEM_BYTES];
    bytes_reverse_32(le, be);
    uint32_t words[SECP256K1_ELEM_WORDS];
    memcpy(words, le, SECP256K1_ELEM_BYTES);
    wr_mem_u32_range_traced(state, ptr, words, SECP256K1_ELEM_WORDS);
}

static inline secp256k1_fe fe_add(secp256k1_fe a, const secp256k1_fe *b) {
    secp256k1_fe_add(&a, b);
    secp256k1_fe_normalize_weak(&a);
    return a;
}

static inline secp256k1_fe fe_sub(const secp256k1_fe *a, const secp256k1_fe *b) {
    secp256k1_fe neg_b;
    secp256k1_fe_negate(&neg_b, b, 1);
    secp256k1_fe r = *a;
    secp256k1_fe_add(&r, &neg_b);
    secp256k1_fe_normalize_weak(&r);
    return r;
}

static inline secp256k1_fe fe_mul(const secp256k1_fe *a, const secp256k1_fe *b) {
    secp256k1_fe r;
    secp256k1_fe_mul(&r, a, b);
    return r;
}

static inline secp256k1_fe fe_inv(const secp256k1_fe *a) {
    secp256k1_fe r;
    secp256k1_fe_inv_var(&r, a);
    return r;
}

static inline int fe_is_zero(const secp256k1_fe *a) {
    secp256k1_fe t = *a;
    secp256k1_fe_normalize_var(&t);
    return secp256k1_fe_is_zero(&t);
}

/* ── k256 scalar (mod n) helpers ─────────────────────────────────────── */

static inline secp256k1_scalar scalar_read(RvState *state, uint32_t ptr) {
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

static inline void scalar_write(RvState *state, uint32_t ptr,
                                const secp256k1_scalar *val) {
    uint8_t be[SECP256K1_ELEM_BYTES];
    secp256k1_scalar_get_b32(be, val);
    uint8_t le[SECP256K1_ELEM_BYTES];
    bytes_reverse_32(le, be);
    uint32_t words[SECP256K1_ELEM_WORDS];
    memcpy(words, le, SECP256K1_ELEM_BYTES);
    wr_mem_u32_range_traced(state, ptr, words, SECP256K1_ELEM_WORDS);
}

/* ── Modular arithmetic: secp256k1 coordinate field (mod p) ──────────── */

__attribute__((preserve_most))
void rvr_ext_mod_add_k256_coord(RvState *state, uint32_t rd_ptr,
                                uint32_t rs1_ptr, uint32_t rs2_ptr) {
    secp256k1_fe a = fe_read(state, rs1_ptr);
    secp256k1_fe b = fe_read(state, rs2_ptr);
    secp256k1_fe r = fe_add(a, &b);
    fe_write(state, rd_ptr, &r);
}

__attribute__((preserve_most))
void rvr_ext_mod_sub_k256_coord(RvState *state, uint32_t rd_ptr,
                                uint32_t rs1_ptr, uint32_t rs2_ptr) {
    secp256k1_fe a = fe_read(state, rs1_ptr);
    secp256k1_fe b = fe_read(state, rs2_ptr);
    secp256k1_fe r = fe_sub(&a, &b);
    fe_write(state, rd_ptr, &r);
}

__attribute__((preserve_most))
void rvr_ext_mod_mul_k256_coord(RvState *state, uint32_t rd_ptr,
                                uint32_t rs1_ptr, uint32_t rs2_ptr) {
    secp256k1_fe a = fe_read(state, rs1_ptr);
    secp256k1_fe b = fe_read(state, rs2_ptr);
    secp256k1_fe r = fe_mul(&a, &b);
    fe_write(state, rd_ptr, &r);
}

__attribute__((preserve_most))
void rvr_ext_mod_div_k256_coord(RvState *state, uint32_t rd_ptr,
                                uint32_t rs1_ptr, uint32_t rs2_ptr) {
    secp256k1_fe a = fe_read(state, rs1_ptr);
    secp256k1_fe b = fe_read(state, rs2_ptr);
    debug_assume(!fe_is_zero(&b));
    secp256k1_fe b_inv = fe_inv(&b);
    secp256k1_fe r = fe_mul(&a, &b_inv);
    fe_write(state, rd_ptr, &r);
}

__attribute__((preserve_most))
uint32_t rvr_ext_mod_iseq_k256_coord(RvState *state, uint32_t rs1_ptr, uint32_t rs2_ptr) {
    secp256k1_fe a = fe_read(state, rs1_ptr);
    secp256k1_fe b = fe_read(state, rs2_ptr);
    return secp256k1_fe_equal(&a, &b) ? 1u : 0u;
}

/* ── Modular arithmetic: secp256k1 scalar field (mod n) ──────────────── */

__attribute__((preserve_most))
void rvr_ext_mod_add_k256_scalar(RvState *state, uint32_t rd_ptr,
                                 uint32_t rs1_ptr, uint32_t rs2_ptr) {
    secp256k1_scalar a = scalar_read(state, rs1_ptr);
    secp256k1_scalar b = scalar_read(state, rs2_ptr);
    secp256k1_scalar r;
    secp256k1_scalar_add(&r, &a, &b);
    scalar_write(state, rd_ptr, &r);
}

__attribute__((preserve_most))
void rvr_ext_mod_sub_k256_scalar(RvState *state, uint32_t rd_ptr,
                                 uint32_t rs1_ptr, uint32_t rs2_ptr) {
    secp256k1_scalar a = scalar_read(state, rs1_ptr);
    secp256k1_scalar b = scalar_read(state, rs2_ptr);
    secp256k1_scalar neg_b;
    secp256k1_scalar_negate(&neg_b, &b);
    secp256k1_scalar r;
    secp256k1_scalar_add(&r, &a, &neg_b);
    scalar_write(state, rd_ptr, &r);
}

__attribute__((preserve_most))
void rvr_ext_mod_mul_k256_scalar(RvState *state, uint32_t rd_ptr,
                                 uint32_t rs1_ptr, uint32_t rs2_ptr) {
    secp256k1_scalar a = scalar_read(state, rs1_ptr);
    secp256k1_scalar b = scalar_read(state, rs2_ptr);
    secp256k1_scalar r;
    secp256k1_scalar_mul(&r, &a, &b);
    scalar_write(state, rd_ptr, &r);
}

__attribute__((preserve_most))
void rvr_ext_mod_div_k256_scalar(RvState *state, uint32_t rd_ptr,
                                 uint32_t rs1_ptr, uint32_t rs2_ptr) {
    secp256k1_scalar a = scalar_read(state, rs1_ptr);
    secp256k1_scalar b = scalar_read(state, rs2_ptr);
    debug_assume(!secp256k1_scalar_is_zero(&b));
    secp256k1_scalar b_inv;
    secp256k1_scalar_inverse_var(&b_inv, &b);
    secp256k1_scalar r;
    secp256k1_scalar_mul(&r, &a, &b_inv);
    scalar_write(state, rd_ptr, &r);
}

__attribute__((preserve_most))
uint32_t rvr_ext_mod_iseq_k256_scalar(RvState *state, uint32_t rs1_ptr, uint32_t rs2_ptr) {
    secp256k1_scalar a = scalar_read(state, rs1_ptr);
    secp256k1_scalar b = scalar_read(state, rs2_ptr);
    return secp256k1_scalar_eq(&a, &b) ? 1u : 0u;
}

/* ── EC ops: secp256k1 (always present in modular, regardless of whether
 * the ECC extension is configured at lift time) ──────────────────────── */

__attribute__((preserve_most))
void rvr_ext_ec_add_ne_k256(RvState *state, uint32_t rd_ptr,
                            uint32_t rs1_ptr, uint32_t rs2_ptr) {
    secp256k1_fe x1 = fe_read(state, rs1_ptr);
    secp256k1_fe y1 = fe_read(state, rs1_ptr + SECP256K1_ELEM_BYTES);
    secp256k1_fe x2 = fe_read(state, rs2_ptr);
    secp256k1_fe y2 = fe_read(state, rs2_ptr + SECP256K1_ELEM_BYTES);

    /* lambda = (y2 - y1) / (x2 - x1) */
    secp256k1_fe dy = fe_sub(&y2, &y1);
    secp256k1_fe dx = fe_sub(&x2, &x1);
    debug_assume(!fe_is_zero(&dx));
    secp256k1_fe dx_inv = fe_inv(&dx);
    secp256k1_fe lambda = fe_mul(&dy, &dx_inv);

    /* x3 = lambda^2 - x1 - x2 */
    secp256k1_fe lsq = fe_mul(&lambda, &lambda);
    secp256k1_fe x3 = fe_sub(&lsq, &x1);
    x3 = fe_sub(&x3, &x2);

    /* y3 = lambda * (x1 - x3) - y1 */
    secp256k1_fe dx13 = fe_sub(&x1, &x3);
    secp256k1_fe y3 = fe_mul(&lambda, &dx13);
    y3 = fe_sub(&y3, &y1);

    fe_write(state, rd_ptr, &x3);
    fe_write(state, rd_ptr + SECP256K1_ELEM_BYTES, &y3);
}

__attribute__((preserve_most))
void rvr_ext_ec_double_k256(RvState *state, uint32_t rd_ptr, uint32_t rs1_ptr) {
    secp256k1_fe x1 = fe_read(state, rs1_ptr);
    secp256k1_fe y1 = fe_read(state, rs1_ptr + SECP256K1_ELEM_BYTES);

    /* lambda = 3*x1^2 / (2*y1)  (a = 0 for secp256k1) */
    secp256k1_fe x1sq = fe_mul(&x1, &x1);
    secp256k1_fe three_x1sq = fe_add(fe_add(x1sq, &x1sq), &x1sq);
    secp256k1_fe two_y1 = fe_add(y1, &y1);
    debug_assume(!fe_is_zero(&two_y1));
    secp256k1_fe two_y1_inv = fe_inv(&two_y1);
    secp256k1_fe lambda = fe_mul(&three_x1sq, &two_y1_inv);

    /* x3 = lambda^2 - 2*x1 */
    secp256k1_fe lsq = fe_mul(&lambda, &lambda);
    secp256k1_fe two_x1 = fe_add(x1, &x1);
    secp256k1_fe x3 = fe_sub(&lsq, &two_x1);

    /* y3 = lambda * (x1 - x3) - y1 */
    secp256k1_fe dx = fe_sub(&x1, &x3);
    secp256k1_fe y3 = fe_mul(&lambda, &dx);
    y3 = fe_sub(&y3, &y1);

    fe_write(state, rd_ptr, &x3);
    fe_write(state, rd_ptr + SECP256K1_ELEM_BYTES, &y3);
}
