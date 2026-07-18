/* RVR wrappers for BLS12-381 Fp, Fr, Fp2, and affine G1 arithmetic using blst. */

#include "openvm.h"
#include "rvr_ext_bls12_381.h"
#include <blst.h>
#include <string.h>

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "BLS12-381 guest limb codecs require a little-endian host"
#endif

static constexpr uint32_t RVR_WORD_SIZE = 8;
static constexpr uint32_t BLS12_381_FP_BYTES = 48;
static constexpr uint32_t BLS12_381_FP_WORDS = BLS12_381_FP_BYTES / RVR_WORD_SIZE;
static constexpr uint32_t BLS12_381_FR_BYTES = 32;
static constexpr uint32_t BLS12_381_FR_WORDS = BLS12_381_FR_BYTES / RVR_WORD_SIZE;

/* ── Fp helpers ────────────────────────────────────────────────────────── */

static inline blst_fp fp_read(RvState *restrict state, uint64_t ptr) {
    uint64_t words[BLS12_381_FP_WORDS];
    read_mem_u64_range(state, ptr, words, BLS12_381_FP_WORDS);
    blst_fp r;
    blst_fp_from_lendian(&r, (const uint8_t *)words);
    return r;
}

static inline void fp_write(RvState *restrict state, uint64_t ptr, const blst_fp *val) {
    uint64_t words[BLS12_381_FP_WORDS];
    blst_lendian_from_fp((uint8_t *)words, val);
    write_mem_u64_range(state, ptr, words, BLS12_381_FP_WORDS);
}

static inline blst_fp fp_add(const blst_fp *a, const blst_fp *b) {
    blst_fp r;
    blst_fp_add(&r, a, b);
    return r;
}

static inline blst_fp fp_sub(const blst_fp *a, const blst_fp *b) {
    blst_fp r;
    blst_fp_sub(&r, a, b);
    return r;
}

static inline blst_fp fp_mul(const blst_fp *a, const blst_fp *b) {
    blst_fp r;
    blst_fp_mul(&r, a, b);
    return r;
}

static inline blst_fp fp_inv(const blst_fp *a) {
    blst_fp r;
    blst_fp_inverse(&r, a);
    return r;
}

/* blst keeps Fp in canonical Montgomery form; same value -> same limbs. */
static inline int fp_eq(const blst_fp *a, const blst_fp *b) {
    return memcmp(a, b, sizeof(blst_fp)) == 0;
}

static inline int fp_is_zero(const blst_fp *a) {
    static const blst_fp zero = {{0, 0, 0, 0, 0, 0}};
    return fp_eq(a, &zero);
}

/* ── Fp2 helpers ───────────────────────────────────────────────────────── */

static inline blst_fp2 fp2_read(RvState *restrict state, uint64_t ptr) {
    blst_fp2 r;
    r.fp[0] = fp_read(state, ptr);
    r.fp[1] = fp_read(state, ptr + BLS12_381_FP_BYTES);
    return r;
}

static inline void fp2_write(RvState *restrict state, uint64_t ptr, const blst_fp2 *val) {
    fp_write(state, ptr, &val->fp[0]);
    fp_write(state, ptr + BLS12_381_FP_BYTES, &val->fp[1]);
}

static inline int fp2_is_zero(const blst_fp2 *a) {
    return fp_is_zero(&a->fp[0]) && fp_is_zero(&a->fp[1]);
}

/* ── Fr helpers ────────────────────────────────────────────────────────── */

static inline blst_fr fr_read(RvState *restrict state, uint64_t ptr) {
    uint64_t words[BLS12_381_FR_WORDS];
    read_mem_u64_range(state, ptr, words, BLS12_381_FR_WORDS);
    blst_scalar s;
    blst_scalar_from_lendian(&s, (const uint8_t *)words);
    blst_fr r;
    blst_fr_from_scalar(&r, &s);
    return r;
}

static inline void fr_write(RvState *restrict state, uint64_t ptr, const blst_fr *val) {
    blst_scalar s;
    blst_scalar_from_fr(&s, val);
    uint64_t words[BLS12_381_FR_WORDS];
    blst_lendian_from_scalar((uint8_t *)words, &s);
    write_mem_u64_range(state, ptr, words, BLS12_381_FR_WORDS);
}

static inline int fr_eq(const blst_fr *a, const blst_fr *b) {
    return memcmp(a, b, sizeof(blst_fr)) == 0;
}

static inline int fr_is_zero(const blst_fr *a) {
    static const blst_fr zero = {{0, 0, 0, 0}};
    return fr_eq(a, &zero);
}

/* ── BLS12-381 base field (mod p, "fq") ──────────────────────────────── */

__attribute__((preserve_most)) void rvr_ext_mod_add_bls12_381_fq(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr,
    uint64_t rs2_ptr
) {
    blst_fp a = fp_read(state, rs1_ptr);
    blst_fp b = fp_read(state, rs2_ptr);
    blst_fp r = fp_add(&a, &b);
    fp_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_sub_bls12_381_fq(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr,
    uint64_t rs2_ptr
) {
    blst_fp a = fp_read(state, rs1_ptr);
    blst_fp b = fp_read(state, rs2_ptr);
    blst_fp r = fp_sub(&a, &b);
    fp_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_mul_bls12_381_fq(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr,
    uint64_t rs2_ptr
) {
    blst_fp a = fp_read(state, rs1_ptr);
    blst_fp b = fp_read(state, rs2_ptr);
    blst_fp r = fp_mul(&a, &b);
    fp_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_div_bls12_381_fq(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr,
    uint64_t rs2_ptr
) {
    blst_fp a = fp_read(state, rs1_ptr);
    blst_fp b = fp_read(state, rs2_ptr);
    debug_assume(!fp_is_zero(&b));
    blst_fp b_inv = fp_inv(&b);
    blst_fp r = fp_mul(&a, &b_inv);
    fp_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) uint32_t
rvr_ext_mod_iseq_bls12_381_fq(RvState *restrict state, uint64_t rs1_ptr, uint64_t rs2_ptr) {
    blst_fp a = fp_read(state, rs1_ptr);
    blst_fp b = fp_read(state, rs2_ptr);
    return fp_eq(&a, &b) ? 1u : 0u;
}

/* ── BLS12-381 scalar field (mod r, "fr") ────────────────────────────── */

__attribute__((preserve_most)) void rvr_ext_mod_add_bls12_381_fr(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr,
    uint64_t rs2_ptr
) {
    blst_fr a = fr_read(state, rs1_ptr);
    blst_fr b = fr_read(state, rs2_ptr);
    blst_fr r;
    blst_fr_add(&r, &a, &b);
    fr_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_sub_bls12_381_fr(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr,
    uint64_t rs2_ptr
) {
    blst_fr a = fr_read(state, rs1_ptr);
    blst_fr b = fr_read(state, rs2_ptr);
    blst_fr r;
    blst_fr_sub(&r, &a, &b);
    fr_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_mul_bls12_381_fr(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr,
    uint64_t rs2_ptr
) {
    blst_fr a = fr_read(state, rs1_ptr);
    blst_fr b = fr_read(state, rs2_ptr);
    blst_fr r;
    blst_fr_mul(&r, &a, &b);
    fr_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_div_bls12_381_fr(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr,
    uint64_t rs2_ptr
) {
    blst_fr a = fr_read(state, rs1_ptr);
    blst_fr b = fr_read(state, rs2_ptr);
    debug_assume(!fr_is_zero(&b));
    blst_fr b_inv;
    blst_fr_inverse(&b_inv, &b);
    blst_fr r;
    blst_fr_mul(&r, &a, &b_inv);
    fr_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) uint32_t
rvr_ext_mod_iseq_bls12_381_fr(RvState *restrict state, uint64_t rs1_ptr, uint64_t rs2_ptr) {
    blst_fr a = fr_read(state, rs1_ptr);
    blst_fr b = fr_read(state, rs2_ptr);
    return fr_eq(&a, &b) ? 1u : 0u;
}

/* ── Fp2 arithmetic ──────────────────────────────────────────────────── */

__attribute__((preserve_most)) void rvr_ext_fp2_add_bls12_381(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr,
    uint64_t rs2_ptr
) {
    blst_fp2 a = fp2_read(state, rs1_ptr);
    blst_fp2 b = fp2_read(state, rs2_ptr);
    blst_fp2 r;
    blst_fp2_add(&r, &a, &b);
    fp2_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_fp2_sub_bls12_381(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr,
    uint64_t rs2_ptr
) {
    blst_fp2 a = fp2_read(state, rs1_ptr);
    blst_fp2 b = fp2_read(state, rs2_ptr);
    blst_fp2 r;
    blst_fp2_sub(&r, &a, &b);
    fp2_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_fp2_mul_bls12_381(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr,
    uint64_t rs2_ptr
) {
    blst_fp2 a = fp2_read(state, rs1_ptr);
    blst_fp2 b = fp2_read(state, rs2_ptr);
    blst_fp2 r;
    blst_fp2_mul(&r, &a, &b);
    fp2_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_fp2_div_bls12_381(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr,
    uint64_t rs2_ptr
) {
    blst_fp2 a = fp2_read(state, rs1_ptr);
    blst_fp2 b = fp2_read(state, rs2_ptr);
    debug_assume(!fp2_is_zero(&b));
    blst_fp2 b_inv;
    blst_fp2_inverse(&b_inv, &b);
    blst_fp2 r;
    blst_fp2_mul(&r, &a, &b_inv);
    fp2_write(state, rd_ptr, &r);
}

/* ── G1 EC ops ────────────────────────────────────────────────────────── */

__attribute__((preserve_most)) void rvr_ext_ec_add_ne_bls12_381(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr,
    uint64_t rs2_ptr
) {
    blst_fp x1 = fp_read(state, rs1_ptr);
    blst_fp y1 = fp_read(state, rs1_ptr + BLS12_381_FP_BYTES);
    blst_fp x2 = fp_read(state, rs2_ptr);
    blst_fp y2 = fp_read(state, rs2_ptr + BLS12_381_FP_BYTES);

    /* lambda = (y2 - y1) / (x2 - x1) */
    blst_fp dy = fp_sub(&y2, &y1);
    blst_fp dx = fp_sub(&x2, &x1);
    debug_assume(!fp_is_zero(&dx));
    blst_fp dx_inv = fp_inv(&dx);
    blst_fp lambda = fp_mul(&dy, &dx_inv);

    /* x3 = lambda^2 - x1 - x2 */
    blst_fp lsq = fp_mul(&lambda, &lambda);
    blst_fp x3 = fp_sub(&lsq, &x1);
    x3 = fp_sub(&x3, &x2);

    /* y3 = lambda * (x1 - x3) - y1 */
    blst_fp dx13 = fp_sub(&x1, &x3);
    blst_fp y3 = fp_mul(&lambda, &dx13);
    y3 = fp_sub(&y3, &y1);

    fp_write(state, rd_ptr, &x3);
    fp_write(state, rd_ptr + BLS12_381_FP_BYTES, &y3);
}

__attribute__((preserve_most)) void rvr_ext_ec_double_bls12_381(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr
) {
    blst_fp x1 = fp_read(state, rs1_ptr);
    blst_fp y1 = fp_read(state, rs1_ptr + BLS12_381_FP_BYTES);

    /* lambda = 3*x1^2 / (2*y1)  (a = 0 for BLS12-381) */
    blst_fp x1sq = fp_mul(&x1, &x1);
    blst_fp two_x1sq = fp_add(&x1sq, &x1sq);
    blst_fp three_x1sq = fp_add(&two_x1sq, &x1sq);
    blst_fp two_y1 = fp_add(&y1, &y1);
    debug_assume(!fp_is_zero(&two_y1));
    blst_fp two_y1_inv = fp_inv(&two_y1);
    blst_fp lambda = fp_mul(&three_x1sq, &two_y1_inv);

    /* x3 = lambda^2 - 2*x1 */
    blst_fp lsq = fp_mul(&lambda, &lambda);
    blst_fp two_x1 = fp_add(&x1, &x1);
    blst_fp x3 = fp_sub(&lsq, &two_x1);

    /* y3 = lambda * (x1 - x3) - y1 */
    blst_fp dx = fp_sub(&x1, &x3);
    blst_fp y3 = fp_mul(&lambda, &dx);
    y3 = fp_sub(&y3, &y1);

    fp_write(state, rd_ptr, &x3);
    fp_write(state, rd_ptr + BLS12_381_FP_BYTES, &y3);
}
