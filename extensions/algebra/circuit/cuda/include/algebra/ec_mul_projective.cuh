#pragma once

#include "algebra/ec_mul_tracegen.cuh"

// Generic short-Weierstrass ladder acceleration. Values remain in the program's Montgomery domain.
// Four K-word slots per row store X, Y, Z, and the prefix product used for batch normalization.
template <uint32_t K>
static constexpr size_t EC_MUL_PROJECTIVE_POINT_WORDS = 4 * K;

template <uint32_t K>
static __device__ __noinline__ void ec_mul_projective_mont_mul(
    const FieldExprProg &s,
    const uint32_t *x,
    const uint32_t *y,
    uint32_t *out,
    uint32_t *work
) {
    mont_mul<K>(s, x, y, out, work);
}

template <uint32_t K>
static __device__ __noinline__ void ec_mul_projective_mont_inv(
    const FieldExprProg &s, const uint32_t *x, uint32_t *out, uint32_t *work
) {
    mont_inv<K>(s, x, out, work);
}

template <uint32_t K>
struct EcMulProjectiveField {
    const FieldExprProg &s;
    uint32_t *work;

    __device__ void mul(const uint32_t *x, const uint32_t *y, uint32_t *out) const {
        ec_mul_projective_mont_mul<K>(s, x, y, out, work);
    }
    __device__ void square(const uint32_t *x, uint32_t *out) const { mul(x, x, out); }
    __device__ void add(const uint32_t *x, const uint32_t *y, uint32_t *out) const {
        add_mod<K>(s, x, y, out, work);
    }
    __device__ void sub(const uint32_t *x, const uint32_t *y, uint32_t *out) const {
        sub_mod<K>(s, x, y, out);
    }
    __device__ void copy(const uint32_t *x, uint32_t *out) const {
        for (uint32_t i = 0; i < K; i++) out[i] = x[i];
    }
    __device__ void canonical_bytes_to_mont(const uint8_t *bytes, uint32_t *out) const {
        uint32_t canonical[K] = {};
        for (uint32_t i = 0; i < s.num_limbs; i++) {
            canonical[i / 4] |= static_cast<uint32_t>(bytes[i]) << (8 * (i % 4));
        }
        mul(canonical, s.r2, out);
    }
    __device__ void mont_to_canonical_bytes(const uint32_t *x, uint8_t *out) const {
        uint32_t one[K] = {}, canonical[K];
        one[0] = 1;
        mul(x, one, canonical);
        for (uint32_t i = 0; i < s.num_limbs; i++) {
            out[i] = static_cast<uint8_t>(canonical[i / 4] >> (8 * (i % 4)));
        }
    }
};

template <uint32_t K>
static __device__ __noinline__ void ec_mul_projective_double(
    const EcMulProjectiveField<K> &f,
    const uint32_t *x,
    const uint32_t *y,
    const uint32_t *z,
    const uint32_t *curve_a,
    uint32_t *xo,
    uint32_t *yo,
    uint32_t *zo,
    uint32_t *t
) {
    // dbl-2009-l, including the generic a*Z^4 term in E.
    uint32_t *a = t, *b = t + K, *c = t + 2 * K, *d = t + 3 * K;
    uint32_t *e = t + 4 * K, *tmp = t + 5 * K, *tmp2 = t + 6 * K;
    f.square(x, a);
    f.square(y, b);
    f.square(b, c);
    f.add(x, b, tmp);
    f.square(tmp, tmp2);
    f.sub(tmp2, a, tmp2);
    f.sub(tmp2, c, tmp2);
    f.add(tmp2, tmp2, d);
    f.add(a, a, tmp);
    f.add(tmp, a, e);
    f.square(z, tmp);
    f.square(tmp, tmp2);
    f.mul(curve_a, tmp2, tmp);
    f.add(e, tmp, e);
    f.square(e, tmp);
    f.add(d, d, tmp2);
    f.sub(tmp, tmp2, xo);
    f.sub(d, xo, tmp);
    f.mul(e, tmp, tmp);
    f.add(c, c, tmp2);
    f.add(tmp2, tmp2, tmp2);
    f.add(tmp2, tmp2, tmp2);
    f.sub(tmp, tmp2, yo);
    f.mul(y, z, tmp);
    f.add(tmp, tmp, zo);
}

template <uint32_t K>
static __device__ __noinline__ void ec_mul_projective_add_base(
    const EcMulProjectiveField<K> &f,
    const uint32_t *x,
    const uint32_t *y,
    const uint32_t *z,
    const uint32_t *px,
    const uint32_t *signed_py,
    uint32_t *xo,
    uint32_t *yo,
    uint32_t *zo,
    uint32_t *t
) {
    // madd-2007-bl, with the second point affine.
    uint32_t *z2 = t, *u2 = t + K, *s2 = t + 2 * K, *h = t + 3 * K;
    uint32_t *hh = t + 4 * K, *i = t + 5 * K, *j = t + 6 * K;
    uint32_t *r = t + 7 * K, *v = t + 8 * K, *tmp = t + 9 * K, *tmp2 = t + 10 * K;
    f.square(z, z2);
    f.mul(px, z2, u2);
    f.mul(z, z2, tmp);
    f.mul(signed_py, tmp, s2);
    f.sub(u2, x, h);
    f.square(h, hh);
    f.add(hh, hh, i);
    f.add(i, i, i);
    f.mul(h, i, j);
    f.sub(s2, y, tmp);
    f.add(tmp, tmp, r);
    f.mul(x, i, v);
    f.add(z, h, tmp);
    f.square(tmp, tmp2);
    f.sub(tmp2, z2, tmp2);
    f.sub(tmp2, hh, zo);
    f.square(r, tmp);
    f.sub(tmp, j, tmp);
    f.add(v, v, tmp2);
    f.sub(tmp, tmp2, xo);
    f.sub(v, xo, tmp);
    f.mul(r, tmp, tmp);
    f.mul(y, j, tmp2);
    f.add(tmp2, tmp2, tmp2);
    f.sub(tmp, tmp2, yo);
}

template <uint32_t K, size_t BLOCKS>
static __device__ __noinline__ bool ec_mul_projective_build_projective(
    const FieldExprProg &s,
    const EcMulTraceInput<BLOCKS> &input,
    uint32_t *rows,
    uint32_t *error
) {
    if (input.is_setup != 0) return true;
    const uint8_t *point = reinterpret_cast<const uint8_t *>(&input.point_blocks[0][0]);
    const uint8_t *scalar = reinterpret_cast<const uint8_t *>(&input.scalar_blocks[0][0]);
    uint32_t work[2 * K + 2], temps[11 * K];
    EcMulProjectiveField<K> f{s, work};
    if (s.n_setup_values != 1) {
        preflight_set_error(error, EC_MUL_BAD_PROGRAM);
        return false;
    }
    uint32_t px[K], py[K], curve_a[K], neg_py[K] = {}, x[K], y[K], z[K], nx[K], ny[K], nz[K];
    f.canonical_bytes_to_mont(point, px);
    f.canonical_bytes_to_mont(point + s.num_limbs, py);
    uint8_t a_bytes[48] = {};
    for (uint32_t byte = 0; byte < s.num_limbs; byte++) {
        a_bytes[byte] = static_cast<uint8_t>(s.setup_values[byte]);
    }
    f.canonical_bytes_to_mont(a_bytes, curve_a);
    f.sub(neg_py, py, neg_py);
    f.copy(px, x);
    f.copy(py, y);
    uint32_t one[K] = {};
    one[0] = 1;
    f.mul(one, s.r2, z);

    for (size_t row = 0; row < EC_MUL_COMPUTE_ROWS; row++) {
        uint32_t *saved = rows + row * EC_MUL_PROJECTIVE_POINT_WORDS<K>;
        f.copy(x, saved);
        f.copy(y, saved + K);
        f.copy(z, saved + 2 * K);
        uint32_t pattern = ec_mul_sign_pattern_for_row(scalar, row);
        for (size_t step = 0; step < EC_MUL_STEPS_PER_ROW; step++) {
            ec_mul_projective_double<K>(f, x, y, z, curve_a, nx, ny, nz, temps);
            bool plus = ((pattern >> (EC_MUL_STEPS_PER_ROW - 1 - step)) & 1u) != 0;
            ec_mul_projective_add_base<K>(f, nx, ny, nz, px, plus ? py : neg_py, x, y, z, temps);
        }
    }
    return true;
}

template <uint32_t K>
static __device__ __noinline__ bool ec_mul_projective_normalize(
    const FieldExprProg &s,
    uint32_t *rows,
    uint8_t *affine,
    size_t affine_rows,
    size_t first_row,
    uint32_t *error
) {
    uint32_t work[2 * K + 2], running[K], inv[K], zi[K], zi2[K], value[K];
    uint8_t canonical[MAX_U32_LIMBS * sizeof(uint32_t)];
    EcMulProjectiveField<K> f{s, work};
    uint32_t *first_prefix = rows + 3 * K;
    f.copy(rows + 2 * K, first_prefix);
    for (size_t row = 1; row < EC_MUL_COMPUTE_ROWS; row++) {
        uint32_t *cur = rows + row * EC_MUL_PROJECTIVE_POINT_WORDS<K>;
        uint32_t *prev = cur - EC_MUL_PROJECTIVE_POINT_WORDS<K>;
        f.mul(prev + 3 * K, cur + 2 * K, cur + 3 * K);
    }
    uint32_t *last = rows + (EC_MUL_COMPUTE_ROWS - 1) * EC_MUL_PROJECTIVE_POINT_WORDS<K>;
    if (limbs_are_zero(last + 3 * K, K)) {
        preflight_set_error(error, EC_MUL_BAD_PROGRAM);
        return false;
    }
    ec_mul_projective_mont_inv<K>(s, last + 3 * K, inv, work);
    f.copy(inv, running);
    for (size_t row = EC_MUL_COMPUTE_ROWS; row-- > 0;) {
        uint32_t *cur = rows + row * EC_MUL_PROJECTIVE_POINT_WORDS<K>;
        if (row == 0) f.copy(running, zi);
        else {
            uint32_t *prev = cur - EC_MUL_PROJECTIVE_POINT_WORDS<K>;
            f.mul(running, prev + 3 * K, zi);
            f.mul(running, cur + 2 * K, running);
        }
        f.square(zi, zi2);
        f.mul(cur, zi2, value);
        f.mont_to_canonical_bytes(value, canonical);
        for (uint32_t byte = 0; byte < s.num_limbs; byte++) {
            affine[byte * affine_rows + first_row + row] = canonical[byte];
        }
        f.mul(zi2, zi, zi2);
        f.mul(cur + K, zi2, value);
        f.mont_to_canonical_bytes(value, canonical);
        for (uint32_t byte = 0; byte < s.num_limbs; byte++) {
            affine[(s.num_limbs + byte) * affine_rows + first_row + row] = canonical[byte];
        }
    }
    return true;
}
