#pragma once

#include "algebra/ec_mul_tracegen.cuh"

// Generic short-Weierstrass ladder acceleration. Values remain in the program's Montgomery domain.
// Every row contains two `D = 2R`, `R' = D + sigma*P` steps. Each state stores its
// Jacobian X/Y/Z, the projective slope numerator, and a prefix-product slot used by the single
// instruction-wide batch inversion.
template <uint32_t K>
static constexpr size_t EC_MUL_PROJECTIVE_STATE_WORDS = 5 * K;
static constexpr size_t EC_MUL_PROJECTIVE_STATES_PER_ROW = 2 * EC_MUL_STEPS_PER_ROW;
static constexpr size_t EC_MUL_PROJECTIVE_STATES =
    EC_MUL_COMPUTE_ROWS * EC_MUL_PROJECTIVE_STATES_PER_ROW;
template <uint32_t K>
static constexpr size_t EC_MUL_PROJECTIVE_INSTRUCTION_WORDS =
    EC_MUL_PROJECTIVE_STATES * EC_MUL_PROJECTIVE_STATE_WORDS<K>;

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
    __device__ void mont_to_canonical(const uint32_t *x, uint32_t *out) const {
        uint32_t one[K] = {};
        one[0] = 1;
        mul(x, one, out);
    }
};

template <uint32_t K>
static __device__ __noinline__ void ec_mul_projective_double(
    const EcMulProjectiveField<K> &f,
    const uint32_t *x,
    const uint32_t *y,
    const uint32_t *z,
    const uint32_t *curve_a,
    uint32_t *slope_numerator,
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
    f.copy(e, slope_numerator);
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
    uint32_t *slope_numerator,
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
    f.copy(r, slope_numerator);
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
    uint32_t px[K] = {}, py[K], curve_a[K], neg_py[K] = {}, x[K], y[K], z[K];
    uint32_t nx[K], ny[K], nz[K], numerator[K];
    uint8_t a_bytes[48] = {};
    for (uint32_t byte = 0; byte < s.num_limbs; byte++) {
        a_bytes[byte] = static_cast<uint8_t>(s.setup_values[byte]);
    }
    f.canonical_bytes_to_mont(a_bytes, curve_a);
    f.canonical_bytes_to_mont(point, px);
    f.canonical_bytes_to_mont(point + s.num_limbs, py);
    f.copy(px, x);
    f.copy(py, y);
    f.sub(neg_py, py, neg_py);
    uint32_t one[K] = {};
    one[0] = 1;
    f.mul(one, s.r2, z);

    for (size_t row = 0; row < EC_MUL_COMPUTE_ROWS; row++) {
        uint32_t pattern = ec_mul_sign_pattern_for_row(scalar, row);
        for (size_t step = 0; step < EC_MUL_STEPS_PER_ROW; step++) {
            size_t state_idx = row * EC_MUL_PROJECTIVE_STATES_PER_ROW + 2 * step;
            uint32_t *d = rows + state_idx * EC_MUL_PROJECTIVE_STATE_WORDS<K>;
            ec_mul_projective_double<K>(
                f, x, y, z, curve_a, numerator, nx, ny, nz, temps
            );
            if (limbs_are_zero(nz, K)) {
                preflight_set_error(error, FIELD_EXPR_ACTIVE_ZERO_DIVISOR);
                return false;
            }
            f.copy(nx, d);
            f.copy(ny, d + K);
            f.copy(nz, d + 2 * K);
            f.copy(numerator, d + 3 * K);

            bool plus = ((pattern >> (EC_MUL_STEPS_PER_ROW - 1 - step)) & 1u) != 0;
            uint32_t *r = d + EC_MUL_PROJECTIVE_STATE_WORDS<K>;
            ec_mul_projective_add_base<K>(
                f, nx, ny, nz, px, plus ? py : neg_py, numerator, x, y, z, temps
            );
            if (limbs_are_zero(z, K)) {
                preflight_set_error(error, FIELD_EXPR_ACTIVE_ZERO_DIVISOR);
                return false;
            }
            f.copy(x, r);
            f.copy(y, r + K);
            f.copy(z, r + 2 * K);
            f.copy(numerator, r + 3 * K);
        }
    }
    return true;
}

template <uint32_t K>
static __device__ __noinline__ bool ec_mul_projective_batch_invert(
    const FieldExprProg &s,
    uint32_t *rows,
    uint32_t *error
) {
    if (s.num_vars != 5 * EC_MUL_STEPS_PER_ROW || s.n_outputs != 2 ||
        s.outputs[0] != 8 || s.outputs[1] != 9) {
        preflight_set_error(error, EC_MUL_BAD_PROGRAM);
        return false;
    }
    uint32_t work[2 * K + 2], running[K], inv[K], zi[K];
    EcMulProjectiveField<K> f{s, work};
    uint32_t *first_prefix = rows + 4 * K;
    f.copy(rows + 2 * K, first_prefix);
    for (size_t state = 1; state < EC_MUL_PROJECTIVE_STATES; state++) {
        uint32_t *cur = rows + state * EC_MUL_PROJECTIVE_STATE_WORDS<K>;
        uint32_t *prev = cur - EC_MUL_PROJECTIVE_STATE_WORDS<K>;
        f.mul(prev + 4 * K, cur + 2 * K, cur + 4 * K);
    }
    uint32_t *last = rows + (EC_MUL_PROJECTIVE_STATES - 1) * EC_MUL_PROJECTIVE_STATE_WORDS<K>;
    if (limbs_are_zero(last + 4 * K, K)) {
        preflight_set_error(error, EC_MUL_BAD_PROGRAM);
        return false;
    }
    ec_mul_projective_mont_inv<K>(s, last + 4 * K, inv, work);
    f.copy(inv, running);
    for (size_t state = EC_MUL_PROJECTIVE_STATES; state-- > 0;) {
        uint32_t *cur = rows + state * EC_MUL_PROJECTIVE_STATE_WORDS<K>;
        if (state == 0) f.copy(running, zi);
        else {
            uint32_t *prev = cur - EC_MUL_PROJECTIVE_STATE_WORDS<K>;
            f.mul(running, prev + 4 * K, zi);
            f.mul(running, cur + 2 * K, running);
        }

        // The prefix slot is dead after the reverse scan reaches this state. Reuse it for Z^-1
        // so a row-parallel pass can materialize the exact affine values independently.
        f.copy(zi, cur + 4 * K);
    }
    return true;
}

template <uint32_t K>
static __device__ __noinline__ void ec_mul_projective_materialize_row(
    const FieldExprProg &s,
    uint32_t *rows,
    uint32_t *vars,
    size_t total_rows,
    size_t flat_row
) {
    uint32_t work[2 * K + 2], zi2[K], value[K], canonical[K];
    EcMulProjectiveField<K> f{s, work};
    size_t instruction = flat_row / EC_MUL_COMPUTE_ROWS;
    size_t row = flat_row % EC_MUL_COMPUTE_ROWS;
    uint32_t *instruction_rows = rows + instruction * EC_MUL_PROJECTIVE_INSTRUCTION_WORDS<K>;
    for (size_t within_row = 0; within_row < EC_MUL_PROJECTIVE_STATES_PER_ROW; within_row++) {
        uint32_t *cur = instruction_rows +
            (row * EC_MUL_PROJECTIVE_STATES_PER_ROW + within_row) *
                EC_MUL_PROJECTIVE_STATE_WORDS<K>;
        const size_t step = within_row / 2;
        const bool post_add = (within_row & 1) != 0;
        const uint32_t var_base = static_cast<uint32_t>(5 * step + (post_add ? 2 : 0));
        const uint32_t *zi = cur + 4 * K;

        // For doubling, Z_D = 2*Y*Z and M = 3*X^2 + a*Z^4, so lambda_d = M/Z_D.
        // For mixed addition, Z_R = 2*Z_D*H and r = 2*(S_2-Y_D), so lambda_a = r/Z_R.
        f.mul(cur + 3 * K, zi, value);
        f.mont_to_canonical(value, canonical);
        for (uint32_t word = 0; word < K; word++) {
            vars[(var_base * K + word) * total_rows + flat_row] = canonical[word];
        }

        f.square(zi, zi2);
        f.mul(cur, zi2, value);
        f.mont_to_canonical(value, canonical);
        for (uint32_t word = 0; word < K; word++) {
            vars[((var_base + 1) * K + word) * total_rows + flat_row] = canonical[word];
        }
        if (post_add) {
            f.mul(zi2, zi, zi2);
            f.mul(cur + K, zi2, value);
            f.mont_to_canonical(value, canonical);
            for (uint32_t word = 0; word < K; word++) {
                vars[((var_base + 2) * K + word) * total_rows + flat_row] = canonical[word];
            }
        }
    }
}
