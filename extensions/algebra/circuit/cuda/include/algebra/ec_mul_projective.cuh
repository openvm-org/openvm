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
static constexpr uint32_t EC_MUL_BATCH_INVERT_THREADS = 32;
static constexpr size_t EC_MUL_BATCH_INVERT_ITEMS_PER_THREAD =
    EC_MUL_PROJECTIVE_STATES / EC_MUL_BATCH_INVERT_THREADS;
static_assert(
    EC_MUL_PROJECTIVE_STATES % EC_MUL_BATCH_INVERT_THREADS == 0,
    "projective states must divide evenly across the inversion warp"
);
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

template <uint32_t K, bool ZERO_A>
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
    if constexpr (!ZERO_A) {
        f.square(z, tmp);
        f.square(tmp, tmp2);
        f.mul(curve_a, tmp2, tmp);
        f.add(e, tmp, e);
    }
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

template <uint32_t K, size_t BLOCKS, bool ZERO_A>
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
    uint32_t px[K] = {}, py[K], curve_a[K] = {}, neg_py[K] = {}, x[K], y[K], z[K];
    uint32_t nx[K], ny[K], nz[K], numerator[K];
    if constexpr (!ZERO_A) {
        uint8_t a_bytes[48] = {};
        for (uint32_t byte = 0; byte < s.num_limbs; byte++) {
            a_bytes[byte] = static_cast<uint8_t>(s.setup_values[byte]);
        }
        f.canonical_bytes_to_mont(a_bytes, curve_a);
    }
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
            ec_mul_projective_double<K, ZERO_A>(
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
static __device__ __noinline__ bool ec_mul_projective_batch_invert_chunked(
    const FieldExprProg &s,
    uint32_t *rows,
    uint32_t *error
) {
    constexpr uint32_t vars_per_step = K == 12 ? 6 : 5;
    constexpr uint32_t output_x = K == 12 ? 10 : 8;
    if (s.num_vars != vars_per_step * EC_MUL_STEPS_PER_ROW || s.n_outputs != 2 ||
        s.outputs[0] != output_x || s.outputs[1] != output_x + 1) {
        if (threadIdx.x == 0) preflight_set_error(error, EC_MUL_BAD_PROGRAM);
        return false;
    }
    if (blockDim.x != EC_MUL_BATCH_INVERT_THREADS) {
        if (threadIdx.x == 0) preflight_set_error(error, EC_MUL_BAD_PROGRAM);
        return false;
    }

    constexpr uint32_t full_warp = 0xffffffffu;
    const uint32_t lane = threadIdx.x;
    const size_t first_state = lane * EC_MUL_BATCH_INVERT_ITEMS_PER_THREAD;
    uint32_t work[2 * K + 2], chunk_product[K], prefix[K], suffix[K], other[K];
    // Every lane evaluates the shuffle operand, even though lane 31 is selected as the source.
    // Keep non-source lanes initialized to avoid architecture-dependent reads of indeterminate data.
    uint32_t inv_total[K] = {}, inv_chunk[K], running[K], zi[K], tmp[K];
    EcMulProjectiveField<K> f{s, work};

    // Each lane builds inclusive prefixes for one contiguous 16-state chunk. Contiguous chunks
    // keep the reverse pass local to the lane; the prefix slots are dead until materialization.
    bool nonzero = true;
    for (size_t item = 0; item < EC_MUL_BATCH_INVERT_ITEMS_PER_THREAD; item++) {
        size_t state = first_state + item;
        uint32_t *cur = rows + state * EC_MUL_PROJECTIVE_STATE_WORDS<K>;
        const uint32_t *z = cur + 2 * K;
        nonzero &= !limbs_are_zero(z, K);
        if (item == 0) f.copy(z, chunk_product);
        else {
            f.mul(chunk_product, z, tmp);
            f.copy(tmp, chunk_product);
        }
        f.copy(chunk_product, cur + 4 * K);
    }
    if (__ballot_sync(full_warp, nonzero) != full_warp) {
        if (lane == 0) preflight_set_error(error, FIELD_EXPR_ACTIVE_ZERO_DIVISOR);
        return false;
    }

    // Inclusive prefix and suffix scans over the 32 chunk products. Montgomery multiplication is
    // associative, so warp shuffle order does not affect the resulting field values.
    f.copy(chunk_product, prefix);
    f.copy(chunk_product, suffix);
    for (uint32_t offset = 1; offset < EC_MUL_BATCH_INVERT_THREADS; offset <<= 1) {
        for (uint32_t word = 0; word < K; word++) {
            other[word] = __shfl_up_sync(full_warp, prefix[word], offset);
        }
        if (lane >= offset) {
            f.mul(other, prefix, tmp);
            f.copy(tmp, prefix);
        }
    }
    for (uint32_t offset = 1; offset < EC_MUL_BATCH_INVERT_THREADS; offset <<= 1) {
        for (uint32_t word = 0; word < K; word++) {
            other[word] = __shfl_down_sync(full_warp, suffix[word], offset);
        }
        if (lane + offset < EC_MUL_BATCH_INVERT_THREADS) {
            f.mul(suffix, other, tmp);
            f.copy(tmp, suffix);
        }
    }

    if (lane == EC_MUL_BATCH_INVERT_THREADS - 1) {
        ec_mul_projective_mont_inv<K>(s, prefix, inv_total, work);
    }
    for (uint32_t word = 0; word < K; word++) {
        inv_total[word] = __shfl_sync(
            full_warp, inv_total[word], EC_MUL_BATCH_INVERT_THREADS - 1
        );
    }

    // Invert the product of this lane's chunk from the inverse total and the products on either
    // side. The edge lanes deliberately omit the empty product, avoiding a separately computed
    // Montgomery-one identity.
    f.copy(inv_total, inv_chunk);
    for (uint32_t word = 0; word < K; word++) {
        other[word] = __shfl_up_sync(full_warp, prefix[word], 1);
    }
    if (lane != 0) {
        f.mul(inv_chunk, other, tmp);
        f.copy(tmp, inv_chunk);
    }
    for (uint32_t word = 0; word < K; word++) {
        other[word] = __shfl_down_sync(full_warp, suffix[word], 1);
    }
    if (lane + 1 != EC_MUL_BATCH_INVERT_THREADS) {
        f.mul(inv_chunk, other, tmp);
        f.copy(tmp, inv_chunk);
    }

    // Reverse each chunk independently. Its inclusive-prefix slots become the individual Z
    // inverses consumed unchanged by the existing row-parallel materialization pass.
    f.copy(inv_chunk, running);
    for (size_t item = EC_MUL_BATCH_INVERT_ITEMS_PER_THREAD; item-- > 0;) {
        size_t state = first_state + item;
        uint32_t *cur = rows + state * EC_MUL_PROJECTIVE_STATE_WORDS<K>;
        if (item == 0) f.copy(running, zi);
        else {
            uint32_t *prev = cur - EC_MUL_PROJECTIVE_STATE_WORDS<K>;
            f.mul(running, prev + 4 * K, zi);
            f.mul(running, cur + 2 * K, running);
        }
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
        constexpr uint32_t vars_per_step = K == 12 ? 6 : 5;
        const uint32_t step_base = static_cast<uint32_t>(vars_per_step * step);
        const uint32_t var_base = step_base +
            (post_add ? (K == 12 ? 3 : 2) : (K == 12 ? 1 : 0));
        const uint32_t *zi = cur + 4 * K;

        // At 48 bytes the expression builder saves E before the doubling quotient to bring its
        // carry bound under the configured maximum. The 32-byte program does not need that slot.
        if constexpr (K == 12) {
            if (!post_add) {
                size_t state_index = row * EC_MUL_PROJECTIVE_STATES_PER_ROW + within_row;
                if (state_index == 0) {
                    f.copy(cur + 3 * K, value);
                } else {
                    uint32_t *previous = cur - EC_MUL_PROJECTIVE_STATE_WORDS<K>;
                    f.square(previous + 4 * K, zi2);
                    f.square(zi2, zi2);
                    f.mul(cur + 3 * K, zi2, value);
                }
                f.mont_to_canonical(value, canonical);
                for (uint32_t word = 0; word < K; word++) {
                    vars[(step_base * K + word) * total_rows + flat_row] = canonical[word];
                }
            }
        }

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
