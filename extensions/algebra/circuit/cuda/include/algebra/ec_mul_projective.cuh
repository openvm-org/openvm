#pragma once

#include "algebra/field_expr_core.cuh"
#include "riscv-adapters/ec_mul_columns.cuh"

#include <cstddef>
#include <cstdint>

// Inversion-free ladder for `EC_MUL` trace generation.
//
// The chip's rows chain, so the accumulator must be advanced sequentially before any row can be
// filled. Advancing it through the field expression would divide four times per row, and the device
// inverts by exponentiation, leaving the trace bound by one sequential thread per instruction.
//
// Jacobian coordinates advance the accumulator without dividing. This header runs the ladder
// projectively, then recovers all `EC_MUL_COMPUTE_ROWS` affine accumulators with one inversion by
// Montgomery's batch trick, leaving each row's divisions to be done in parallel.
//
// Every value here is in Montgomery form, as the interpreter's field helpers expect.

// Scratch slots of `K` words the point operations need.
static constexpr uint32_t EC_MUL_JACOBIAN_TEMPS = 14;
// Leading `K`-word slots of the shared workspace, reserved for Montgomery multiplication.
static constexpr uint32_t EC_MUL_MONT_WORKSPACE_WORDS = 2;

// Total workspace one thread needs to run the ladder.
template <uint32_t K>
static constexpr uint32_t EC_MUL_LADDER_WORKSPACE_WORDS =
    EC_MUL_MONT_WORKSPACE_WORDS * K + 2 + EC_MUL_JACOBIAN_TEMPS * K;

// `__noinline__` wrappers around the field primitives.
//
// The point operations below call these from straight-line code, dozens of times per ladder step.
// Inlined, each copy expands the primitive's K-by-K limb loops, and the resulting function is large
// enough that nvcc's frontend gives up. One frame per call keeps it bounded; the call overhead is
// negligible against the K-squared body.
template <uint32_t K>
static __device__ __noinline__ void ec_mul_mont_mul(
    const FieldExprProg &s, const uint32_t *a, const uint32_t *b, uint32_t *r, uint32_t *ws
) {
    mont_mul<K>(s, a, b, r, ws);
}

template <uint32_t K>
static __device__ __noinline__ void ec_mul_mont_inv(
    const FieldExprProg &s, const uint32_t *a, uint32_t *r, uint32_t *ws
) {
    mont_inv<K>(s, a, r, ws);
}

// Montgomery arithmetic over the expression's prime, on `K`-word values.
template <uint32_t K> struct EcMulMont {
    const FieldExprProg &s;
    uint32_t *ws; // 2K + 2 words

    __device__ void mul(const uint32_t *a, const uint32_t *b, uint32_t *r) const {
        ec_mul_mont_mul<K>(s, a, b, r, ws);
    }
    __device__ void sqr(const uint32_t *a, uint32_t *r) const {
        ec_mul_mont_mul<K>(s, a, a, r, ws);
    }
    __device__ void add(const uint32_t *a, const uint32_t *b, uint32_t *r) const {
        add_mod<K>(s, a, b, r, ws);
    }
    __device__ void sub(const uint32_t *a, const uint32_t *b, uint32_t *r) const {
        sub_mod<K>(s, a, b, r);
    }
    __device__ void twice(const uint32_t *a, uint32_t *r) const { add_mod<K>(s, a, a, r, ws); }
    __device__ void invert(const uint32_t *a, uint32_t *r) const {
        ec_mul_mont_inv<K>(s, a, r, ws);
    }
    __device__ void copy(const uint32_t *a, uint32_t *r) const {
        for (uint32_t i = 0; i < K; i++) r[i] = a[i];
    }
    __device__ void set_zero(uint32_t *r) const {
        for (uint32_t i = 0; i < K; i++) r[i] = 0;
    }
    __device__ bool is_zero(const uint32_t *a) const { return limbs_are_zero(a, K); }

    // Montgomery form of one, that is `R mod p`.
    __device__ void set_one(uint32_t *r) const {
        uint32_t one[K];
        for (uint32_t i = 0; i < K; i++) one[i] = i == 0 ? 1u : 0u;
        mul(one, s.r2, r);
    }

    // Canonical little-endian bytes to Montgomery form. The blob validator pins `limb_bits` to 8,
    // so byte `i` lands in word `i / 4`.
    __device__ void from_bytes(const uint8_t *bytes, uint32_t num_limbs, uint32_t *r) const {
        uint32_t canonical[K];
        for (uint32_t i = 0; i < K; i++) canonical[i] = 0;
        for (uint32_t byte = 0; byte < num_limbs; byte++) {
            canonical[byte / 4] |= static_cast<uint32_t>(bytes[byte]) << ((byte % 4) * 8);
        }
        mul(canonical, s.r2, r);
    }

    // Montgomery form back to canonical little-endian bytes.
    __device__ void to_bytes(const uint32_t *a, uint32_t num_limbs, uint8_t *bytes) const {
        uint32_t one[K];
        for (uint32_t i = 0; i < K; i++) one[i] = i == 0 ? 1u : 0u;
        uint32_t canonical[K];
        mul(a, one, canonical);
        for (uint32_t byte = 0; byte < num_limbs; byte++) {
            bytes[byte] = static_cast<uint8_t>(canonical[byte / 4] >> ((byte % 4) * 8));
        }
    }
};

// A point in Jacobian coordinates, representing affine `(X / Z^2, Y / Z^3)`.
template <uint32_t K> struct EcMulJacobian {
    uint32_t x[K];
    uint32_t y[K];
    uint32_t z[K];
};

// `2 * P` in Jacobian coordinates, for any `a` (EFD `dbl-2007-bl`).
//
// Undefined when `P` is the identity or has order two. The `mul` module documents why the ladder
// reaches neither.
template <uint32_t K>
static __device__ __noinline__ void ec_mul_jacobian_double(
    const EcMulMont<K> &f,
    const EcMulJacobian<K> &p,
    const uint32_t *a_mont,
    EcMulJacobian<K> &out,
    uint32_t *t // EC_MUL_JACOBIAN_TEMPS * K
) {
    uint32_t *xx = t, *yy = t + K, *yyyy = t + 2 * K, *zz = t + 3 * K;
    uint32_t *s_val = t + 4 * K, *m = t + 5 * K, *tt = t + 6 * K, *u = t + 7 * K,
             *v = t + 8 * K;

    f.sqr(p.x, xx);          // XX = X^2
    f.sqr(p.y, yy);          // YY = Y^2
    f.sqr(yy, yyyy);         // YYYY = YY^2
    f.sqr(p.z, zz);          // ZZ = Z^2

    f.add(p.x, yy, u);       // X + YY
    f.sqr(u, v);             // (X + YY)^2
    f.sub(v, xx, v);
    f.sub(v, yyyy, v);
    f.twice(v, s_val);       // S = 2 * ((X + YY)^2 - XX - YYYY)

    f.sqr(zz, u);            // ZZ^2
    f.mul(a_mont, u, u);     // a * ZZ^2
    f.twice(xx, v);
    f.add(v, xx, v);         // 3 * XX
    f.add(v, u, m);          // M = 3*XX + a*ZZ^2

    f.sqr(m, tt);
    f.twice(s_val, u);
    f.sub(tt, u, tt);        // T = M^2 - 2*S

    // Z' reads the original Y and Z, so it is taken before X and Y are overwritten.
    f.add(p.y, p.z, u);
    f.sqr(u, v);
    f.sub(v, yy, v);
    f.sub(v, zz, out.z);     // Z' = (Y + Z)^2 - YY - ZZ

    f.sub(s_val, tt, u);     // S - T
    f.mul(m, u, u);          // M * (S - T)
    f.twice(yyyy, v);
    f.twice(v, v);
    f.twice(v, v);           // 8 * YYYY
    f.sub(u, v, out.y);      // Y' = M*(S-T) - 8*YYYY
    f.copy(tt, out.x);       // X' = T
}

// `P + Q` for Jacobian `P` and affine `Q` (EFD `madd-2007-bl`).
//
// Requires `Q.x != P.x`, the same precondition as the chip's incomplete addition, so both are
// undefined on exactly the same inputs.
template <uint32_t K>
static __device__ __noinline__ void ec_mul_jacobian_add_affine(
    const EcMulMont<K> &f,
    const EcMulJacobian<K> &p,
    const uint32_t *qx,
    const uint32_t *qy,
    EcMulJacobian<K> &out,
    uint32_t *t // EC_MUL_JACOBIAN_TEMPS * K
) {
    uint32_t *z1z1 = t, *u2 = t + K, *s2 = t + 2 * K, *h = t + 3 * K, *hh = t + 4 * K,
             *i_val = t + 5 * K, *j_val = t + 6 * K, *r_val = t + 7 * K, *v_val = t + 8 * K,
             *u = t + 9 * K, *w = t + 10 * K;

    f.sqr(p.z, z1z1);        // Z1Z1 = Z1^2
    f.mul(qx, z1z1, u2);     // U2 = x2 * Z1Z1
    f.mul(p.z, z1z1, u);
    f.mul(qy, u, s2);        // S2 = y2 * Z1 * Z1Z1

    f.sub(u2, p.x, h);       // H = U2 - X1
    f.sqr(h, hh);            // HH = H^2
    f.twice(hh, i_val);
    f.twice(i_val, i_val);   // I = 4 * HH
    f.mul(h, i_val, j_val);  // J = H * I
    f.sub(s2, p.y, u);
    f.twice(u, r_val);       // r = 2 * (S2 - Y1)
    f.mul(p.x, i_val, v_val);// V = X1 * I

    // Z3 reads the original Z1, so it is taken before X3 and Y3 overwrite shared scratch.
    f.add(p.z, h, u);
    f.sqr(u, w);
    f.sub(w, z1z1, w);
    f.sub(w, hh, out.z);     // Z3 = (Z1 + H)^2 - Z1Z1 - HH

    f.sqr(r_val, u);
    f.sub(u, j_val, u);
    f.twice(v_val, w);
    f.sub(u, w, u);          // X3 = r^2 - J - 2*V

    f.sub(v_val, u, w);      // V - X3
    f.mul(r_val, w, w);      // r * (V - X3)
    f.mul(p.y, j_val, i_val);
    f.twice(i_val, i_val);   // 2 * Y1 * J
    f.sub(w, i_val, out.y);  // Y3 = r*(V - X3) - 2*Y1*J
    f.copy(u, out.x);
}

// The curve coefficient `a` in Montgomery form.
//
// The expression folds `a` in as a constant and declares it as its only setup value, so this is the
// same coefficient the constraints use.
template <uint32_t K>
static __device__ bool ec_mul_load_curve_a(
    const EcMulMont<K> &f, const FieldExprProg &s, uint32_t *a_mont
) {
    if (s.n_setup_values != 1) {
        return false;
    }
    uint32_t canonical[K];
    for (uint32_t i = 0; i < K; i++) canonical[i] = 0;
    for (uint32_t byte = 0; byte < s.num_limbs; byte++) {
        canonical[byte / 4] |= (s.setup_values[byte] & 0xffu) << ((byte % 4) * 8);
    }
    f.mul(canonical, s.r2, a_mont);
    return true;
}

// Walks one instruction's ladder, writing the accumulator entering each compute row.
//
// `jacobian` receives `EC_MUL_COMPUTE_ROWS` points. The first is `P`, the most significant digit
// being `+1`; each later entry is the previous one advanced by `EC_MUL_STEPS_PER_ROW` steps of
// `R = 2R + sigma*P`.
template <uint32_t K>
static __device__ __noinline__ bool ec_mul_projective_accumulators(
    const FieldExprProg &s,
    const uint8_t *point_bytes,
    const uint8_t *scalar_bytes,
    EcMulJacobian<K> *jacobian,
    uint32_t *workspace, // EC_MUL_MONT_WORKSPACE_WORDS * K + 2 + EC_MUL_JACOBIAN_TEMPS * K
    uint32_t *err
) {
    uint32_t *mont_ws = workspace;
    uint32_t *temps = workspace + EC_MUL_MONT_WORKSPACE_WORDS * K + 2;
    EcMulMont<K> f{s, mont_ws};

    uint32_t a_mont[K];
    if (!ec_mul_load_curve_a<K>(f, s, a_mont)) {
        preflight_set_error(err, EC_MUL_BAD_PROGRAM);
        return false;
    }

    // The base point and its negation, which the negative digits add.
    uint32_t px[K], py[K], neg_py[K];
    f.from_bytes(point_bytes, s.num_limbs, px);
    f.from_bytes(point_bytes + s.num_limbs, s.num_limbs, py);
    f.set_zero(neg_py);
    f.sub(neg_py, py, neg_py);

    EcMulJacobian<K> accumulator;
    f.copy(px, accumulator.x);
    f.copy(py, accumulator.y);
    f.set_one(accumulator.z);

    EcMulJacobian<K> doubled;
    for (size_t row = 0; row < EC_MUL_COMPUTE_ROWS; row++) {
        jacobian[row] = accumulator;
        uint32_t pattern = ec_mul_sign_pattern_for_row(scalar_bytes, row);
        for (size_t step = 0; step < EC_MUL_STEPS_PER_ROW; step++) {
            // Steps read the pattern's bits most significant first, matching `sign_of`.
            bool positive = ((pattern >> (EC_MUL_STEPS_PER_ROW - 1 - step)) & 1) != 0;
            ec_mul_jacobian_double<K>(f, accumulator, a_mont, doubled, temps);
            ec_mul_jacobian_add_affine<K>(
                f, doubled, px, positive ? py : neg_py, accumulator, temps
            );
        }
    }
    return true;
}

// Recovers every accumulator's affine coordinates with one inversion.
//
// Montgomery's trick: given prefix products of the `Z` values, inverting the last prefix yields
// each `Z^-1` by walking backwards, in place of `EC_MUL_COMPUTE_ROWS` separate inversions.
//
// `affine_out` receives `EC_MUL_COMPUTE_ROWS * 2 * num_limbs` bytes, each row's `x` then `y`, in
// the little-endian form the interpreter reads inputs as.
template <uint32_t K>
static __device__ __noinline__ bool ec_mul_affine_from_jacobian(
    const FieldExprProg &s,
    const EcMulJacobian<K> *jacobian,
    uint32_t *prefix, // EC_MUL_COMPUTE_ROWS * K
    uint8_t *affine_out,
    uint32_t *workspace,
    uint32_t *err
) {
    uint32_t *mont_ws = workspace;
    uint32_t *temps = workspace + EC_MUL_MONT_WORKSPACE_WORDS * K + 2;
    EcMulMont<K> f{s, mont_ws};

    uint32_t *running = temps;
    uint32_t *inverse = temps + K;
    uint32_t *z_inv = temps + 2 * K;
    uint32_t *z_inv2 = temps + 3 * K;
    uint32_t *value = temps + 4 * K;

    // prefix[i] = Z_0 * ... * Z_i
    f.copy(jacobian[0].z, prefix);
    for (size_t row = 1; row < EC_MUL_COMPUTE_ROWS; row++) {
        f.mul(prefix + (row - 1) * K, jacobian[row].z, prefix + row * K);
    }
    // A zero product means some accumulator was the identity, which the ladder cannot reach.
    if (f.is_zero(prefix + (EC_MUL_COMPUTE_ROWS - 1) * K)) {
        preflight_set_error(err, EC_MUL_IDENTITY_ACCUMULATOR);
        return false;
    }
    f.invert(prefix + (EC_MUL_COMPUTE_ROWS - 1) * K, inverse);

    f.copy(inverse, running);
    for (size_t index = EC_MUL_COMPUTE_ROWS; index-- > 0;) {
        if (index == 0) {
            f.copy(running, z_inv);
        } else {
            f.mul(running, prefix + (index - 1) * K, z_inv);
            f.mul(running, jacobian[index].z, running);
        }
        f.sqr(z_inv, z_inv2);
        f.mul(jacobian[index].x, z_inv2, value);
        f.to_bytes(value, s.num_limbs, affine_out + index * 2 * s.num_limbs);
        f.mul(z_inv2, z_inv, z_inv2); // Z^-3
        f.mul(jacobian[index].y, z_inv2, value);
        f.to_bytes(value, s.num_limbs, affine_out + index * 2 * s.num_limbs + s.num_limbs);
    }
    return true;
}
