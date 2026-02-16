#include "fp.h"
#include "fpext.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector_types.h>

struct FpExtPair {
    FpExt first;
    FpExt second;
};

__device__ __forceinline__ FpExt exp_power_of_2_ext(FpExt x, uint32_t power_log) {
    FpExt res = x;
    for (uint32_t i = 0; i < power_log; ++i) {
        res *= res;
    }
    return res;
}

__device__ __forceinline__ FpExt eval_eq_bits_ext(const FpExt *x, uint32_t b, uint32_t num_bits) {
    const FpExt one = FpExt(Fp::one());
    FpExt res = one;
    for (uint32_t i = 0; i < num_bits; ++i) {
        bool bit = (b >> i) & 1;
        FpExt x_i = x[i];
        res *= bit ? x_i : (one - x_i);
    }
    return res;
}

__device__ __forceinline__ FpExt eval_eq_uni_ext(FpExt x, FpExt y, uint32_t l_skip) {
    FpExt res = FpExt(Fp::one());
    for (uint32_t i = 0; i < l_skip; i++) {
        res = (x + y) * res + (x - Fp::one()) * (y - Fp::one());
        x *= x;
        y *= y;
    }
    return res * Fp::one().mul_2exp_neg_n(l_skip);
}

__device__ __forceinline__ FpExt eval_eq_uni_at_one_ext(FpExt x, uint32_t l_skip) {
    FpExt res = FpExt(Fp::one());
    for (uint32_t i = 0; i < l_skip; i++) {
        res *= (x + FpExt(Fp::one()));
        x *= x;
    }
    return res * Fp::one().mul_2exp_neg_n(l_skip);
}

__device__ __forceinline__ FpExt eval_eq_mle_ext(const FpExt *x, const FpExt *y, uint32_t len) {
    FpExt res = FpExt(Fp::one());
    for (uint32_t i = 0; i < len; i++) {
        FpExt mul = x[i] * y[i];
        res *= (FpExt(Fp::one()) - y[i] - x[i] + mul + mul);
    }
    return res;
}

__device__ __forceinline__ FpExtPair
eval_eq_rot_cube_ext(const FpExt *x, const FpExt *y, uint32_t len) {
    FpExt eq = FpExt(Fp::one());
    FpExt rot = FpExt(Fp::one());
    for (int32_t i = (int32_t)len - 1; i >= 0; i--) {
        FpExt one_minus_x = FpExt(Fp::one()) - x[i];
        FpExt one_minus_y = FpExt(Fp::one()) - y[i];
        FpExt xy = x[i] * y[i];
        rot = (x[i] - xy) * eq + (y[i] - xy) * rot;
        eq *= xy + xy + FpExt(Fp::one()) - x[i] - y[i];
    }
    return {eq, rot};
}

__device__ __forceinline__ FpExt eval_in_uni_ext(FpExt x, int32_t n, uint32_t l_skip) {
    FpExt res = FpExt(Fp::one());
    if (n < 0) {
        x = exp_power_of_2_ext(x, ((int32_t)l_skip) + n);
        for (uint32_t i = 0; i < -n; i++) {
            res *= x + FpExt(Fp::one());
            x *= x;
        }
        res *= Fp::one().mul_2exp_neg_n(-n);
    }
    return res;
}

__device__ __forceinline__ FpExt
eval_eq_prism_ext(const FpExt *x, const FpExt *y, uint32_t len, uint32_t l_skip) {
    return eval_eq_uni_ext(x[0], y[0], l_skip) * eval_eq_mle_ext(x + 1, y + 1, len - 1);
}

__device__ __forceinline__ FpExt
eval_rot_kernel_prism_ext(const FpExt *x, const FpExt *y, uint32_t len, uint32_t l_skip) {
    const Fp omega = TWO_ADIC_GENERATORS[l_skip];
    auto [eq_cube, rot_cube] = eval_eq_rot_cube_ext(x + 1, y + 1, len - 1);
    FpExt x0 = x[0];
    FpExt y0 = y[0] * omega;
    return (eval_eq_uni_ext(x0, y0, l_skip) * eq_cube) +
           (eval_eq_uni_at_one_ext(x0, l_skip) * eval_eq_uni_at_one_ext(y0, l_skip) *
            (rot_cube - eq_cube));
}
