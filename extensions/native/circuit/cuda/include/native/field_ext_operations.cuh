#pragma once

#include "fp.h"
#include "primitives/constants.h"

using namespace native;

template <typename F> struct FieldExtElement {
    F el[EXT_DEG];

    __device__ __forceinline__ FieldExtElement() : el{0} {}
    __device__ __forceinline__ FieldExtElement(const F el[EXT_DEG]) {
#pragma unroll
        for (size_t i = 0; i < EXT_DEG; i++) {
            this->el[i] = el[i];
        }
    }

    __device__ __forceinline__ const F &operator[](size_t idx) const { return el[idx]; }
    __device__ __forceinline__ F &operator[](size_t idx) { return el[idx]; }
};

struct FieldExtOperations {
    static __device__ __forceinline__ FieldExtElement<Fp> add(
        const FieldExtElement<Fp> &x,
        const FieldExtElement<Fp> &y
    ) {
        FieldExtElement<Fp> z;
#pragma unroll
        for (size_t i = 0; i < EXT_DEG; i++) {
            z[i] = x[i] + y[i];
        }
        return z;
    }

    static __device__ __forceinline__ FieldExtElement<Fp> add(
        const FieldExtElement<Fp> &x,
        const Fp &y
    ) {
        FieldExtElement<Fp> z(x);
        z[0] += y;
        return z;
    }

    static __device__ __forceinline__ FieldExtElement<Fp> subtract(
        const FieldExtElement<Fp> &x,
        const FieldExtElement<Fp> &y
    ) {
        FieldExtElement<Fp> z;
#pragma unroll
        for (size_t i = 0; i < EXT_DEG; i++) {
            z[i] = x[i] - y[i];
        }
        return z;
    }

    static __device__ __forceinline__ FieldExtElement<Fp> multiply(
        const FieldExtElement<Fp> &x,
        const FieldExtElement<Fp> &y
    ) {
        FieldExtElement<Fp> z;
        z[0] = x[0] * y[0] + (x[1] * y[3] + x[2] * y[2] + x[3] * y[1]) * Fp(BETA);
        z[1] = x[0] * y[1] + x[1] * y[0] + (x[2] * y[3] + x[3] * y[2]) * Fp(BETA);
        z[2] = x[0] * y[2] + x[1] * y[1] + x[2] * y[0] + (x[3] * y[3]) * Fp(BETA);
        z[3] = x[0] * y[3] + x[1] * y[2] + x[2] * y[1] + x[3] * y[0];
        return z;
    }

    static __device__ __forceinline__ FieldExtElement<Fp> divide(
        const FieldExtElement<Fp> &x,
        const FieldExtElement<Fp> &y
    ) {
        return multiply(x, invert(y));
    }

    static __device__ __forceinline__ FieldExtElement<Fp> invert(const FieldExtElement<Fp> &a) {
        FieldExtElement<Fp> z;
        Fp b0 = a[0] * a[0] - Fp(BETA) * (Fp(2) * a[1] * a[3] - a[2] * a[2]);
        Fp b2 = Fp(2) * a[0] * a[2] - a[1] * a[1] - Fp(BETA) * a[3] * a[3];

        Fp c = b0 * b0 - Fp(BETA) * b2 * b2;
        Fp inv_c = inv(c);

        b0 *= inv_c;
        b2 *= inv_c;
        z[0] = a[0] * b0 - a[2] * b2 * Fp(BETA);
        z[1] = -a[1] * b0 + a[3] * b2 * Fp(BETA);
        z[2] = -a[0] * b2 + a[2] * b0;
        z[3] = a[1] * b2 - a[3] * b0;
        return z;
    }
};