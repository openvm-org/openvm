#pragma once

#include "fpext.h"
#include "launcher.cuh"

#include <cassert>
#include <cstddef>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <vector_types.h>

// Affine map over FpExt: f(x) = a * x + b
struct AffineFpExt {
    FpExt a;
    FpExt b;
};

// Affine composition note:
// For affine maps f(x)=a1*x+b1 and g(x)=a2*x+b2:
//   (f ∘ g)(x) = f(g(x))  =>  a = a1*a2,  b = a1*b2 + b1
struct AffineCompose {
    __device__ __forceinline__ AffineFpExt
    operator()(const AffineFpExt &f, const AffineFpExt &g) const {
        // CUB InclusiveScanByKey computes: out[i] = op(out[i-1], in[i]).
        // Here we define op(prefix=f, current=g) := (g ∘ f), i.e. apply the current
        // affine AFTER the prefix affine. This is convenient for some recurrences of the
        // form x <- a*x + b as we scan forward in memory.
        return AffineFpExt{g.a * f.a, g.a * f.b + g.b};
    }
};

struct UInt2Equal {
    __device__ __forceinline__ bool operator()(uint2 a, uint2 b) const {
        return a.x == b.x && a.y == b.y;
    }
};

/*
 * Takes size-n AffineFpExt values and performs an in-place inclusive segmented scan,
 * where segments are defined by equality of adjacent keys in `d_keys`.
 *
 * Because `AffineCompose` is defined as op(prefix, current) = (current ∘ prefix),
 * for each i within a segment [s..t], this computes:
 *   out[i] = in[i] ∘ in[i-1] ∘ ... ∘ in[s]
 *
 * `out[i].b` is the composed affine applied to 0 (i.e. the folded constant term).
 *
 * Notes:
 * - `d_keys` is `uint2` (e.g. pack (proof_idx, sort_idx) into a single key).
 * - Requires a temporary buffer allocated by the caller (see helper below).
 */
__host__ inline int affine_scan_by_key(
    const uint2 *d_keys,
    AffineFpExt *d_affines,
    size_t n,
    void *d_temp,
    size_t temp_n,
    cudaStream_t stream = cudaStreamPerThread
) {
    if (!d_keys || !d_affines || n == 0) {
        return 0;
    }
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveScanByKey(
        nullptr, temp_bytes, d_keys, d_affines, d_affines, AffineCompose{}, n, UInt2Equal{}, stream
    );
    assert(temp_bytes <= temp_n);
    cub::DeviceScan::InclusiveScanByKey(
        d_temp, temp_bytes, d_keys, d_affines, d_affines, AffineCompose{}, n, UInt2Equal{}, stream
    );
    return CHECK_KERNEL();
}

__host__ inline int get_affine_scan_by_key_temp_bytes(
    const uint2 *d_keys,
    AffineFpExt *d_affines,
    size_t n,
    size_t &temp_bytes_out,
    cudaStream_t stream = cudaStreamPerThread
) {
    if (!d_keys || !d_affines || n == 0) {
        temp_bytes_out = 0;
        return 0;
    }
    cub::DeviceScan::InclusiveScanByKey(
        nullptr,
        temp_bytes_out,
        d_keys,
        d_affines,
        d_affines,
        AffineCompose{},
        n,
        UInt2Equal{},
        stream
    );
    return CHECK_KERNEL();
}
