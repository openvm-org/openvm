#pragma once

#include "fpext.h"
#include "launcher.cuh"
#include "ptr_array.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <vector_types.h>

/*
 * Affine map over FpExt: f(x) = a * x + b
 */
struct AffineFpExt {
    FpExt a;
    FpExt b;
};

struct FpExtWithTidx {
    FpExt a;
    uint32_t tidx;
};

/*
 * For affine maps f(x) = a1 * x + b1 and g(x) = a2 * x + b2:
 *   (f ∘ g)(x) = f(g(x)) => a = a1 * a2,  b = a1 * b2 + b1
 */
struct AffineCompose {
    __device__ __forceinline__ AffineFpExt
    operator()(const AffineFpExt &f, const AffineFpExt &g) const {
        // CUB InclusiveScanByKey computes out[i] = op(out[i-1], in[i]). Here we define
        // op(prefix=f, current=g) := (g ∘ f), i.e. apply the current affine AFTER the
        // prefix affine. This is convenient for some recurrences of the form x <- a * x
        // + b as we scan forward in memory.
        return AffineFpExt{g.a * f.a, g.a * f.b + g.b};
    }
};

struct UInt2Equal {
    __device__ __forceinline__ bool operator()(uint2 a, uint2 b) const {
        return a.x == b.x && a.y == b.y;
    }
};

/*
 * Kernel to set up a reverse (i.e. suffix) segmented affine scan given the keys,
 * values, and index bounds for both key tuples.
 */
template <size_t NUM_X_SEGMENTS>
__global__ void reverse_affines_setup(
    const uint2 *__restrict__ keys,
    AffineFpExt *__restrict__ affines,
    const FpExtWithTidx *__restrict__ a_constants,
    const FpExt *__restrict__ b_values,
    const Array<uint32_t, NUM_X_SEGMENTS> x_bounds,
    const PtrArray<uint32_t, NUM_X_SEGMENTS> y_bounds,
    uint32_t n
) {
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= n) {
        return;
    }

    auto [x, y] = keys[global_idx];
    uint32_t start_idx_for_x = (x == 0) ? 0 : x_bounds[x - 1];
    uint32_t x_segment_idx = global_idx - start_idx_for_x;

    uint32_t start_idx_for_y = (y == 0) ? 0 : y_bounds[x][y - 1];
    uint32_t y_segment_idx = x_segment_idx - start_idx_for_y;

    uint32_t end_idx_for_y = y_bounds[x][y];
    uint32_t rev_idx = start_idx_for_x + end_idx_for_y - 1 - y_segment_idx;

    auto [a, _] = a_constants[x];
    affines[rev_idx] = {a, b_values[global_idx]};
}

/*
 * Takes size-n AffineFpExt values and performs an in-place inclusive segmented scan,
 * where segments are defined by equality of adjacent keys in `d_keys`.
 *
 * Because `AffineCompose` is defined as op(prefix, current) = (current ∘ prefix),
 * for each i within a segment [s..t], this computes:
 *   out[i] = in[i] ∘ in[i-1] ∘ ... ∘ in[s]
 *
 * `out[i].b` is the composed affine applied to a (i.e. the folded constant term).
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
