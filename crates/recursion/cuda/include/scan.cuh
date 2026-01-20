#pragma once

#include "fp.h"
#include "launcher.cuh"

#include <cassert>
#include <cstddef>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <driver_types.h>

struct FpAdd {
    __device__ __forceinline__ Fp operator()(const Fp &a, const Fp &b) const { return a + b; }
};

struct FpEqual {
    __device__ __forceinline__ bool operator()(const Fp &a, const Fp &b) const { return a == b; }
};

/* 
 * Takes a size-n Fp array and performs an in-place inclusive prefix scan. Note
 * this requires a temporary buffer, which you should allocate in Rust (due to
 * VPMM). To see how to get the minimum temporary storage amount in bytes, see
 * external function _get_prefix_scan_temp_bytes in scan.cu.
 */
__host__ int prefix_scan(
    Fp *d_arr,
    size_t n,
    void *d_temp,
    size_t temp_n,
    cudaStream_t stream = cudaStreamPerThread
);

/*
 * Takes size-n `Fp` values and performs an in-place inclusive segmented prefix scan,
 * where segments are defined by equality of adjacent keys in `d_keys`.
 *
 * Concretely, this computes, for each i:
 *   out[i] = sum_{j = segment_start(i)}^i in[j]
 *
 * This is implemented via CUB's InclusiveScanByKey.
 */
__host__ int prefix_scan_by_key(
    const Fp *d_keys,
    Fp *d_arr,
    size_t n,
    void *d_temp,
    size_t temp_n,
    cudaStream_t stream = cudaStreamPerThread
);

/*
 * Convenience helper for N arrays stored in SoA form:
 *   d_arr = [ arr0[0..n), arr1[0..n), ..., arr(N-1)[0..n) ].
 *
 * Runs `prefix_scan` once per array (N total scans).
 */
template <size_t NUM_ARRAYS>
__host__ inline int prefix_scan_n_arrays(
    Fp *d_arr,
    size_t n,
    void *d_temp,
    size_t temp_n,
    cudaStream_t stream = cudaStreamPerThread
) {
    if (!d_arr || n == 0) {
        return 0;
    }
    for (size_t i = 0; i < NUM_ARRAYS; i++) {
        int err = prefix_scan(d_arr + i * n, n, d_temp, temp_n, stream);
        if (err) {
            return err;
        }
    }
    return 0;
}

/*
 * Convenience helper for N arrays stored in SoA form:
 *   d_arr = [ arr0[0..n), arr1[0..n), ..., arr(N-1)[0..n) ].
 *
 * Runs `prefix_scan_by_key` once per array (N total scans), using the same keys.
 */
template <size_t NUM_ARRAYS>
__host__ inline int prefix_scan_by_key_n_arrays(
    const Fp *d_keys,
    Fp *d_arr,
    size_t n,
    void *d_temp,
    size_t temp_n,
    cudaStream_t stream = cudaStreamPerThread
) {
    if (!d_keys || !d_arr || n == 0) {
        return 0;
    }
    for (size_t i = 0; i < NUM_ARRAYS; i++) {
        int err = prefix_scan_by_key(d_keys, d_arr + i * n, n, d_temp, temp_n, stream);
        if (err) {
            return err;
        }
    }
    return 0;
}

// ============================================================================
// Temp-bytes helpers (header-only, no extern "C")
// ============================================================================

__host__ inline int get_fp_prefix_scan_temp_bytes(
    Fp *d_arr,
    size_t n,
    size_t &temp_bytes_out,
    cudaStream_t stream = cudaStreamPerThread
) {
    if (!d_arr || n == 0) {
        temp_bytes_out = 0;
        return 0;
    }
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes_out, d_arr, d_arr, n, stream);
    return CHECK_KERNEL();
}

__host__ inline int get_fp_prefix_scan_by_key_temp_bytes(
    const Fp *d_keys,
    Fp *d_arr,
    size_t n,
    size_t &temp_bytes_out,
    cudaStream_t stream = cudaStreamPerThread
) {
    if (!d_keys || !d_arr || n == 0) {
        temp_bytes_out = 0;
        return 0;
    }
    cub::DeviceScan::InclusiveScanByKey(
        nullptr, temp_bytes_out, d_keys, d_arr, d_arr, FpAdd{}, n, FpEqual{}, stream
    );
    return CHECK_KERNEL();
}

template <size_t NUM_ARRAYS>
__host__ inline int get_fp_prefix_scan_n_arrays_temp_bytes(
    Fp *d_arr,
    size_t n,
    size_t &temp_bytes_out,
    cudaStream_t stream = cudaStreamPerThread
) {
    (void)NUM_ARRAYS;
    return get_fp_prefix_scan_temp_bytes(d_arr, n, temp_bytes_out, stream);
}

template <size_t NUM_ARRAYS>
__host__ inline int get_fp_prefix_scan_by_key_n_arrays_temp_bytes(
    const Fp *d_keys,
    Fp *d_arr,
    size_t n,
    size_t &temp_bytes_out,
    cudaStream_t stream = cudaStreamPerThread
) {
    (void)NUM_ARRAYS;
    return get_fp_prefix_scan_by_key_temp_bytes(d_keys, d_arr, n, temp_bytes_out, stream);
}
