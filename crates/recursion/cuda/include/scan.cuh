#pragma once

#include "fp.h"
#include "launcher.cuh"

#include <cassert>
#include <cstddef>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <thrust/iterator/reverse_iterator.h>

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
 * Takes a size-n Fp array and performs an in-place inclusive suffix scan.
 * Concretely, this computes, for each i:
 *   out[i] = sum_{j = i}^{n-1} in[j]
 *
 * Implemented by scanning over `thrust::reverse_iterator`.
 */
__host__ int suffix_scan(
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
template <typename KeyT, typename KeyEqualOp>
__host__ inline int prefix_scan_by_key(
    const KeyT *d_keys,
    Fp *d_arr,
    size_t n,
    void *d_temp,
    size_t temp_n,
    KeyEqualOp key_equal,
    cudaStream_t stream = cudaStreamPerThread
) {
    if (!d_keys || !d_arr || n == 0) {
        return 0;
    }
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveScanByKey(
        nullptr, temp_bytes, d_keys, d_arr, d_arr, FpAdd{}, n, key_equal, stream
    );
    assert(temp_bytes <= temp_n);
    cub::DeviceScan::InclusiveScanByKey(
        d_temp, temp_bytes, d_keys, d_arr, d_arr, FpAdd{}, n, key_equal, stream
    );
    return CHECK_KERNEL();
}

/*
 * Segmented inclusive suffix scan by key, where segments are defined by equality of
 * adjacent keys in `d_keys`.
 *
 * Concretely, for each i:
 *   out[i] = sum_{j = i}^{segment_end(i)} in[j]
 *
 * Implemented via CUB's InclusiveScanByKey over `thrust::reverse_iterator`.
 */
template <typename KeyT, typename KeyEqualOp>
__host__ inline int suffix_scan_by_key(
    const KeyT *d_keys,
    Fp *d_arr,
    size_t n,
    void *d_temp,
    size_t temp_n,
    KeyEqualOp key_equal,
    cudaStream_t stream = cudaStreamPerThread
) {
    if (!d_keys || !d_arr || n == 0) {
        return 0;
    }
    auto k_rbegin = thrust::make_reverse_iterator(d_keys + n);
    auto v_rbegin = thrust::make_reverse_iterator(d_arr + n);
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveScanByKey(
        nullptr, temp_bytes, k_rbegin, v_rbegin, v_rbegin, FpAdd{}, n, key_equal, stream
    );
    assert(temp_bytes <= temp_n);
    cub::DeviceScan::InclusiveScanByKey(
        d_temp, temp_bytes, k_rbegin, v_rbegin, v_rbegin, FpAdd{}, n, key_equal, stream
    );
    return CHECK_KERNEL();
}

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
template <size_t NUM_ARRAYS, typename KeyT, typename KeyEqualOp>
__host__ inline int prefix_scan_by_key_n_arrays(
    const KeyT *d_keys,
    Fp *d_arr,
    size_t n,
    void *d_temp,
    size_t temp_n,
    KeyEqualOp key_equal,
    cudaStream_t stream = cudaStreamPerThread
) {
    if (!d_keys || !d_arr || n == 0) {
        return 0;
    }
    for (size_t i = 0; i < NUM_ARRAYS; i++) {
        int err = prefix_scan_by_key(d_keys, d_arr + i * n, n, d_temp, temp_n, key_equal, stream);
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
 * Runs `suffix_scan` once per array (N total scans).
 */
template <size_t NUM_ARRAYS>
__host__ inline int suffix_scan_n_arrays(
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
        int err = suffix_scan(d_arr + i * n, n, d_temp, temp_n, stream);
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
 * Runs `suffix_scan_by_key` once per array (N total scans), using the same keys.
 */
template <size_t NUM_ARRAYS, typename KeyT, typename KeyEqualOp>
__host__ inline int suffix_scan_by_key_n_arrays(
    const KeyT *d_keys,
    Fp *d_arr,
    size_t n,
    void *d_temp,
    size_t temp_n,
    KeyEqualOp key_equal,
    cudaStream_t stream = cudaStreamPerThread
) {
    if (!d_keys || !d_arr || n == 0) {
        return 0;
    }
    for (size_t i = 0; i < NUM_ARRAYS; i++) {
        int err = suffix_scan_by_key(d_keys, d_arr + i * n, n, d_temp, temp_n, key_equal, stream);
        if (err) {
            return err;
        }
    }
    return 0;
}

// ============================================================================
// Temp-bytes helpers
// ============================================================================

__host__ int prefix_scan_temp_bytes(
    Fp *d_arr,
    size_t n,
    size_t &temp_bytes_out,
    cudaStream_t stream = cudaStreamPerThread
);

template <typename KeyT, typename KeyEqualOp>
__host__ inline int prefix_scan_by_key_temp_bytes(
    const KeyT *d_keys,
    Fp *d_arr,
    size_t n,
    size_t &temp_bytes_out,
    KeyEqualOp key_equal,
    cudaStream_t stream = cudaStreamPerThread
) {
    if (!d_keys || !d_arr || n == 0) {
        temp_bytes_out = 0;
        return 0;
    }
    cub::DeviceScan::InclusiveScanByKey(
        nullptr, temp_bytes_out, d_keys, d_arr, d_arr, FpAdd{}, n, key_equal, stream
    );
    return CHECK_KERNEL();
}

__host__ int suffix_scan_temp_bytes(
    Fp *d_arr,
    size_t n,
    size_t &temp_bytes_out,
    cudaStream_t stream = cudaStreamPerThread
);

template <typename KeyT, typename KeyEqualOp>
__host__ inline int suffix_scan_by_key_temp_bytes(
    const KeyT *d_keys,
    Fp *d_arr,
    size_t n,
    size_t &temp_bytes_out,
    KeyEqualOp key_equal,
    cudaStream_t stream = cudaStreamPerThread
) {
    if (!d_keys || !d_arr || n == 0) {
        temp_bytes_out = 0;
        return 0;
    }
    auto k_rbegin = thrust::make_reverse_iterator(d_keys + n);
    auto v_rbegin = thrust::make_reverse_iterator(d_arr + n);
    cub::DeviceScan::InclusiveScanByKey(
        nullptr, temp_bytes_out, k_rbegin, v_rbegin, v_rbegin, FpAdd{}, n, key_equal, stream
    );
    return CHECK_KERNEL();
}

template <size_t NUM_ARRAYS>
__host__ inline int prefix_scan_n_arrays_temp_bytes(
    Fp *d_arr,
    size_t n,
    size_t &temp_bytes_out,
    cudaStream_t stream = cudaStreamPerThread
) {
    (void)NUM_ARRAYS;
    return prefix_scan_temp_bytes(d_arr, n, temp_bytes_out, stream);
}

template <size_t NUM_ARRAYS, typename KeyT, typename KeyEqualOp>
__host__ inline int prefix_scan_by_key_n_arrays_temp_bytes(
    const KeyT *d_keys,
    Fp *d_arr,
    size_t n,
    size_t &temp_bytes_out,
    KeyEqualOp key_equal,
    cudaStream_t stream = cudaStreamPerThread
) {
    (void)NUM_ARRAYS;
    return prefix_scan_by_key_temp_bytes(d_keys, d_arr, n, temp_bytes_out, key_equal, stream);
}

template <size_t NUM_ARRAYS>
__host__ inline int suffix_scan_n_arrays_temp_bytes(
    Fp *d_arr,
    size_t n,
    size_t &temp_bytes_out,
    cudaStream_t stream = cudaStreamPerThread
) {
    (void)NUM_ARRAYS;
    return suffix_scan_temp_bytes(d_arr, n, temp_bytes_out, stream);
}

template <size_t NUM_ARRAYS, typename KeyT, typename KeyEqualOp>
__host__ inline int suffix_scan_by_key_n_arrays_temp_bytes(
    const KeyT *d_keys,
    Fp *d_arr,
    size_t n,
    size_t &temp_bytes_out,
    KeyEqualOp key_equal,
    cudaStream_t stream = cudaStreamPerThread
) {
    (void)NUM_ARRAYS;
    return suffix_scan_by_key_temp_bytes(d_keys, d_arr, n, temp_bytes_out, key_equal, stream);
}
