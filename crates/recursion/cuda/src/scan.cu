#include "fp.h"
#include "launcher.cuh"
#include "scan.cuh"

#include <cassert>
#include <cstddef>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <thrust/iterator/detail/reverse_iterator.inl>

__host__ int prefix_scan(Fp *d_arr, size_t n, void *d_temp, size_t temp_n, cudaStream_t stream) {
    if (!d_arr || n == 0) {
        return 0;
    }
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d_arr, d_arr, n, stream);
    assert(temp_bytes <= temp_n);
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_arr, d_arr, n, stream);
    return CHECK_KERNEL();
}

__host__ int suffix_scan(Fp *d_arr, size_t n, void *d_temp, size_t temp_n, cudaStream_t stream) {
    if (!d_arr || n == 0) {
        return 0;
    }
    auto rbegin = thrust::make_reverse_iterator(d_arr + n);
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, rbegin, rbegin, n, stream);
    assert(temp_bytes <= temp_n);
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, rbegin, rbegin, n, stream);
    return CHECK_KERNEL();
}

__host__ int prefix_scan_temp_bytes(
    Fp *d_arr,
    size_t n,
    size_t &temp_bytes_out,
    cudaStream_t stream
) {
    if (!d_arr || n == 0) {
        temp_bytes_out = 0;
        return 0;
    }
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes_out, d_arr, d_arr, n, stream);
    return CHECK_KERNEL();
}

__host__ int suffix_scan_temp_bytes(
    Fp *d_arr,
    size_t n,
    size_t &temp_bytes_out,
    cudaStream_t stream
) {
    if (!d_arr || n == 0) {
        temp_bytes_out = 0;
        return 0;
    }
    auto rbegin = thrust::make_reverse_iterator(d_arr + n);
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes_out, rbegin, rbegin, n, stream);
    return CHECK_KERNEL();
}
