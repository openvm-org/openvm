#include "fp.h"
#include "launcher.cuh"
#include "scan.cuh"

#include <cassert>
#include <cstddef>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <driver_types.h>

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

extern "C" int _get_fp_prefix_scan_temp_bytes(Fp *d_arr, size_t n, size_t *h_temp_n) {
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d_arr, d_arr, n, cudaStreamPerThread);
    *h_temp_n = temp_bytes;
    return CHECK_KERNEL();
}

extern "C" int _fp_prefix_scan(Fp *d_arr, size_t n, void *d_temp, size_t temp_n) {
    prefix_scan(d_arr, n, d_temp, temp_n, cudaStreamPerThread);
    return CHECK_KERNEL();
}
