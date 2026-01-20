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

__host__ int prefix_scan_by_key(
    const Fp *d_keys,
    Fp *d_arr,
    size_t n,
    void *d_temp,
    size_t temp_n,
    cudaStream_t stream
) {
    if (!d_keys || !d_arr || n == 0) {
        return 0;
    }
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveScanByKey(
        nullptr, temp_bytes, d_keys, d_arr, d_arr, FpAdd{}, n, FpEqual{}, stream
    );
    assert(temp_bytes <= temp_n);
    cub::DeviceScan::InclusiveScanByKey(
        d_temp, temp_bytes, d_keys, d_arr, d_arr, FpAdd{}, n, FpEqual{}, stream
    );
    return CHECK_KERNEL();
}
