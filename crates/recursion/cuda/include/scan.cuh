#pragma once

#include "fp.h"

#include <cstddef>
#include <driver_types.h>

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
