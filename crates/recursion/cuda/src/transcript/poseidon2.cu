#include "fp.h"
#include "launcher.cuh"
#include "poseidon2-air/columns.cuh"
#include "poseidon2-air/params.cuh"
#include "poseidon2-air/tracegen.cuh"
#include "primitives/fp_array.cuh"
#include "primitives/trace_access.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cub/cub.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <driver_types.h>

struct Poseidon2Count {
    uint32_t perm;
    uint32_t compress;
};

struct Poseidon2CountCompose {
    __device__ __forceinline__ Poseidon2Count
    operator()(const Poseidon2Count &a, const Poseidon2Count &b) const {
        return {a.perm + b.perm, a.compress + b.compress};
    }
};

__global__ void fill_count_buffer(
    Poseidon2Count *d_counts,
    size_t num_prefix_perms,
    size_t num_compress_inputs,
    size_t num_suffix_perms
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_prefix_perms) {
        d_counts[idx] = {1, 0};
    } else if (idx < num_prefix_perms + num_compress_inputs) {
        d_counts[idx] = {0, 1};
    } else if (idx < num_prefix_perms + num_compress_inputs + num_suffix_perms) {
        d_counts[idx] = {1, 0};
    }
}

template <size_t WIDTH, typename PoseidonParams>
__global__ void cukernel_poseidon2_tracegen(
    Fp *d_trace,
    size_t trace_height,
    size_t trace_width,
    Fp *d_records,
    Poseidon2Count *d_counts,
    size_t num_records
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    using Poseidon2Row = poseidon2::Poseidon2Row<
        WIDTH,
        PoseidonParams::SBOX_DEGREE,
        PoseidonParams::SBOX_REGS,
        PoseidonParams::HALF_FULL_ROUNDS,
        PoseidonParams::PARTIAL_ROUNDS>;
#ifdef CUDA_DEBUG
    assert(Poseidon2Row::get_total_size() + 2 == trace_width);
#endif
    if (idx < trace_height) {
        Poseidon2Row row(d_trace + idx, trace_height);
        if (idx < num_records) {
            RowSlice state(d_records + idx * WIDTH, 1);
            poseidon2::generate_trace_row_for_perm(row, state);
            auto count = d_counts[idx];
            d_trace[idx + Poseidon2Row::get_total_size() * trace_height] = count.perm;
            d_trace[idx + (Poseidon2Row::get_total_size() + 1) * trace_height] = count.compress;
        } else {
            Fp dummy[Poseidon2Row::get_total_size()] = {0};
            RowSlice dummy_row(dummy, 1);
            poseidon2::generate_trace_row_for_perm(row, dummy_row);
            d_trace[idx + Poseidon2Row::get_total_size() * trace_height] = 0;
            d_trace[idx + (Poseidon2Row::get_total_size() + 1) * trace_height] = 0;
        }
    }
}

extern "C" int _poseidon2_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    Fp *d_records,
    Poseidon2Count *d_counts,
    size_t num_records,
    size_t sbox_regs
) {
    auto [grid, block] = kernel_launch_params(height);

    switch (sbox_regs) {
    case 1:
        cukernel_poseidon2_tracegen<16, Poseidon2ParamsS1><<<grid, block, 0, cudaStreamPerThread>>>(
            d_trace, height, width, d_records, d_counts, num_records
        );
        break;
    case 0:
        cukernel_poseidon2_tracegen<16, Poseidon2ParamsS0><<<grid, block, 0, cudaStreamPerThread>>>(
            d_trace, height, width, d_records, d_counts, num_records
        );
        break;
    default:
        return cudaErrorInvalidConfiguration;
    }

    return CHECK_KERNEL();
}

// Prepares d_num_records for use with sort reduce and stores the temporary buffer
// size necessary for both cub functions (i.e. sort and reduce).
extern "C" int _poseidon2_deduplicate_records_get_temp_bytes(
    Fp *d_records,
    Poseidon2Count *d_counts,
    size_t num_records,
    size_t *d_num_records,
    size_t *h_temp_bytes_out
) {
    FpArray<16> *d_records_fp16 = reinterpret_cast<FpArray<16> *>(d_records);

    size_t sort_storage_bytes = 0;
    cub::DeviceMergeSort::SortPairs(
        nullptr,
        sort_storage_bytes,
        d_records_fp16,
        d_counts,
        num_records,
        Fp16CompareOp(),
        cudaStreamPerThread
    );

    size_t reduce_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(
        nullptr,
        reduce_storage_bytes,
        d_records_fp16,
        d_records_fp16,
        d_counts,
        d_counts,
        d_num_records,
        Poseidon2CountCompose{},
        num_records,
        cudaStreamPerThread
    );

    *h_temp_bytes_out = std::max(sort_storage_bytes, reduce_storage_bytes);
    return CHECK_KERNEL();
}

// Reduces the records, removing duplicates and storing the number of times
// each occurs in d_counts. The number of records after reduction is stored
// into host pointer num_records. The value of temp_storage_bytes should be
// computed using _poseidon2_deduplicate_records_get_temp_bytes.
extern "C" int _poseidon2_deduplicate_records(
    Fp *d_records,
    Poseidon2Count *d_counts,
    size_t num_records,
    size_t *d_num_records,
    size_t num_prefix_perms,
    size_t num_compress_inputs,
    size_t num_suffix_perms,
    void *d_temp_storage,
    size_t temp_storage_bytes
) {
    auto [grid, block] = kernel_launch_params(num_records);
    fill_count_buffer<<<grid, block, 0, cudaStreamPerThread>>>(
        d_counts, num_prefix_perms, num_compress_inputs, num_suffix_perms
    );

    FpArray<16> *d_records_fp16 = reinterpret_cast<FpArray<16> *>(d_records);

    // TODO: We currently can't use DeviceRadixSort since each key is 64 bytes
    // which causes Fp16Decomposer usage to exceed shared memory. We need to
    // investigate better ways to sort, as merge sort is comparison-based.
    cub::DeviceMergeSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_records_fp16,
        d_counts,
        num_records,
        Fp16CompareOp(),
        cudaStreamPerThread
    );

    // Removes duplicate values from d_records, and stores the number of times
    // they occur in d_counts. The number of unique values is stored into
    // d_num_records.
    cub::DeviceReduce::ReduceByKey(
        d_temp_storage,
        temp_storage_bytes,
        d_records_fp16,
        d_records_fp16,
        d_counts,
        d_counts,
        d_num_records,
        Poseidon2CountCompose{},
        num_records,
        cudaStreamPerThread
    );

    return CHECK_KERNEL();
}
