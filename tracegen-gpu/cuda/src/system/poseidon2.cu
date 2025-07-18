#include "launcher.cuh"
#include "poseidon2/fp_array.cuh"
#include "poseidon2/tracegen.cuh"
#include "trace_access.h"
#include "utils.cuh"
#include <cstdint>
#include <cub/cub.cuh>

template <
    size_t WIDTH,
    size_t SBOX_DEGREE,
    size_t SBOX_REGS,
    size_t HALF_FULL_ROUNDS,
    size_t PARTIAL_ROUNDS>
__global__ void cukernel_system_poseidon2_tracegen(
    Fp *d_trace,
    size_t trace_height,
    size_t trace_width,
    Fp *d_records,
    uint32_t *d_counts,
    size_t num_records
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_records) {
        using Poseidon2Row = poseidon2::
            Poseidon2Row<WIDTH, SBOX_DEGREE, SBOX_REGS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>;
        assert(Poseidon2Row::get_total_size() + 1 == trace_width);

        Poseidon2Row row(d_trace + idx, trace_height);
        RowSlice state(d_records + idx * WIDTH, 1);
        poseidon2::generate_trace_row_for_perm<
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS>(row, state);

        d_trace[idx + Poseidon2Row::get_total_size() * trace_height] = d_counts[idx];
    } else if (idx < trace_height) {
        RowSlice row(d_trace + idx, trace_height);
        for (size_t i = 0; i < trace_width; ++i) {
            row[i] = Fp::zero();
        }
    }
}

extern "C" int _system_poseidon2_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    Fp *d_records,
    uint32_t *d_counts,
    size_t num_records,
    size_t sbox_regs
) {
    auto [grid, block] = kernel_launch_params(height);

    switch (sbox_regs) {
    case 1:
        cukernel_system_poseidon2_tracegen<16, 7, 1, 4, 13>
            <<<grid, block, 0, cudaStreamPerThread>>>(
                d_trace, height, width, d_records, d_counts, num_records
            );
        break;
    case 0:
        cukernel_system_poseidon2_tracegen<16, 7, 0, 4, 13>
            <<<grid, block, 0, cudaStreamPerThread>>>(
                d_trace, height, width, d_records, d_counts, num_records
            );
        break;
    default:
        return cudaErrorInvalidConfiguration;
    }

    return cudaGetLastError();
}

// Reduces the records, removing duplicates and storing the number of times
// each occurs in d_counts. The number of records after reduction is stored
// into host pointer num_records.
extern "C" int _system_poseidon2_deduplicate_records(
    Fp *d_records,
    uint32_t *d_counts,
    size_t *num_records
) {
    auto [grid, block] = kernel_launch_params(*num_records);
    FpArray<16> *d_records_fp16 = reinterpret_cast<FpArray<16> *>(d_records);
    size_t *d_num_records;

    // We want to sort and reduce the raw records, keeping track of how many
    // each occurs in d_counts. To prepare for reduce we need to a) allocate
    // d_num_records, b) fill d_counts with 1s, and c) group keys together
    // using sort.
    cudaMallocAsync(&d_num_records, sizeof(size_t), cudaStreamPerThread);
    cudaMemcpyAsync(
        d_num_records, num_records, sizeof(size_t), cudaMemcpyHostToDevice, cudaStreamPerThread
    );
    fill_buffer<uint32_t><<<grid, block, 0, cudaStreamPerThread>>>(d_counts, 1, *num_records);

    size_t sort_storage_bytes = 0;
    cub::DeviceMergeSort::SortKeys(
        nullptr,
        sort_storage_bytes,
        d_records_fp16,
        *num_records,
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
        std::plus(),
        *num_records,
        cudaStreamPerThread
    );

    size_t temp_storage_bytes = std::max(sort_storage_bytes, reduce_storage_bytes);
    void *d_temp_storage = nullptr;
    cudaMallocAsync(&d_temp_storage, temp_storage_bytes, cudaStreamPerThread);

    // TODO: We currently can't use DeviceRadixSort since each key is 64 bytes
    // which causes Fp16Decomposer usage to exceed shared memory. We need to
    // investigate better ways to sort, as merge sort is comparison-based.
    cub::DeviceMergeSort::SortKeys(
        d_temp_storage,
        temp_storage_bytes,
        d_records_fp16,
        *num_records,
        Fp16CompareOp(),
        cudaStreamPerThread
    );

    // Removes duplicate values from d_records, and stores the number of times
    // they occur in d_counts. The number of unique values is stored into
    // d_num_records. We allow cub to use d_trace as temporary storage, which
    // in release we assume to be much more than the space required.
    cub::DeviceReduce::ReduceByKey(
        d_temp_storage,
        temp_storage_bytes,
        d_records_fp16,
        d_records_fp16,
        d_counts,
        d_counts,
        d_num_records,
        std::plus(),
        *num_records,
        cudaStreamPerThread
    );

    cudaMemcpyAsync(
        num_records, d_num_records, sizeof(size_t), cudaMemcpyDeviceToHost, cudaStreamPerThread
    );
    cudaFreeAsync(d_num_records, cudaStreamPerThread);
    cudaFreeAsync(d_temp_storage, cudaStreamPerThread);
    return cudaGetLastError();
}
