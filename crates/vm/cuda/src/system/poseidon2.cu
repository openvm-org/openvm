#include "launcher.cuh"
#include "poseidon2-air/params.cuh"
#include "poseidon2-air/tracegen.cuh"
#include "primitives/fp_array.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include <cstdint>
#include <cub/cub.cuh>

template <size_t WIDTH, typename PoseidonParams>
__global__ void cukernel_system_poseidon2_tracegen(
    Fp *d_trace,
    size_t trace_height,
    size_t trace_width,
    Fp *d_records,
    uint32_t *d_counts,
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
    assert(Poseidon2Row::get_total_size() + 1 == trace_width);
#endif
    if (idx < trace_height) {
        Poseidon2Row row(d_trace + idx, trace_height);
        if (idx < num_records) {
            RowSlice state(d_records + idx * WIDTH, 1);
            poseidon2::generate_trace_row_for_perm(row, state);

            d_trace[idx + Poseidon2Row::get_total_size() * trace_height] = d_counts[idx];
        } else {
            Fp dummy[Poseidon2Row::get_total_size()] = {0};
            RowSlice dummy_row(dummy, 1);
            poseidon2::generate_trace_row_for_perm(row, dummy_row);

            d_trace[idx + Poseidon2Row::get_total_size() * trace_height] = 0;
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
        cukernel_system_poseidon2_tracegen<16, Poseidon2ParamsS1>
            <<<grid, block, 0, cudaStreamPerThread>>>(
                d_trace, height, width, d_records, d_counts, num_records
            );
        break;
    case 0:
        cukernel_system_poseidon2_tracegen<16, Poseidon2ParamsS0>
            <<<grid, block, 0, cudaStreamPerThread>>>(
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
extern "C" int _system_poseidon2_deduplicate_records_get_temp_bytes(
    Fp *d_records,
    uint32_t *d_counts,
    size_t num_records,
    size_t *d_num_records,
    size_t *h_temp_bytes_out
) {
    auto [grid, block] = kernel_launch_params(num_records);
    FpArray<16> *d_records_fp16 = reinterpret_cast<FpArray<16> *>(d_records);

    // We want to sort and reduce the raw records, keeping track of how many
    // each occurs in d_counts. To prepare for reduce we need to a) fill
    // d_counts with 1s, and b) group keys together using sort. Note we do
    // b) in the kernel below.
    fill_buffer<uint32_t><<<grid, block>>>(d_counts, 1, num_records);

    size_t sort_storage_bytes = 0;
    cub::DeviceMergeSort::SortKeys(
        nullptr,
        sort_storage_bytes,
        d_records_fp16,
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
        std::plus(),
        num_records,
        cudaStreamPerThread
    );

    *h_temp_bytes_out = std::max(sort_storage_bytes, reduce_storage_bytes);
    return CHECK_KERNEL();
}

// Reduces the records, removing duplicates and storing the number of times
// each occurs in d_counts. The number of records after reduction is stored
// into host pointer num_records. The value of temp_storage_bytes should be
// computed using _system_poseidon2_deduplicate_records_get_temp_bytes.
extern "C" int _system_poseidon2_deduplicate_records(
    Fp *d_records,
    uint32_t *d_counts,
    size_t num_records,
    size_t *d_num_records,
    void *d_temp_storage,
    size_t temp_storage_bytes
) {
    FpArray<16> *d_records_fp16 = reinterpret_cast<FpArray<16> *>(d_records);

    // TODO: We currently can't use DeviceRadixSort since each key is 64 bytes
    // which causes Fp16Decomposer usage to exceed shared memory. We need to
    // investigate better ways to sort, as merge sort is comparison-based.
    cub::DeviceMergeSort::SortKeys(
        d_temp_storage,
        temp_storage_bytes,
        d_records_fp16,
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
        std::plus(),
        num_records,
        cudaStreamPerThread
    );

    return CHECK_KERNEL();
}
