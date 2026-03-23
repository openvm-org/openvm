#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cub/cub.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <driver_types.h>

#include "def_poseidon2_buffer.cuh"
#include "fp.h"
#include "launcher.cuh"
#include "poseidon2-air/columns.cuh"
#include "poseidon2-air/params.cuh"
#include "poseidon2-air/tracegen.cuh"
#include "primitives/fp_array.cuh"
#include "primitives/trace_access.h"

using namespace deferral;

struct DeferralPoseidon2CountCompose {
    __device__ __forceinline__ DeferralPoseidon2Count
    operator()(const DeferralPoseidon2Count &a, const DeferralPoseidon2Count &b) const {
        return {a.compress_mult + b.compress_mult, a.capacity_mult + b.capacity_mult};
    }
};

template <typename T, typename PoseidonParams> struct DeferralPoseidon2Cols {
    poseidon2::Poseidon2SubCols<
        T,
        16,
        Poseidon2DefaultParams::SBOX_DEGREE,
        PoseidonParams::SBOX_REGS,
        Poseidon2DefaultParams::HALF_FULL_ROUNDS,
        Poseidon2DefaultParams::PARTIAL_ROUNDS>
        inner;
    T compress_mult;
    T capacity_mult;
};

template <size_t WIDTH, typename PoseidonParams>
__global__ void cukernel_deferral_poseidon2_tracegen(
    Fp *d_trace,
    size_t trace_height,
    size_t trace_width,
    Fp *d_records,
    DeferralPoseidon2Count *d_counts,
    size_t num_records
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    using Poseidon2Row = poseidon2::Poseidon2Row<
        WIDTH,
        PoseidonParams::SBOX_DEGREE,
        PoseidonParams::SBOX_REGS,
        PoseidonParams::HALF_FULL_ROUNDS,
        PoseidonParams::PARTIAL_ROUNDS>;

#ifdef CUDA_DEBUG
    assert(sizeof(DeferralPoseidon2Cols<uint8_t, PoseidonParams>) == trace_width);
    assert(Poseidon2Row::get_total_size() + 2 == trace_width);
#endif

    if (idx >= trace_height) {
        return;
    }

    RowSlice row(d_trace + idx, trace_height);
    Poseidon2Row poseidon2_row(row);
    constexpr size_t compress_mult_idx = Poseidon2Row::get_total_size();
    constexpr size_t capacity_mult_idx = compress_mult_idx + 1;

    if (idx < num_records) {
        RowSlice state(d_records + idx * WIDTH, 1);
        poseidon2::generate_trace_row_for_perm(poseidon2_row, state);

        const auto count = d_counts[idx];
        row.write(compress_mult_idx, Fp(count.compress_mult));
        row.write(capacity_mult_idx, Fp(count.capacity_mult));
    } else {
        Fp zero_state[WIDTH] = {0};
        RowSlice zero_state_row(zero_state, 1);
        poseidon2::generate_trace_row_for_perm(poseidon2_row, zero_state_row);

        row.write(compress_mult_idx, Fp::zero());
        row.write(capacity_mult_idx, Fp::zero());
    }
}

extern "C" int _deferral_poseidon2_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    Fp *d_records,
    DeferralPoseidon2Count *d_counts,
    size_t num_records,
    size_t sbox_regs
) {
    auto [grid, block] = kernel_launch_params(height);

    switch (sbox_regs) {
    case 1:
        cukernel_deferral_poseidon2_tracegen<16, Poseidon2ParamsS1>
            <<<grid, block, 0, cudaStreamPerThread>>>(
                d_trace, height, width, d_records, d_counts, num_records
            );
        break;
    case 0:
        cukernel_deferral_poseidon2_tracegen<16, Poseidon2ParamsS0>
            <<<grid, block, 0, cudaStreamPerThread>>>(
                d_trace, height, width, d_records, d_counts, num_records
            );
        break;
    default:
        return cudaErrorInvalidConfiguration;
    }

    return CHECK_KERNEL();
}

// Prepares d_num_records for use with sort reduce and stores the temporary
// buffer size necessary for both cub functions (i.e. sort and reduce).
extern "C" int _deferral_poseidon2_deduplicate_records_get_temp_bytes(
    Fp *d_records,
    DeferralPoseidon2Count *d_counts,
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
        DeferralPoseidon2CountCompose{},
        num_records,
        cudaStreamPerThread
    );

    *h_temp_bytes_out = std::max(sort_storage_bytes, reduce_storage_bytes);
    return CHECK_KERNEL();
}

// Reduces the records, removing duplicates and summing multiplicities into
// d_counts. The number of records after reduction is stored into d_num_records.
// The value of temp_storage_bytes should be computed using the _get_temp_bytes
// function above.
extern "C" int _deferral_poseidon2_deduplicate_records(
    Fp *d_records,
    DeferralPoseidon2Count *d_counts,
    size_t num_records,
    size_t *d_num_records,
    void *d_temp_storage,
    size_t temp_storage_bytes
) {
    FpArray<16> *d_records_fp16 = reinterpret_cast<FpArray<16> *>(d_records);

    cub::DeviceMergeSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_records_fp16,
        d_counts,
        num_records,
        Fp16CompareOp(),
        cudaStreamPerThread
    );

    cub::DeviceReduce::ReduceByKey(
        d_temp_storage,
        temp_storage_bytes,
        d_records_fp16,
        d_records_fp16,
        d_counts,
        d_counts,
        d_num_records,
        DeferralPoseidon2CountCompose{},
        num_records,
        cudaStreamPerThread
    );

    return CHECK_KERNEL();
}
