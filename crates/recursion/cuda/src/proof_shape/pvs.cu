#include "fp.h"
#include "launcher.cuh"
#include "primitives/trace_access.h"
#include "ptr_array.h"
#include "switch_macro.h"
#include "types.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

template <typename T> struct PublicValuesCols {
    T proof_idx;
    T is_valid;
    T is_first;
    T air_idx;
    T tidx;
    T num_pvs_left;
    T is_first_for_air;
    T value;
};

template <size_t NUM_PROOFS>
__global__ void public_values_tracegen(
    Fp *trace,
    size_t height,
    PtrArray<PublicValueData, NUM_PROOFS> pvs_data,
    PtrArray<size_t, NUM_PROOFS> pvs_tidx,
    size_t num_pvs
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < NUM_PROOFS * num_pvs) {
        size_t proof_idx = idx / num_pvs;
        size_t record_idx = idx % num_pvs;
        PublicValueData pv_data = pvs_data[proof_idx][record_idx];
        size_t starting_tidx = pvs_tidx[proof_idx][pv_data.air_idx];
        COL_WRITE_VALUE(row, PublicValuesCols, proof_idx, proof_idx);
        COL_WRITE_VALUE(row, PublicValuesCols, is_valid, Fp::one());
        COL_WRITE_VALUE(row, PublicValuesCols, is_first, record_idx == 0);
        COL_WRITE_VALUE(row, PublicValuesCols, air_idx, pv_data.air_idx);
        COL_WRITE_VALUE(row, PublicValuesCols, tidx, starting_tidx + pv_data.pv_idx);
        COL_WRITE_VALUE(row, PublicValuesCols, num_pvs_left, pv_data.air_num_pvs - pv_data.pv_idx);
        COL_WRITE_VALUE(row, PublicValuesCols, is_first_for_air, pv_data.pv_idx == 0);
        COL_WRITE_VALUE(row, PublicValuesCols, value, pv_data.value);
    } else {
        COL_WRITE_VALUE(row, PublicValuesCols, proof_idx, NUM_PROOFS);
        row.fill_zero(1, sizeof(PublicValuesCols<uint8_t>) - 1);
    }
}

extern "C" int _public_values_recursion_tracegen(
    Fp *d_trace,
    size_t height,
    PublicValueData **d_pvs_data,
    size_t **d_pvs_tidx,
    size_t num_proofs,
    size_t num_pvs
) {
    assert((height & (height - 1)) == 0);
    auto [grid, block] = kernel_launch_params(height);
    SWITCH_BLOCK(
        num_proofs,
        NUM_PROOFS,
        (public_values_tracegen<NUM_PROOFS><<<grid, block>>>(
             d_trace,
             height,
             PtrArray<PublicValueData, NUM_PROOFS>(d_pvs_data),
             PtrArray<size_t, NUM_PROOFS>(d_pvs_tidx),
             num_pvs
        );),
        1,
        2,
        3,
        4,
        5
    )
    return CHECK_KERNEL();
}
