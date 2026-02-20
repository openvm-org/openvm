#include "affine_scan.cuh"
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "nested_for_loop.h"
#include "primitives/trace_access.h"
#include "ptr_array.h"
#include "switch_macro.h"
#include "types.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector_types.h>

template <typename T> struct ConstraintsFoldingCols {
    T is_valid;
    T is_first;
    T proof_idx;

    T air_idx;
    T sort_idx;
    T constraint_idx;
    T n_lift;

    T lambda_tidx;
    T lambda[D_EF];

    T value[D_EF];
    T cur_sum[D_EF];
    T eq_n[D_EF];

    T is_first_in_air;
    NestedForLoopAuxCols<T, 1> loop_aux;
};

template <size_t NUM_PROOFS>
__global__ void constraints_folding_tracegen(
    Fp *trace,
    size_t height,
    const uint2 *__restrict__ proof_and_sort_idxs,              // [num_valid_rows]
    const AffineFpExt *__restrict__ cur_sum_evals,              // [num_valid_rows]
    const FpExt *__restrict__ values,                           // [num_valid_rows]
    const Array<uint32_t, NUM_PROOFS> row_bounds,               // [NUM_PROOFS]
    const PtrArray<uint32_t, NUM_PROOFS> constraint_bounds,     // [NUM_PROOFS][num_airs]
    const PtrArray<TraceHeight, NUM_PROOFS> sorted_trace_vdata, // [NUM_PROOFS][num_airs]
    const PtrArray<FpExt, NUM_PROOFS> eq_ns,                    // [NUM_PROOFS][n_stack]
    const FpExtWithTidx *per_proof,                             // [NUM_PROOFS]
    uint32_t num_airs,
    uint32_t num_valid_rows,
    uint32_t l_skip
) {
    uint32_t global_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + global_row_idx, height);

    if (global_row_idx >= num_valid_rows) {
        row.fill_zero(0, sizeof(ConstraintsFoldingCols<uint8_t>));
        COL_WRITE_VALUE(row, ConstraintsFoldingCols, is_first, Fp::one());
        COL_WRITE_VALUE(
            row, ConstraintsFoldingCols, proof_idx, NUM_PROOFS + global_row_idx - num_valid_rows
        );
        COL_WRITE_VALUE(row, ConstraintsFoldingCols, is_first_in_air, Fp::one());
        return;
    }

    auto [proof_idx, sort_idx] = proof_and_sort_idxs[global_row_idx];
    bool is_last = global_row_idx + 1 == row_bounds[proof_idx];
    uint32_t proof_start_idx = proof_idx == 0 ? 0 : row_bounds[proof_idx - 1];
    uint32_t row_idx = global_row_idx - proof_start_idx;

    COL_WRITE_VALUE(row, ConstraintsFoldingCols, is_valid, Fp::one());
    COL_WRITE_VALUE(row, ConstraintsFoldingCols, is_first, row_idx == 0);
    COL_WRITE_VALUE(row, ConstraintsFoldingCols, proof_idx, proof_idx);

    uint32_t start_constraint_idx =
        (sort_idx == 0) ? 0 : constraint_bounds[proof_idx][sort_idx - 1];
    uint32_t constraint_idx = row_idx - start_constraint_idx;

    COL_WRITE_VALUE(row, ConstraintsFoldingCols, is_first_in_air, constraint_idx == 0);
    COL_WRITE_VALUE(
        row, ConstraintsFoldingCols, loop_aux.is_transition[0], is_last ? Fp::zero() : Fp::one()
    );

    auto [air_idx, log_height] = sorted_trace_vdata[proof_idx][sort_idx];
    uint32_t n_lift = log_height > l_skip ? log_height - l_skip : 0;

    COL_WRITE_VALUE(row, ConstraintsFoldingCols, air_idx, air_idx);
    COL_WRITE_VALUE(row, ConstraintsFoldingCols, sort_idx, sort_idx);
    COL_WRITE_VALUE(row, ConstraintsFoldingCols, constraint_idx, constraint_idx);
    COL_WRITE_VALUE(row, ConstraintsFoldingCols, n_lift, n_lift);

    auto [lambda, lambda_tidx] = per_proof[proof_idx];
    COL_WRITE_VALUE(row, ConstraintsFoldingCols, lambda_tidx, lambda_tidx);
    COL_WRITE_ARRAY(row, ConstraintsFoldingCols, lambda, lambda.elems);

    FpExt value = values[global_row_idx];
    COL_WRITE_ARRAY(row, ConstraintsFoldingCols, value, value.elems);

    // The cur_sum values in cur_sum_evals are reversed within each AIR's segment,
    // so we need to get the reverse idx
    uint32_t end_constraint_idx = constraint_bounds[proof_idx][sort_idx] - 1;
    uint32_t rev_row_idx = end_constraint_idx - constraint_idx;
    uint32_t rev_global_row_idx = proof_start_idx + rev_row_idx;
    auto [_, cur_sum] = cur_sum_evals[rev_global_row_idx];
    COL_WRITE_ARRAY(row, ConstraintsFoldingCols, cur_sum, cur_sum.elems);

    FpExt eq_n = eq_ns[proof_idx][n_lift];
    COL_WRITE_ARRAY(row, ConstraintsFoldingCols, eq_n, eq_n.elems);
}

// ============================================================================
// LAUNCHERS
// ============================================================================

extern "C" int _constraints_folding_tracegen_temp_bytes(
    const uint2 *d_proof_and_sort_idxs,
    AffineFpExt *d_cur_sum_evals,
    uint32_t num_valid_rows,
    size_t *h_temp_bytes_out
) {
    size_t temp_bytes;
    int ret = get_affine_scan_by_key_temp_bytes(
        d_proof_and_sort_idxs, d_cur_sum_evals, num_valid_rows, temp_bytes
    );
    if (ret) {
        return ret;
    }
    *h_temp_bytes_out = temp_bytes;
    return ret;
}

extern "C" int _constraints_folding_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    const uint2 *d_proof_and_sort_idxs,
    AffineFpExt *d_cur_sum_evals,
    const FpExt *d_values,
    uint32_t *h_row_bounds,
    uint32_t **d_constraint_bounds,
    TraceHeight **d_sorted_trace_vdata,
    FpExt **eq_ns,
    FpExtWithTidx *d_per_proof,
    uint32_t num_proofs,
    uint32_t num_airs,
    uint32_t num_valid_rows,
    uint32_t l_skip,
    void *d_temp_buffer,
    size_t temp_bytes
) {
    assert(width == sizeof(ConstraintsFoldingCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 256);

    // We use a prefix scan to compute each row's cur_sum. Within each key tuple
    // (proof_idx, sort_idx) with n constraint values, each cur_sum[i] = value[i]
    // + lambda * value[i + 1] + ... + lambda^{n - i} * value[n]. To compute this,
    // we store affine[m - 1 - i] = (lambda, value[i]) and do an affine prefix
    // scan, which results in each cur_sum[i] being stored in affine[m - 1 - i].b.
    SWITCH_BLOCK(
        num_proofs,
        NUM_PROOFS,
        (reverse_affines_setup<NUM_PROOFS><<<grid, block>>>(
             d_proof_and_sort_idxs,
             d_cur_sum_evals,
             d_per_proof,
             d_values,
             Array<uint32_t, NUM_PROOFS>(h_row_bounds),
             PtrArray<uint32_t, NUM_PROOFS>(d_constraint_bounds),
             num_valid_rows
        );
         int ret = CHECK_KERNEL();
         if (ret) return ret;
         ret = affine_scan_by_key(
             d_proof_and_sort_idxs, d_cur_sum_evals, num_valid_rows, d_temp_buffer, temp_bytes
         );
         if (ret) return ret;
         constraints_folding_tracegen<NUM_PROOFS><<<grid, block>>>(
             d_trace,
             height,
             d_proof_and_sort_idxs,
             d_cur_sum_evals,
             d_values,
             Array<uint32_t, NUM_PROOFS>(h_row_bounds),
             PtrArray<uint32_t, NUM_PROOFS>(d_constraint_bounds),
             PtrArray<TraceHeight, NUM_PROOFS>(d_sorted_trace_vdata),
             PtrArray<FpExt, NUM_PROOFS>(eq_ns),
             d_per_proof,
             num_airs,
             num_valid_rows,
             l_skip
         );),
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8
    )

    return CHECK_KERNEL();
}
