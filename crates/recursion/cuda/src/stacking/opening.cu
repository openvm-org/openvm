#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "poly_common.cuh"
#include "primitives/trace_access.h"
#include "ptr_array.h"
#include "scan.cuh"
#include "stacking_blob.cuh"
#include "switch_macro.h"
#include "types.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cub/device/device_scan.cuh>
#include <driver_types.h>

template <typename T> struct OpeningClaimsCols {
    T proof_idx;
    T is_valid;
    T is_first;
    T is_last;

    T sort_idx;
    T part_idx;
    T col_idx;
    T col_claim[D_EF];
    T rot_claim[D_EF];

    T is_main;
    T is_transition_main;

    T hypercube_dim;
    T log_lifted_height;
    T lifted_height;
    T lifted_height_inv;

    T tidx;
    T lambda[D_EF];
    T lambda_pow[D_EF];

    T commit_idx;
    T stacked_col_idx;
    T row_idx;
    T is_last_for_claim;

    T eq_in[D_EF];
    T k_rot_in[D_EF];
    T eq_bits[D_EF];

    T lambda_pow_eq_bits[D_EF];

    T stacking_claim_coefficient[D_EF];

    T s_0[D_EF];
};

struct ColumnOpeningClaims {
    uint32_t sort_idx;
    uint32_t part_idx;
    uint32_t col_idx;
    FpExt col_claim;
    FpExt rot_claim;
};

struct OpeningRecordsPerProof {
    uint32_t tidx_before_column_openings;
    uint32_t last_main_idx;
    FpExt lambda;
};

template <size_t NUM_PROOFS>
__global__ void opening_claims_tracegen(
    Fp *trace,
    size_t height,
    const Array<uint32_t, NUM_PROOFS> row_bounds,
    const PtrArray<ColumnOpeningClaims, NUM_PROOFS> claims,  // [row_bounds[i] - row_bounds[i - 1]]
    const PtrArray<StackedSliceData, NUM_PROOFS> slice_data, // [row_bounds[i] - row_bounds[i - 1]]
    const PtrArray<PolyPrecomputation, NUM_PROOFS> precomps, // [row_bounds[i] - row_bounds[i - 1]]
    const PtrArray<FpExt, NUM_PROOFS> lambda_pows,           // [row_bounds[i] - row_bounds[i - 1]]
    const OpeningRecordsPerProof *__restrict__ records,      // [NUM_PROOFS]
    uint32_t l_skip
) {
    uint32_t global_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + global_row_idx, height);

    uint32_t row_idx, num_rows, last_row_bound = 0;
    uint32_t proof_idx = NUM_PROOFS;
#pragma unroll
    for (uint32_t i = 0; i < NUM_PROOFS; i++) {
        uint32_t row_bound_i = row_bounds[i];
        num_rows = row_bound_i - last_row_bound;
        if (row_bound_i > global_row_idx) {
            proof_idx = i;
            row_idx = global_row_idx - last_row_bound;
            break;
        }
        last_row_bound = row_bound_i;
    }

    if (proof_idx == NUM_PROOFS) {
        row.fill_zero(0, sizeof(OpeningClaimsCols<uint8_t>));
        COL_WRITE_VALUE(row, OpeningClaimsCols, proof_idx, NUM_PROOFS);
        COL_WRITE_VALUE(row, OpeningClaimsCols, is_last, global_row_idx + 1 == height);
        return;
    }

    COL_WRITE_VALUE(row, OpeningClaimsCols, proof_idx, proof_idx);
    COL_WRITE_VALUE(row, OpeningClaimsCols, is_valid, Fp::one());
    COL_WRITE_VALUE(row, OpeningClaimsCols, is_first, row_idx == 0);
    COL_WRITE_VALUE(row, OpeningClaimsCols, is_last, row_idx + 1 == num_rows);

    ColumnOpeningClaims claim = claims[proof_idx][row_idx];
    StackedSliceData slice = slice_data[proof_idx][row_idx];
    FpExt lambda_pow = lambda_pows[proof_idx][row_idx];
    auto [eq_in, k_rot_in, eq_bits] = precomps[proof_idx][row_idx];
    auto [tidx_before_column_openings, last_main_idx, lambda] = records[proof_idx];

    COL_WRITE_VALUE(row, OpeningClaimsCols, sort_idx, claim.sort_idx);
    COL_WRITE_VALUE(row, OpeningClaimsCols, part_idx, claim.part_idx);
    COL_WRITE_VALUE(row, OpeningClaimsCols, col_idx, claim.col_idx);
    COL_WRITE_ARRAY(row, OpeningClaimsCols, col_claim, claim.col_claim.elems);
    COL_WRITE_ARRAY(row, OpeningClaimsCols, rot_claim, claim.rot_claim.elems);

    COL_WRITE_VALUE(row, OpeningClaimsCols, is_main, claim.part_idx == 0);
    COL_WRITE_VALUE(
        row,
        OpeningClaimsCols,
        is_transition_main,
        row_idx + 1 != num_rows && row_idx != last_main_idx
    );

    uint32_t n_lift = (uint32_t)max(slice.n, 0);
    uint32_t log_lifted_height = n_lift + l_skip;
    uint32_t n_abs = abs(slice.n);
    Fp lifted_height = Fp(1 << log_lifted_height);
    COL_WRITE_VALUE(row, OpeningClaimsCols, hypercube_dim, slice.n >= 0 ? Fp(n_lift) : -Fp(n_abs));
    COL_WRITE_VALUE(row, OpeningClaimsCols, log_lifted_height, log_lifted_height);
    COL_WRITE_VALUE(row, OpeningClaimsCols, lifted_height, lifted_height);
    COL_WRITE_VALUE(row, OpeningClaimsCols, lifted_height_inv, inv(lifted_height));

    COL_WRITE_VALUE(
        row, OpeningClaimsCols, tidx, tidx_before_column_openings + (row_idx << 1) * D_EF
    );
    COL_WRITE_ARRAY(row, OpeningClaimsCols, lambda, lambda.elems);
    COL_WRITE_ARRAY(row, OpeningClaimsCols, lambda_pow, lambda_pow.elems);

    COL_WRITE_VALUE(row, OpeningClaimsCols, commit_idx, slice.commit_idx);
    COL_WRITE_VALUE(row, OpeningClaimsCols, stacked_col_idx, slice.col_idx);
    COL_WRITE_VALUE(row, OpeningClaimsCols, row_idx, slice.row_idx);
    COL_WRITE_VALUE(row, OpeningClaimsCols, is_last_for_claim, slice.is_last_for_claim);

    COL_WRITE_ARRAY(row, OpeningClaimsCols, eq_in, eq_in.elems);
    COL_WRITE_ARRAY(row, OpeningClaimsCols, k_rot_in, k_rot_in.elems);
    COL_WRITE_ARRAY(row, OpeningClaimsCols, eq_bits, eq_bits.elems);

    FpExt lambda_pow_eq_bits = lambda_pow * eq_bits;
    COL_WRITE_ARRAY(row, OpeningClaimsCols, lambda_pow_eq_bits, lambda_pow_eq_bits.elems);

    // Needs to be accumulated via prefix scan by key (commit_idx)
    FpExt stacking_claim_coefficient = lambda_pow_eq_bits * (eq_in + lambda * k_rot_in);
    COL_WRITE_ARRAY(
        row, OpeningClaimsCols, stacking_claim_coefficient, stacking_claim_coefficient.elems
    );

    // Needs to be accumulated via prefix scan
    FpExt s_0 = lambda_pow * (claim.col_claim + lambda * claim.rot_claim);
    COL_WRITE_ARRAY(row, OpeningClaimsCols, s_0, s_0.elems);
}

// ============================================================================
// LAUNCHERS
// ============================================================================

extern "C" int _opening_claims_tracegen_temp_bytes(
    Fp *d_trace,
    size_t height,
    Fp *d_keys_buffer,
    size_t *h_temp_bytes_out
) {
    Fp *d_last_for_claim = d_trace + COL_INDEX(OpeningClaimsCols, is_last_for_claim) * height;
    size_t exclusive_scan_temp_bytes;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        exclusive_scan_temp_bytes,
        d_last_for_claim,
        d_keys_buffer,
        height,
        cudaStreamPerThread
    );
    int ret = CHECK_KERNEL();
    if (ret) {
        return ret;
    }

    Fp *d_coeffs = d_trace + COL_INDEX(OpeningClaimsCols, stacking_claim_coefficient) * height;
    size_t coeffs_temp_bytes;
    ret = get_fp_prefix_scan_by_key_n_arrays_temp_bytes<D_EF>(
        d_keys_buffer, d_coeffs, height, coeffs_temp_bytes
    );
    if (ret) {
        return ret;
    }

    Fp *d_proof_idx = d_trace + COL_INDEX(OpeningClaimsCols, proof_idx) * height;
    Fp *d_s_0 = d_trace + COL_INDEX(OpeningClaimsCols, s_0) * height;
    size_t s_0_temp_bytes;
    ret = get_fp_prefix_scan_by_key_n_arrays_temp_bytes<D_EF>(
        d_proof_idx, d_s_0, height, s_0_temp_bytes
    );
    if (ret) {
        return ret;
    }

    *h_temp_bytes_out =
        std::max(exclusive_scan_temp_bytes, std::max(coeffs_temp_bytes, s_0_temp_bytes));
    return ret;
}

extern "C" int _opening_claims_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint32_t *h_row_bounds,
    ColumnOpeningClaims **d_claims,
    StackedSliceData **d_slice_data,
    PolyPrecomputation **d_precomps,
    FpExt **d_lambda_pows,
    OpeningRecordsPerProof *d_records,
    uint32_t num_proofs,
    uint32_t l_skip,
    Fp *d_keys_buffer,
    void *d_temp_buffer,
    size_t temp_bytes
) {
    assert(width == sizeof(OpeningClaimsCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 256);

    SWITCH_BLOCK(
        num_proofs,
        NUM_PROOFS,
        (opening_claims_tracegen<NUM_PROOFS><<<grid, block>>>(
             d_trace,
             height,
             Array<uint32_t, NUM_PROOFS>(h_row_bounds),
             PtrArray<ColumnOpeningClaims, NUM_PROOFS>(d_claims),
             PtrArray<StackedSliceData, NUM_PROOFS>(d_slice_data),
             PtrArray<PolyPrecomputation, NUM_PROOFS>(d_precomps),
             PtrArray<FpExt, NUM_PROOFS>(d_lambda_pows),
             d_records,
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

    int ret = CHECK_KERNEL();
    if (ret) {
        return ret;
    }

    Fp *d_last_for_claim = d_trace + COL_INDEX(OpeningClaimsCols, is_last_for_claim) * height;
    cub::DeviceScan::ExclusiveSum(
        d_temp_buffer, temp_bytes, d_last_for_claim, d_keys_buffer, height, cudaStreamPerThread
    );
    ret = CHECK_KERNEL();
    if (ret) {
        return ret;
    }

    Fp *d_coeffs = d_trace + COL_INDEX(OpeningClaimsCols, stacking_claim_coefficient) * height;
    ret = prefix_scan_by_key_n_arrays<D_EF>(
        d_keys_buffer, d_coeffs, height, d_temp_buffer, temp_bytes
    );
    if (ret) {
        return ret;
    }

    Fp *d_proof_idx = d_trace + COL_INDEX(OpeningClaimsCols, proof_idx) * height;
    Fp *d_s_0 = d_trace + COL_INDEX(OpeningClaimsCols, s_0) * height;
    ret = prefix_scan_by_key_n_arrays<D_EF>(d_proof_idx, d_s_0, height, d_temp_buffer, temp_bytes);
    return ret;
}
