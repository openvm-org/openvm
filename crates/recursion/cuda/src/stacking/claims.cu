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

template <typename T> struct StackingClaimsCols {
    T proof_idx;
    T is_valid;
    T is_first;
    T is_last;

    T commit_idx;
    T stacked_col_idx;

    T tidx;
    T mu[D_EF];
    T mu_pow[D_EF];

    T stacking_claim[D_EF];
    T claim_coefficient[D_EF];

    T final_s_eval[D_EF];

    T whir_claim[D_EF];
};

struct StackingClaim {
    uint32_t commit_idx;
    uint32_t stacked_col_idx;
    FpExt claim;
};

struct ClaimsRecordsPerProof {
    uint32_t initial_tidx;
    FpExt mu;
};

template <size_t NUM_PROOFS>
__global__ void stacking_claims_tracegen(
    Fp *trace,
    size_t height,
    const Array<uint32_t, NUM_PROOFS> row_bounds,
    const PtrArray<StackingClaim, NUM_PROOFS> claims, // [row_bounds[i] - row_bounds[i - 1]]
    const PtrArray<FpExt, NUM_PROOFS> coeffs,         // [row_bounds[i] - row_bounds[i - 1]]
    const PtrArray<FpExt, NUM_PROOFS> mu_pows,        // [row_bounds[i] - row_bounds[i - 1]]
    const ClaimsRecordsPerProof *__restrict__ records // [NUM_PROOFS]
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
        row.fill_zero(1, sizeof(StackingClaimsCols<uint8_t>));
        COL_WRITE_VALUE(row, StackingClaimsCols, proof_idx, NUM_PROOFS);
        COL_WRITE_VALUE(row, StackingClaimsCols, is_last, global_row_idx + 1 == height);
        return;
    }

    COL_WRITE_VALUE(row, StackingClaimsCols, proof_idx, proof_idx);
    COL_WRITE_VALUE(row, StackingClaimsCols, is_valid, Fp::one());
    COL_WRITE_VALUE(row, StackingClaimsCols, is_first, row_idx == 0);
    COL_WRITE_VALUE(row, StackingClaimsCols, is_last, row_idx + 1 == num_rows);

    StackingClaim claim = claims[proof_idx][row_idx];
    FpExt coeff = coeffs[proof_idx][row_idx];
    FpExt mu_pow = mu_pows[proof_idx][row_idx];
    auto [initial_tidx, mu] = records[proof_idx];

    COL_WRITE_VALUE(row, StackingClaimsCols, commit_idx, claim.commit_idx);
    COL_WRITE_VALUE(row, StackingClaimsCols, stacked_col_idx, claim.stacked_col_idx);

    COL_WRITE_VALUE(row, StackingClaimsCols, tidx, initial_tidx + (row_idx * D_EF));
    COL_WRITE_ARRAY(row, StackingClaimsCols, mu, mu.elems);
    COL_WRITE_ARRAY(row, StackingClaimsCols, mu_pow, mu_pow.elems);

    COL_WRITE_ARRAY(row, StackingClaimsCols, stacking_claim, claim.claim.elems);
    COL_WRITE_ARRAY(row, StackingClaimsCols, claim_coefficient, coeff.elems);

    // Needs to be accumulated via prefix scan
    FpExt final_s_eval = coeff * claim.claim;
    COL_WRITE_ARRAY(row, StackingClaimsCols, final_s_eval, final_s_eval.elems);

    // Needs to be accumulated via prefix scan
    FpExt whir_claim = mu_pow * claim.claim;
    COL_WRITE_ARRAY(row, StackingClaimsCols, whir_claim, whir_claim.elems);
}

// ============================================================================
// LAUNCHERS
// ============================================================================

extern "C" int _stacking_claims_tracegen_temp_bytes(
    Fp *d_trace,
    size_t height,
    size_t *h_temp_bytes_out
) {
    Fp *d_proof_idx = d_trace + COL_INDEX(StackingClaimsCols, proof_idx) * height;
    Fp *d_final_s_eval = d_trace + COL_INDEX(StackingClaimsCols, final_s_eval) * height;
    size_t final_s_eval_temp_bytes;
    int ret = get_fp_prefix_scan_by_key_n_arrays_temp_bytes<D_EF>(
        d_proof_idx, d_final_s_eval, height, final_s_eval_temp_bytes
    );
    if (ret) {
        return ret;
    }

    Fp *d_whir_claim = d_trace + COL_INDEX(StackingClaimsCols, whir_claim) * height;
    size_t whir_claim_temp_bytes;
    ret = get_fp_prefix_scan_by_key_n_arrays_temp_bytes<D_EF>(
        d_proof_idx, d_whir_claim, height, whir_claim_temp_bytes
    );
    if (ret) {
        return ret;
    }

    *h_temp_bytes_out = std::max(final_s_eval_temp_bytes, whir_claim_temp_bytes);
    return ret;
}

extern "C" int _stacking_claims_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint32_t *h_row_bounds,
    StackingClaim **d_claims,
    FpExt **d_coeffs,
    FpExt **d_mu_pows,
    ClaimsRecordsPerProof *d_records,
    uint32_t num_proofs,
    void *d_temp_buffer,
    size_t temp_bytes
) {
    assert(width == sizeof(StackingClaimsCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 256);

    SWITCH_BLOCK(
        num_proofs,
        NUM_PROOFS,
        (stacking_claims_tracegen<NUM_PROOFS><<<grid, block>>>(
             d_trace,
             height,
             Array<uint32_t, NUM_PROOFS>(h_row_bounds),
             PtrArray<StackingClaim, NUM_PROOFS>(d_claims),
             PtrArray<FpExt, NUM_PROOFS>(d_coeffs),
             PtrArray<FpExt, NUM_PROOFS>(d_mu_pows),
             d_records
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

    Fp *d_proof_idx = d_trace + COL_INDEX(StackingClaimsCols, proof_idx) * height;
    Fp *d_final_s_eval = d_trace + COL_INDEX(StackingClaimsCols, final_s_eval) * height;
    ret = prefix_scan_by_key_n_arrays<D_EF>(
        d_proof_idx, d_final_s_eval, height, d_temp_buffer, temp_bytes
    );
    if (ret) {
        return ret;
    }

    Fp *d_whir_claim = d_trace + COL_INDEX(StackingClaimsCols, whir_claim) * height;
    ret = prefix_scan_by_key_n_arrays<D_EF>(
        d_proof_idx, d_whir_claim, height, d_temp_buffer, temp_bytes
    );
    return ret;
}
