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

#include <cassert>
#include <cstddef>
#include <cstdint>

template <typename T> struct StackingClaimsCols {
    T proof_idx;
    T is_valid;
    T is_padding;
    T is_first;
    T is_last;

    T commit_idx;
    T stacked_col_idx;

    T tidx;
    T mu[D_EF];
    T mu_pow[D_EF];

    T mu_pow_witness;
    T mu_pow_sample;
    T global_col_idx;

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
    uint32_t num_valid;
    FpExt mu;
    Fp mu_pow_witness;
    Fp mu_pow_sample;
};

template <size_t NUM_PROOFS>
__global__ void stacking_claims_tracegen(
    Fp *trace,
    size_t height,
    const Array<uint32_t, NUM_PROOFS> row_bounds,
    const PtrArray<StackingClaim, NUM_PROOFS> claims, // [records[i].num_valid]
    const PtrArray<FpExt, NUM_PROOFS> coeffs,         // [records[i].num_valid]
    const PtrArray<FpExt, NUM_PROOFS> mu_pows,        // [records[i].num_valid]
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
        row.fill_zero(0, sizeof(StackingClaimsCols<uint8_t>));
        COL_WRITE_VALUE(row, StackingClaimsCols, proof_idx, NUM_PROOFS);
        COL_WRITE_VALUE(row, StackingClaimsCols, is_last, global_row_idx + 1 == height);
        return;
    }

    uint32_t num_valid = records[proof_idx].num_valid;
    bool is_valid = row_idx < num_valid;
    bool is_padding = !is_valid;
    bool is_last = is_padding ? (row_idx + 1 == num_rows)
                              : (num_valid == num_rows && row_idx + 1 == num_valid);

    row.fill_zero(0, sizeof(StackingClaimsCols<uint8_t>));
    COL_WRITE_VALUE(row, StackingClaimsCols, proof_idx, proof_idx);
    COL_WRITE_VALUE(row, StackingClaimsCols, is_valid, is_valid);
    COL_WRITE_VALUE(row, StackingClaimsCols, is_padding, is_padding);
    COL_WRITE_VALUE(row, StackingClaimsCols, is_first, is_valid && row_idx == 0);
    COL_WRITE_VALUE(row, StackingClaimsCols, is_last, is_last);
    COL_WRITE_VALUE(row, StackingClaimsCols, global_col_idx, row_idx);

    if (is_padding) {
        // Padding rows leave scan inputs as zero. A post-scan kernel resets
        // scanned accumulators for padding rows to zero.
        return;
    }

    StackingClaim claim = claims[proof_idx][row_idx];
    FpExt coeff = coeffs[proof_idx][row_idx];
    FpExt mu_pow = mu_pows[proof_idx][row_idx];
    ClaimsRecordsPerProof record = records[proof_idx];

    COL_WRITE_VALUE(row, StackingClaimsCols, commit_idx, claim.commit_idx);
    COL_WRITE_VALUE(row, StackingClaimsCols, stacked_col_idx, claim.stacked_col_idx);

    COL_WRITE_VALUE(row, StackingClaimsCols, tidx, record.initial_tidx + (row_idx * D_EF));
    COL_WRITE_ARRAY(row, StackingClaimsCols, mu, record.mu.elems);
    COL_WRITE_ARRAY(row, StackingClaimsCols, mu_pow, mu_pow.elems);

    COL_WRITE_VALUE(row, StackingClaimsCols, mu_pow_witness, record.mu_pow_witness);
    COL_WRITE_VALUE(row, StackingClaimsCols, mu_pow_sample, record.mu_pow_sample);

    COL_WRITE_ARRAY(row, StackingClaimsCols, stacking_claim, claim.claim.elems);
    COL_WRITE_ARRAY(row, StackingClaimsCols, claim_coefficient, coeff.elems);

    // Needs to be accumulated via prefix scan
    FpExt final_s_eval = coeff * claim.claim;
    COL_WRITE_ARRAY(row, StackingClaimsCols, final_s_eval, final_s_eval.elems);

    // Needs to be accumulated via prefix scan
    FpExt whir_claim = mu_pow * claim.claim;
    COL_WRITE_ARRAY(row, StackingClaimsCols, whir_claim, whir_claim.elems);
}

template <size_t NUM_PROOFS>
__global__ void stacking_claims_zero_padding_accums(
    Fp *trace,
    size_t height,
    const Array<uint32_t, NUM_PROOFS> row_bounds,
    const ClaimsRecordsPerProof *__restrict__ records // [NUM_PROOFS]
) {
    uint32_t global_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_row_idx >= height) {
        return;
    }

    uint32_t row_idx = 0, num_rows = 0, last_row_bound = 0;
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
        return;
    }

    uint32_t num_valid = records[proof_idx].num_valid;
    bool is_padding = row_idx >= num_valid;
    if (!is_padding || num_valid == 0 || num_valid == num_rows) {
        return;
    }

    uint32_t proof_row_start = proof_idx == 0 ? 0 : row_bounds[proof_idx - 1];
    uint32_t last_valid_global_row_idx = proof_row_start + num_valid - 1;

    RowSlice row(trace + global_row_idx, height);
    RowSlice last_valid_row(trace + last_valid_global_row_idx, height);

    constexpr uint32_t final_s_eval_col = COL_INDEX(StackingClaimsCols, final_s_eval);
    constexpr uint32_t whir_claim_col = COL_INDEX(StackingClaimsCols, whir_claim);

#pragma unroll
    for (uint32_t i = 0; i < D_EF; i++) {
        row[final_s_eval_col + i] = row[final_s_eval_col + i] - last_valid_row[final_s_eval_col + i];
        row[whir_claim_col + i] = row[whir_claim_col + i] - last_valid_row[whir_claim_col + i];
    }
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
    Fp *d_claim_accums = d_trace + COL_INDEX(StackingClaimsCols, final_s_eval) * height;
    return prefix_scan_by_key_n_arrays_temp_bytes<2 * D_EF>(
        d_proof_idx, d_claim_accums, height, *h_temp_bytes_out, FpEqual{}
    );
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

    // Single SWITCH_BLOCK for both kernel dispatches to work around an NVCC
    // 12.9 bug.
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
        );
        {
            int ret = CHECK_KERNEL();
            if (ret) return ret;
            Fp *d_proof_idx = d_trace + COL_INDEX(StackingClaimsCols, proof_idx) * height;
            Fp *d_claim_accums = d_trace + COL_INDEX(StackingClaimsCols, final_s_eval) * height;
            ret = prefix_scan_by_key_n_arrays<2 * D_EF>(
                d_proof_idx, d_claim_accums, height, d_temp_buffer, temp_bytes, FpEqual{}
            );
            if (ret) return ret;
        }
        stacking_claims_zero_padding_accums<NUM_PROOFS><<<grid, block>>>(
             d_trace, height, Array<uint32_t, NUM_PROOFS>(h_row_bounds), d_records
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
