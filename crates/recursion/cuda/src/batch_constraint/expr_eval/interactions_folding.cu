#include "affine_scan.cuh"
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "nested_for_loop.h"
#include "primitives/trace_access.h"
#include "ptr_array.h"
#include "scan.cuh"
#include "switch_macro.h"
#include "types.h"
#include "util.cuh"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector_types.h>

template <typename T> struct InteractionsFoldingCols {
    T is_valid;
    T is_first;
    T proof_idx;

    T beta_tidx;

    T air_idx;
    T sort_idx;
    T interaction_idx;
    T node_idx;

    T has_interactions;

    T is_first_in_air;
    T is_first_in_message;
    T is_second_in_message;
    T is_bus_index;

    NestedForLoopAuxCols<T, 2> loop_aux;

    T idx_in_message;
    T value[D_EF];
    T cur_sum[D_EF];
    T beta[D_EF];
    T eq_3b[D_EF];

    T final_acc_num[D_EF];
    T final_acc_denom[D_EF];
};

struct InteractionRecord {
    uint32_t interaction_num_rows;
    uint32_t global_start_row;
    uint32_t stacked_idx;
};

template <size_t NUM_PROOFS>
__global__ void interactions_folding_tracegen(
    Fp *trace,
    size_t height,
    uint2 *__restrict__ idx_keys,                                // [num_valid_rows]
    const AffineFpExt *__restrict__ cur_sum_evals,               // [num_valid_rows]
    const FpExt *__restrict__ values,                            // [num_valid_rows]
    const uint32_t *node_idxs,                                   // [num_valid_rows]
    const PtrArray<uint32_t, NUM_PROOFS> air_interaction_bounds, // [NUM_PROOFS][num_airs]
    const PtrArray<TraceHeight, NUM_PROOFS> sorted_trace_vdata,  // [NUM_PROOFS][num_airs]
    const PtrArray<InteractionRecord, NUM_PROOFS> records,       // [NUM_PROOFS}[num_interactions]
    const PtrArray<FpExt, NUM_PROOFS> xis,                       // [NUM_PROOFS}[l_skip + n_global]
    const FpExtWithTidx *per_proof,                              // [NUM_PROOFS]
    const Array<uint32_t, NUM_PROOFS> num_airs,                  // [NUM_PROOFS]
    const Array<uint32_t, NUM_PROOFS> n_logups,                  // [NUM_PROOFS]
    uint32_t num_valid_rows,
    uint32_t l_skip
) {
    uint32_t global_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + global_row_idx, height);

    if (global_row_idx >= num_valid_rows) {
        row.fill_zero(0, sizeof(InteractionsFoldingCols<uint8_t>));
        COL_WRITE_VALUE(row, InteractionsFoldingCols, is_first, Fp::one());
        COL_WRITE_VALUE(
            row, InteractionsFoldingCols, proof_idx, NUM_PROOFS + global_row_idx - num_valid_rows
        );
        COL_WRITE_VALUE(row, InteractionsFoldingCols, is_first_in_air, Fp::one());
        COL_WRITE_VALUE(row, InteractionsFoldingCols, is_first_in_message, Fp::one());
        return;
    }

    // The initial affine scan used to generate cur_sum_evals will segment by proof_idx
    // and global_interaction_idx (i.e. interaction_idx within proof).
    auto [proof_idx, global_interaction_idx] = idx_keys[global_row_idx];

    // Two suffix scans will be required to compute final_acc_num and final_acc_denom,
    // which requires segmentation by proof_idx and sort_idx instead.
    const uint32_t *interaction_bounds = air_interaction_bounds[proof_idx];
    uint32_t sort_idx =
        partition_point_leq(interaction_bounds, num_airs[proof_idx], global_interaction_idx);
    idx_keys[global_row_idx].y = sort_idx;

    uint32_t air_start_interaction_idx = sort_idx == 0 ? 0 : interaction_bounds[sort_idx - 1];
    uint32_t air_end_interaction_idx = interaction_bounds[sort_idx] - 1;

    auto [interaction_num_rows, global_start_row, stacked_idx] =
        records[proof_idx][global_interaction_idx];
    uint32_t interaction_idx = global_interaction_idx - air_start_interaction_idx;
    uint32_t interaction_row_idx = global_row_idx - global_start_row;

    // AIRs without interactions get a single row that "counts" as a global interaction,
    // the per_interaction record for which has interaction_num_rows = 1. Otherwise, each
    // interaction will take at least 3 rows (i.e. for mult, message, and then bus index).
    bool has_interactions = interaction_num_rows > 1;
    bool is_first_in_message = interaction_row_idx == 0;
    bool is_last_in_message = interaction_row_idx + 1 == interaction_num_rows;
    bool is_first_in_air = is_first_in_message && interaction_idx == 0;
    bool is_last_in_air = global_interaction_idx == air_end_interaction_idx && is_last_in_message;
    bool is_last_in_proof = is_last_in_air && sort_idx + 1 == num_airs[proof_idx];

    COL_WRITE_VALUE(row, InteractionsFoldingCols, is_valid, Fp::one());
    COL_WRITE_VALUE(row, InteractionsFoldingCols, is_first, is_first_in_air && sort_idx == 0);
    COL_WRITE_VALUE(row, InteractionsFoldingCols, proof_idx, proof_idx);

    auto [air_idx, log_height] = sorted_trace_vdata[proof_idx][sort_idx];
    uint32_t node_idx = node_idxs[global_row_idx];
    COL_WRITE_VALUE(row, InteractionsFoldingCols, air_idx, air_idx);
    COL_WRITE_VALUE(row, InteractionsFoldingCols, sort_idx, sort_idx);
    COL_WRITE_VALUE(row, InteractionsFoldingCols, interaction_idx, interaction_idx);
    COL_WRITE_VALUE(row, InteractionsFoldingCols, node_idx, node_idx);

    COL_WRITE_VALUE(row, InteractionsFoldingCols, has_interactions, has_interactions);

    bool is_second_in_message = interaction_row_idx == 1;
    bool is_bus_index = is_last_in_message && has_interactions;
    COL_WRITE_VALUE(row, InteractionsFoldingCols, is_first_in_air, is_first_in_air);
    COL_WRITE_VALUE(row, InteractionsFoldingCols, is_first_in_message, is_first_in_message);
    COL_WRITE_VALUE(row, InteractionsFoldingCols, is_second_in_message, is_second_in_message);
    COL_WRITE_VALUE(row, InteractionsFoldingCols, is_bus_index, is_bus_index);

    COL_WRITE_VALUE(row, InteractionsFoldingCols, loop_aux.is_transition[0], !is_last_in_proof);
    COL_WRITE_VALUE(row, InteractionsFoldingCols, loop_aux.is_transition[1], !is_last_in_air);

    uint32_t idx_in_message = is_first_in_message ? 0
                                                  : (is_bus_index ? (interaction_num_rows - 1)
                                                                  : (interaction_row_idx - 1));
    COL_WRITE_VALUE(row, InteractionsFoldingCols, idx_in_message, idx_in_message);

    auto [beta, beta_tidx] = per_proof[proof_idx];
    COL_WRITE_VALUE(row, InteractionsFoldingCols, beta_tidx, beta_tidx);
    COL_WRITE_ARRAY(row, InteractionsFoldingCols, beta, beta.elems);

    FpExt value = values[global_row_idx];
    COL_WRITE_ARRAY(row, InteractionsFoldingCols, value, value.elems);

    // On the first interaction row (either mult or dummy), cur_sum is just value (i.e. no
    // beta folding). On subsequent rows, cur_sum is the beta-folded suffix.
    FpExt cur_sum;
    if (is_first_in_message) {
        cur_sum = value;
    } else {
        // The cur_sum values in cur_sum_evals are reversed within each global interaction
        // segment, so we need to get the reverse idx.
        uint32_t rev_global_row_idx =
            global_start_row + interaction_num_rows - 1 - interaction_row_idx;
        auto [_, folded] = cur_sum_evals[rev_global_row_idx];
        cur_sum = folded;
    }
    COL_WRITE_ARRAY(row, InteractionsFoldingCols, cur_sum, cur_sum.elems);

    uint32_t n_lift = log_height > l_skip ? log_height - l_skip : 0;
    FpExt eq_3b;
    if (has_interactions) {
        eq_3b = FpExt(Fp::one());
        for (uint32_t i = l_skip + n_lift; i < l_skip + n_logups[proof_idx]; i++) {
            FpExt xi = xis[proof_idx][i];
            if ((stacked_idx & (1 << i)) == 0) {
                xi = FpExt(Fp::one()) - xi;
            }
            eq_3b *= xi;
        }
    } else {
        eq_3b = FpExt(Fp::zero());
    }
    COL_WRITE_ARRAY(row, InteractionsFoldingCols, eq_3b, eq_3b.elems);

    // We will compute the final values for final_accum_num and final_acc_denom
    // using a suffix sum (not in this kernel).
    FpExt cur_sum_eq_3b = cur_sum * eq_3b;
    if (is_first_in_message) {
        COL_WRITE_ARRAY(row, InteractionsFoldingCols, final_acc_num, cur_sum_eq_3b.elems);
    } else {
        row.fill_zero(COL_INDEX(InteractionsFoldingCols, final_acc_num), D_EF);
    }
    if (is_second_in_message || !has_interactions) {
        COL_WRITE_ARRAY(row, InteractionsFoldingCols, final_acc_denom, cur_sum_eq_3b.elems);
    } else {
        row.fill_zero(COL_INDEX(InteractionsFoldingCols, final_acc_denom), D_EF);
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================

extern "C" int _interaction_folding_tracegen_temp_bytes(
    Fp *d_trace,
    size_t height,
    const uint2 *d_idx_keys,
    AffineFpExt *d_cur_sum_evals,
    uint32_t num_valid_rows,
    size_t *h_temp_bytes_out
) {
    size_t affine_scan_temp_bytes;
    int ret = get_affine_scan_by_key_temp_bytes(
        d_idx_keys, d_cur_sum_evals, num_valid_rows, affine_scan_temp_bytes
    );
    if (ret) {
        return ret;
    }

    Fp *d_final_acc_num = d_trace + COL_INDEX(InteractionsFoldingCols, final_acc_num) * height;
    size_t num_scan_temp_bytes;
    ret = suffix_scan_by_key_temp_bytes(
        d_idx_keys, d_final_acc_num, num_valid_rows, num_scan_temp_bytes, UInt2Equal{}
    );
    if (ret) {
        return ret;
    }

    Fp *d_final_acc_denom = d_trace + COL_INDEX(InteractionsFoldingCols, final_acc_denom) * height;
    size_t denom_scan_temp_bytes;
    ret = suffix_scan_by_key_temp_bytes(
        d_idx_keys, d_final_acc_denom, num_valid_rows, denom_scan_temp_bytes, UInt2Equal{}
    );
    if (ret) {
        return ret;
    }

    *h_temp_bytes_out =
        std::max(affine_scan_temp_bytes, std::max(num_scan_temp_bytes, denom_scan_temp_bytes));
    return ret;
}

extern "C" int _interactions_folding_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint2 *d_idx_keys,
    AffineFpExt *d_cur_sum_evals,
    const FpExt *d_values,
    const uint32_t *d_node_idxs,
    uint32_t *h_row_bounds,
    uint32_t **d_air_interaction_bounds,
    uint32_t **d_interaction_row_bounds,
    TraceHeight **d_sorted_trace_vdata,
    InteractionRecord **d_records,
    FpExt **d_xis,
    FpExtWithTidx *d_per_proof,
    uint32_t *h_num_airs,
    uint32_t *h_n_logups,
    uint32_t num_proofs,
    uint32_t num_valid_rows,
    uint32_t l_skip,
    void *d_temp_buffer,
    size_t temp_bytes
) {
    assert(width == sizeof(InteractionsFoldingCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 256);

    // We use a prefix scan to compute each row's cur_sum. Within each key (proof_idx,
    // global_interaction_idx) with n constraint values, each cur_sum[i] = value[i] +
    // lambda * value[i + 1] + ... + lambda^{n - i} * value[n]. To compute this, we
    // store affine[m - 1 - i] = (lambda, value[i]) and do an affine prefix scan, which
    // results in each cur_sum[i] being stored in affine[m - 1 - i].b.
    SWITCH_BLOCK(
        num_proofs,
        NUM_PROOFS,
        (reverse_affines_setup<NUM_PROOFS><<<grid, block>>>(
             d_idx_keys,
             d_cur_sum_evals,
             d_per_proof,
             d_values,
             Array<uint32_t, NUM_PROOFS>(h_row_bounds),
             PtrArray<uint32_t, NUM_PROOFS>(d_interaction_row_bounds),
             num_valid_rows
        );
         int ret = CHECK_KERNEL();
         if (ret) return ret;
         ret = affine_scan_by_key(
             d_idx_keys, d_cur_sum_evals, num_valid_rows, d_temp_buffer, temp_bytes
         );
         if (ret) return ret;
         interactions_folding_tracegen<NUM_PROOFS><<<grid, block>>>(
             d_trace,
             height,
             d_idx_keys,
             d_cur_sum_evals,
             d_values,
             d_node_idxs,
             PtrArray<uint32_t, NUM_PROOFS>(d_air_interaction_bounds),
             PtrArray<TraceHeight, NUM_PROOFS>(d_sorted_trace_vdata),
             PtrArray<InteractionRecord, NUM_PROOFS>(d_records),
             PtrArray<FpExt, NUM_PROOFS>(d_xis),
             d_per_proof,
             Array<uint32_t, NUM_PROOFS>(h_num_airs),
             Array<uint32_t, NUM_PROOFS>(h_n_logups),
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

    int ret = CHECK_KERNEL();
    if (ret) {
        return ret;
    }

    Fp *d_final_acc_num = d_trace + COL_INDEX(InteractionsFoldingCols, final_acc_num) * height;
    Fp *d_final_acc_denom = d_trace + COL_INDEX(InteractionsFoldingCols, final_acc_denom) * height;
    for (uint32_t i = 0; i < D_EF; i++) {
        ret = suffix_scan_by_key(
            d_idx_keys,
            d_final_acc_num + i * height,
            num_valid_rows,
            d_temp_buffer,
            temp_bytes,
            UInt2Equal{}
        );
        if (ret) {
            return ret;
        }
        ret = suffix_scan_by_key(
            d_idx_keys,
            d_final_acc_denom + i * height,
            num_valid_rows,
            d_temp_buffer,
            temp_bytes,
            UInt2Equal{}
        );
        if (ret) {
            return ret;
        }
    }

    return ret;
}
