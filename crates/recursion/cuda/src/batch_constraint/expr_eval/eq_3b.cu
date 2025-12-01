#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "primitives/trace_access.h"
#include "types.h"
#include "util.cuh"

#include <cassert>
#include <cstddef>
#include <cstdint>

template <typename T> struct Eq3bColumns {
    T is_valid;
    T is_first;
    T proof_idx;

    T sort_idx;
    T interaction_idx;

    T n_lift;
    T two_to_the_n_lift;
    T n;
    T hypercube_volume;
    T n_at_least_n_lift;

    T has_no_interactions;

    T is_first_in_air;
    T is_first_in_interaction;

    T idx;
    T running_idx;
    T nth_bit;

    T loop_transitions[2];

    T xi[D_EF];
    T eq[D_EF];
};

struct Eq3bStackedIdxRecord {
    uint32_t sort_idx;
    uint32_t interaction_idx;
    uint32_t stacked_idx;
    uint32_t n_lift;
    bool is_last_in_air;
    bool no_interactions;
};

struct RecordIdx {
    uint32_t record_idx;
    uint32_t local_n;
};

__global__ void eq_3b_tracegen(
    Fp *trace,
    size_t num_valid_rows,
    size_t height,
    size_t num_proofs,
    size_t l_skip,
    const Eq3bStackedIdxRecord *__restrict__ records,
    const size_t *__restrict__ record_bounds,
    const RecordIdx *__restrict__ record_idxs,
    const size_t *__restrict__ record_idxs_bounds,
    const size_t *__restrict__ rows_per_proof_bounds,
    const size_t *__restrict__ n_logups,
    const FpExt *__restrict__ xis,
    const size_t *__restrict__ xi_bounds
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= height) {
        return;
    }

    RowSlice row(trace + row_idx, height);

    if (row_idx >= num_valid_rows) {
        row.fill_zero(0, sizeof(Eq3bColumns<uint8_t>));
        uint32_t pad_idx = row_idx - num_valid_rows;
        COL_WRITE_VALUE(row, Eq3bColumns, proof_idx, num_proofs + pad_idx);
        COL_WRITE_VALUE(row, Eq3bColumns, is_first, 1);
        COL_WRITE_VALUE(row, Eq3bColumns, is_first_in_air, 1);
        COL_WRITE_VALUE(row, Eq3bColumns, is_first_in_interaction, 1);
        return;
    }

    row.fill_zero(0, sizeof(Eq3bColumns<uint8_t>));

    uint32_t proof_idx =
        partition_point_leq<size_t>(rows_per_proof_bounds + 1, num_proofs, row_idx);
    if (proof_idx >= num_proofs) {
        return;
    }

    uint32_t proof_row_start = rows_per_proof_bounds[proof_idx];
    uint32_t local_row_idx = row_idx - proof_row_start;

    uint32_t n_logup = n_logups[proof_idx];

    if (n_logup + 1 == 0) {
        return;
    }

    uint32_t record_idxs_start = record_idxs_bounds[proof_idx];
    RecordIdx record_idx_info = record_idxs[record_idxs_start + local_row_idx];
    uint32_t record_idx = record_idx_info.record_idx;
    uint32_t local_n = record_idx_info.local_n;

    uint32_t record_start = record_bounds[proof_idx];
    uint32_t record_end = record_bounds[proof_idx + 1];
    uint32_t record_count = record_end - record_start;

    if (record_idx >= record_count) {
        return;
    }

    const Eq3bStackedIdxRecord &record = records[record_start + record_idx];

    uint32_t shifted_idx = record.stacked_idx >> l_skip;

    bool nth_bit = !!(shifted_idx & (1u << local_n));

    uint32_t xi_offset = xi_bounds[proof_idx];
    uint32_t xi_len = xi_bounds[proof_idx + 1] - xi_offset;
    const bool has_xi = xis != nullptr;
    const FpExt *xi_slice = has_xi ? xis + xi_offset : nullptr;

    const FpExt one_ext = FpExt(Fp::one());
    const FpExt zero_ext = FpExt(Fp::zero());

    FpExt eq_acc = one_ext;
    for (uint32_t m = 0; m < local_n; ++m) {
        bool bit_m = !!(shifted_idx & (1u << m));
        bool use_xi_m = has_xi && (m >= record.n_lift) && (m < n_logup) && (l_skip + m < xi_len);
        FpExt xi_m = use_xi_m ? xi_slice[l_skip + m] : (bit_m ? one_ext : zero_ext);
        FpExt factor = bit_m ? xi_m : (one_ext - xi_m);
        eq_acc = eq_acc * factor;
    }

    bool use_xi_cur =
        has_xi && (local_n >= record.n_lift) && (local_n < n_logup) && (l_skip + local_n < xi_len);
    FpExt xi_cur = use_xi_cur ? xi_slice[l_skip + local_n] : (nth_bit ? one_ext : zero_ext);

    COL_WRITE_VALUE(row, Eq3bColumns, is_valid, 1);
    COL_WRITE_VALUE(row, Eq3bColumns, is_first, (record_idx == 0) && (local_n == 0));
    COL_WRITE_VALUE(row, Eq3bColumns, proof_idx, proof_idx);
    COL_WRITE_VALUE(row, Eq3bColumns, sort_idx, record.sort_idx);
    COL_WRITE_VALUE(row, Eq3bColumns, interaction_idx, record.interaction_idx);
    COL_WRITE_VALUE(row, Eq3bColumns, n_lift, record.n_lift);
    COL_WRITE_VALUE(row, Eq3bColumns, two_to_the_n_lift, 1u << record.n_lift);
    COL_WRITE_VALUE(row, Eq3bColumns, n, local_n);
    COL_WRITE_VALUE(row, Eq3bColumns, hypercube_volume, 1u << local_n);
    COL_WRITE_VALUE(row, Eq3bColumns, has_no_interactions, record.no_interactions);
    COL_WRITE_VALUE(row, Eq3bColumns, n_at_least_n_lift, local_n >= record.n_lift);
    COL_WRITE_VALUE(
        row,
        Eq3bColumns,
        is_first_in_air,
        (record.interaction_idx == 0 && local_n == 0) || record.no_interactions
    );
    COL_WRITE_VALUE(
        row, Eq3bColumns, is_first_in_interaction, local_n == 0 || record.no_interactions
    );
    COL_WRITE_VALUE(row, Eq3bColumns, idx, shifted_idx & ((1u << local_n) - 1));
    COL_WRITE_VALUE(row, Eq3bColumns, running_idx, shifted_idx);
    COL_WRITE_VALUE(row, Eq3bColumns, nth_bit, nth_bit);
    COL_WRITE_VALUE(
        row,
        Eq3bColumns,
        loop_transitions[0],
        (record_idx + 1) < record_count || (local_n < n_logup && !record.no_interactions)
    );
    COL_WRITE_VALUE(
        row,
        Eq3bColumns,
        loop_transitions[1],
        (!record.is_last_in_air || local_n < n_logup) && !record.no_interactions
    );
    COL_WRITE_ARRAY(row, Eq3bColumns, xi, xi_cur.elems);
    COL_WRITE_ARRAY(row, Eq3bColumns, eq, eq_acc.elems);
}

extern "C" int _eq_3b_tracegen(
    Fp *trace_d,
    size_t num_valid_rows,
    size_t height,
    size_t num_proofs,
    size_t l_skip,
    const Eq3bStackedIdxRecord *records_d,
    const size_t *record_bounds_d,
    const RecordIdx *record_idxs_d,
    const size_t *record_idxs_bounds_d,
    const size_t *rows_per_proof_bounds_d,
    const size_t *n_logups_d,
    const FpExt *xis_d,
    const size_t *xi_bounds_d
) {
    assert((height & (height - 1)) == 0);
    auto [grid, block] = kernel_launch_params(height);
    eq_3b_tracegen<<<grid, block>>>(
        trace_d,
        num_valid_rows,
        height,
        num_proofs,
        l_skip,
        records_d,
        record_bounds_d,
        record_idxs_d,
        record_idxs_bounds_d,
        rows_per_proof_bounds_d,
        n_logups_d,
        xis_d,
        xi_bounds_d
    );
    return CHECK_KERNEL();
}
