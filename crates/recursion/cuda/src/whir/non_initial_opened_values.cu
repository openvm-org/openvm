#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "primitives/trace_access.h"
#include "types.h"
#include "util.cuh"
#include <cassert>
#include <cstddef>
#include <cstdint>

template <typename T> struct NonInitialOpenedValuesCols {
    T is_enabled;
    T proof_idx;
    T whir_round;
    T query_idx;
    T coset_idx;
    T is_first_in_proof;
    T is_first_in_round;
    T is_first_in_query;
    T merkle_idx_bit_src;
    T zi_root;
    T zi;
    T twiddle;
    T value[D_EF];
    T value_hash[WIDTH];
    T yi[D_EF];
};

__global__ void non_initial_opened_values_tracegen(
    Fp *trace,
    size_t num_valid_rows,
    size_t height,
    const FpExt *codeword_opened_values,
    const Fp *codeword_states,
    size_t num_whir_rounds,
    size_t k_whir,
    Fp omega_k,
    const Fp *zis,
    const Fp *zi_roots,
    const FpExt *yis,
    const Fp *raw_queries,
    const size_t *round_row_offsets,
    size_t rows_per_proof,
    const size_t *query_offsets,
    size_t total_queries
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= height)
        return;

    RowSlice row(trace + row_idx, height);

    if (row_idx >= num_valid_rows) {
        row.fill_zero(0, sizeof(NonInitialOpenedValuesCols<uint8_t>));
        return;
    }

    const size_t rows_per_query = 1ull << k_whir;
    assert(rows_per_proof > 0);

    const size_t row_idx_usize = static_cast<size_t>(row_idx);
    const size_t proof_idx = row_idx_usize / rows_per_proof;
    const size_t row_in_proof = row_idx_usize - proof_idx * rows_per_proof;

    // Find whir_round using round_row_offsets (num_whir_rounds - 1 entries for rounds 1..num_rounds)
    const size_t round_minus_1 =
        partition_point_leq(round_row_offsets + 1, num_whir_rounds - 1, row_in_proof);
    const size_t whir_round = round_minus_1 + 1;
    const size_t row_in_round = row_in_proof - round_row_offsets[round_minus_1];

    const size_t query_idx = row_in_round / rows_per_query;
    const size_t coset_idx = row_in_round % rows_per_query;

    const bool is_first_in_proof = row_in_proof == 0;
    const bool is_first_in_query = coset_idx == 0;
    const bool is_first_in_round = is_first_in_query && query_idx == 0;

    // Use query_offsets for proper indexing into flattened arrays
    const size_t per_proof_offset = proof_idx * total_queries;
    const size_t round_query_idx =
        per_proof_offset + query_offsets[whir_round] + query_idx;

    COL_WRITE_VALUE(row, NonInitialOpenedValuesCols, is_enabled, Fp::one());
    COL_WRITE_VALUE(row, NonInitialOpenedValuesCols, proof_idx, proof_idx);
    COL_WRITE_VALUE(row, NonInitialOpenedValuesCols, whir_round, whir_round);
    COL_WRITE_VALUE(row, NonInitialOpenedValuesCols, query_idx, query_idx);
    COL_WRITE_VALUE(row, NonInitialOpenedValuesCols, coset_idx, coset_idx);
    COL_WRITE_VALUE(row, NonInitialOpenedValuesCols, is_first_in_proof, is_first_in_proof);
    COL_WRITE_VALUE(row, NonInitialOpenedValuesCols, is_first_in_round, is_first_in_round);
    COL_WRITE_VALUE(row, NonInitialOpenedValuesCols, is_first_in_query, is_first_in_query);
    COL_WRITE_VALUE(
        row,
        NonInitialOpenedValuesCols,
        merkle_idx_bit_src,
        raw_queries[round_query_idx]
    );
    COL_WRITE_VALUE(row, NonInitialOpenedValuesCols, zi_root, zi_roots[round_query_idx]);
    COL_WRITE_VALUE(row, NonInitialOpenedValuesCols, zi, zis[round_query_idx]);

    Fp twiddle = pow(omega_k, coset_idx);
    COL_WRITE_VALUE(row, NonInitialOpenedValuesCols, twiddle, twiddle);

    COL_WRITE_ARRAY(row, NonInitialOpenedValuesCols, value, codeword_opened_values[row_idx].elems);
    COL_WRITE_ARRAY(row, NonInitialOpenedValuesCols, value_hash, codeword_states + row_idx * WIDTH);

    const FpExt yi = yis[round_query_idx];
    COL_WRITE_ARRAY(row, NonInitialOpenedValuesCols, yi, yi.elems);
}

extern "C" int _non_initial_opened_values_tracegen(
    Fp *trace_d,
    size_t num_valid_rows,
    size_t height,
    const FpExt *codeword_opened_values_d,
    const Fp *codeword_states_d,
    size_t num_whir_rounds,
    size_t k_whir,
    Fp omega_k,
    const Fp *zis_d,
    const Fp *zi_roots_d,
    const FpExt *yis_d,
    const Fp *raw_queries_d,
    const size_t *round_row_offsets_d,
    size_t rows_per_proof,
    const size_t *query_offsets_d,
    size_t total_queries
) {
    assert((height & (height - 1)) == 0);
    auto [grid, block] = kernel_launch_params(height);

    non_initial_opened_values_tracegen<<<grid, block>>>(
        trace_d,
        num_valid_rows,
        height,
        codeword_opened_values_d,
        codeword_states_d,
        num_whir_rounds,
        k_whir,
        omega_k,
        zis_d,
        zi_roots_d,
        yis_d,
        raw_queries_d,
        round_row_offsets_d,
        rows_per_proof,
        query_offsets_d,
        total_queries
    );
    return CHECK_KERNEL();
}
