#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "primitives/trace_access.h"
#include "switch_macro.h"
#include "types.h"
#include "util.cuh"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

template <typename T> struct InitialOpenedValuesCols {
    T proof_idx;
    T query_idx;
    T commit_idx;
    T coset_idx;
    T col_chunk_idx;
    T is_first_in_proof;
    T is_first_in_query;
    T is_first_in_commit;
    T is_first_in_coset;
    T flags[CHUNK];
    T codeword_value_acc[D_EF];
    T codeword_value_next_acc[D_EF];
    T mu_pows_even_clamped[CHUNK / 2][D_EF];
    T mu_pow_last_clamped[D_EF];
    T mu[D_EF];
    T pre_state[WIDTH];
    T post_state[WIDTH];
    T twiddle;
    T zi_root;
    T zi;
    T yi[D_EF];
    T merkle_idx_bit_src;
};

typedef struct {
    Fp pre_state[WIDTH];
    Fp post_state[WIDTH];
} PoseidonStatePair;

template <size_t NUM_PROOFS>
__global__ void initial_opened_values_tracegen(
    Fp *trace,
    size_t num_valid_rows,
    size_t height,
    FpExt *codeword_value_accs,
    PoseidonStatePair *poseidon_states,
    size_t k_whir,
    size_t num_initial_queries,
    size_t total_queries,
    Fp omega_k,
    FpExt *mus_per_proof,
    Fp *zis_per_proof,
    Fp *zi_roots_per_proof,
    FpExt *yis_per_proof,
    Fp *raw_queries,
    size_t *rows_per_proof_psums,
    size_t *commits_per_proof_psums,
    size_t *stacking_chunks_psums,
    size_t *stacking_widths_psums,
    FpExt *mu_pows
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= height)
        return;

    RowSlice row(trace + row_idx, height);

    if (row_idx >= num_valid_rows) {
        row.fill_zero(0, sizeof(InitialOpenedValuesCols<uint8_t>));
        return;
    }

    // rows_per_proof_psums has length NUM_PROOFS + 1, with psums[0] = 0.
    const size_t proof_idx = partition_point_leq(
        rows_per_proof_psums + 1,
        NUM_PROOFS,
        static_cast<size_t>(row_idx)
    );

    const size_t record_idx = row_idx - rows_per_proof_psums[proof_idx];

    const size_t cp_start = commits_per_proof_psums[proof_idx];
    const size_t cp_end = commits_per_proof_psums[proof_idx + 1];

    const size_t chunks_before_proof = stacking_chunks_psums[cp_start];
    const size_t chunks_after_proof = stacking_chunks_psums[cp_end];
    const size_t records_per_coset_idx = chunks_after_proof - chunks_before_proof;

    const size_t coset_span = (1 << k_whir);

    const size_t coset_idx = (record_idx / records_per_coset_idx) % coset_span;
    const size_t query_idx = (record_idx / (records_per_coset_idx * coset_span)) % num_initial_queries;

    const size_t local_chunk_idx = record_idx % records_per_coset_idx;
    const size_t absolute_chunk_idx = chunks_before_proof + local_chunk_idx;

    const size_t commit_idx = partition_point_leq(
        stacking_chunks_psums + cp_start + 1,
        cp_end - cp_start,
        absolute_chunk_idx
    );

    const size_t commit_chunks_before = stacking_chunks_psums[cp_start + commit_idx];
    const size_t chunk_idx = absolute_chunk_idx - commit_chunks_before;

    const bool is_first_in_commit = chunk_idx == 0;
    const bool is_first_in_coset = is_first_in_commit && commit_idx == 0;
    const bool is_first_in_query = is_first_in_coset && coset_idx == 0;
    const bool is_first_in_proof = is_first_in_query && query_idx == 0;

    const size_t num_chunks =
        stacking_chunks_psums[cp_start + commit_idx + 1] -
        stacking_chunks_psums[cp_start + commit_idx];
    const bool is_same_commit = chunk_idx + 1 < num_chunks;

    size_t chunk_len;
    if (is_same_commit) {
        chunk_len = CHUNK;
    } else {
        const size_t total_width = stacking_widths_psums[cp_start + commit_idx + 1] -
                                   stacking_widths_psums[cp_start + commit_idx];
        chunk_len = (total_width % CHUNK ?: CHUNK);
    };

    COL_WRITE_VALUE(row, InitialOpenedValuesCols, proof_idx, proof_idx);
    COL_WRITE_VALUE(row, InitialOpenedValuesCols, query_idx, query_idx);
    COL_WRITE_VALUE(row, InitialOpenedValuesCols, coset_idx, coset_idx);
    COL_WRITE_VALUE(row, InitialOpenedValuesCols, commit_idx, commit_idx);
    COL_WRITE_VALUE(row, InitialOpenedValuesCols, col_chunk_idx, chunk_idx);

    COL_WRITE_VALUE(row, InitialOpenedValuesCols, is_first_in_proof, is_first_in_proof);
    COL_WRITE_VALUE(row, InitialOpenedValuesCols, is_first_in_query, is_first_in_query);
    COL_WRITE_VALUE(row, InitialOpenedValuesCols, is_first_in_coset, is_first_in_coset);
    COL_WRITE_VALUE(row, InitialOpenedValuesCols, is_first_in_commit, is_first_in_commit);

    for (int i = 0; i < chunk_len; i++) {
        COL_WRITE_VALUE(row, InitialOpenedValuesCols, flags[i], Fp::one());
    }
    row.fill_zero(
        offsetof(InitialOpenedValuesCols<uint8_t>, flags) + chunk_len,
        CHUNK - chunk_len
    );

    Fp twiddle = pow(omega_k, coset_idx);
    COL_WRITE_VALUE(row, InitialOpenedValuesCols, twiddle, twiddle);

    // For round 0, query_idx indexes directly (no round offset needed)
    const size_t proof_query_idx = proof_idx * total_queries + query_idx;

    COL_WRITE_ARRAY(row, InitialOpenedValuesCols, codeword_value_acc, codeword_value_accs[row_idx].elems);
    COL_WRITE_VALUE(row, InitialOpenedValuesCols, zi, zis_per_proof[proof_query_idx]);
    COL_WRITE_VALUE(row, InitialOpenedValuesCols, zi_root, zi_roots_per_proof[proof_query_idx]);
    COL_WRITE_ARRAY(row, InitialOpenedValuesCols, yi, yis_per_proof[proof_query_idx].elems);
    COL_WRITE_VALUE(
        row,
        InitialOpenedValuesCols,
        merkle_idx_bit_src,
        raw_queries[proof_query_idx]
    );

    FpExt mu = mus_per_proof[proof_idx];
    COL_WRITE_ARRAY(row, InitialOpenedValuesCols, mu, mu.elems);

    size_t width_before_proof = stacking_widths_psums[cp_start];
    size_t exponent_base =
        stacking_widths_psums[cp_start + commit_idx] - width_before_proof;

    const size_t chunk_base = exponent_base + chunk_idx * CHUNK;

    // Fill mu_pows_even_clamped[k] = mu^(min(b + 2k, opened_row_len - 1))
    for (int k = 0; k < CHUNK / 2; k++) {
        size_t offset = 2 * k;
        size_t exponent = chunk_base + (offset < chunk_len - 1 ? offset : chunk_len - 1);
        FpExt mu_pow = mu_pows[width_before_proof + exponent];
        COL_WRITE_ARRAY(row, InitialOpenedValuesCols, mu_pows_even_clamped[k], mu_pow.elems);
    }

    // Fill mu_pow_last_clamped = mu^(b + chunk_len - 1)
    {
        size_t exponent = chunk_base + chunk_len - 1;
        FpExt mu_pow = mu_pows[width_before_proof + exponent];
        COL_WRITE_ARRAY(row, InitialOpenedValuesCols, mu_pow_last_clamped, mu_pow.elems);
    }

    PoseidonStatePair state_pair = poseidon_states[row_idx];
    COL_WRITE_ARRAY(row, InitialOpenedValuesCols, pre_state, state_pair.pre_state);
    COL_WRITE_ARRAY(row, InitialOpenedValuesCols, post_state, state_pair.post_state);

    // Compute codeword_value_next_acc = codeword_value_acc + sum_i(mu_pows[i] * pre_state[i])
    FpExt next_acc = codeword_value_accs[row_idx];
    for (size_t i = 0; i < chunk_len; i++) {
        size_t exponent = chunk_base + i;
        FpExt mu_pow_i = mu_pows[width_before_proof + exponent];
        next_acc += mu_pow_i * FpExt(state_pair.pre_state[i]);
    }
    COL_WRITE_ARRAY(row, InitialOpenedValuesCols, codeword_value_next_acc, next_acc.elems);
}

extern "C" int _initial_opened_values_tracegen(
    Fp *trace_d,
    size_t num_valid_rows,
    size_t height,
    FpExt *codeword_value_accs_d,
    PoseidonStatePair *poseidon_states_d,
    size_t k_whir,
    size_t num_initial_queries,
    size_t total_queries,
    Fp omega_k,
    FpExt *mu_per_proof,
    Fp *zi_d,
    Fp *zi_roots_d,
    FpExt *yi_d,
    Fp *merkle_idx_bit_src_d,
    size_t *rows_per_proof_psums,
    size_t *commits_per_proof_psums,
    size_t *stacking_chunks_psums_per_proof,
    size_t *stacking_widths_psums_per_proof,
    FpExt *mu_pows,
    size_t num_proofs
) {
    assert((height & (height - 1)) == 0);
    auto [grid, block] = kernel_launch_params(height, 512);

    SWITCH_BLOCK(num_proofs, NUM_PROOFS, (initial_opened_values_tracegen<NUM_PROOFS><<<grid, block>>>(
        trace_d,
        num_valid_rows,
        height,
        codeword_value_accs_d,
        poseidon_states_d,
        k_whir,
        num_initial_queries,
        total_queries,
        omega_k,
        mu_per_proof,
        zi_d,
        zi_roots_d,
        yi_d,
        merkle_idx_bit_src_d,
        rows_per_proof_psums,
        commits_per_proof_psums,
        stacking_chunks_psums_per_proof,
        stacking_widths_psums_per_proof,
        mu_pows
    );),
    1, 2, 3, 4, 5, 6, 7, 8
    )
    return CHECK_KERNEL();
}
