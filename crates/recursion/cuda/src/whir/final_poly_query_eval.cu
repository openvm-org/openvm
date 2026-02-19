#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "primitives/trace_access.h"
#include "types.h"
#include "util.cuh"
#include <cassert>
#include <cstddef>
#include <cstdint>

template <typename T> struct FinalPolyQueryEvalCols {
    T is_enabled;
    T proof_idx;
    T whir_round;
    T query_idx;
    T phase_idx;
    T eval_idx;
    T is_first_in_proof;
    T is_first_in_round;
    T is_first_in_query;
    T is_first_in_phase;
    T is_last_round;
    T is_query_zero;
    T query_pow[D_EF];
    T alpha[D_EF];
    T gamma[D_EF];
    T gamma_pow[D_EF];
    T final_poly_coeff[D_EF];
    T final_value_acc[D_EF];
    T gamma_eq_acc[D_EF];
    T horner_acc[D_EF];
    T do_carry;
};

typedef struct {
    FpExt alpha;
    FpExt query_pow;
    FpExt gamma_eq_acc;
    FpExt horner_acc;
    FpExt final_poly_coeff;
    FpExt final_value_acc;
    FpExt gamma_pow;
} FinalPolyQueryEvalRecord;

__global__ void final_poly_query_eval_tracegen(
    Fp *trace,
    size_t num_valid_rows,
    size_t height,
    const FinalPolyQueryEvalRecord *records,
    const FpExt *gammas,
    size_t num_whir_rounds,
    size_t rows_per_proof,
    const size_t *round_offsets,
    size_t log_final_poly_len,
    const size_t *num_queries_per_round
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= height)
        return;

    RowSlice row(trace + row_idx, height);

    if (row_idx >= num_valid_rows) {
        row.fill_zero(0, sizeof(FinalPolyQueryEvalCols<uint8_t>));
        return;
    }

    const size_t row_idx_usize = static_cast<size_t>(row_idx);
    const FinalPolyQueryEvalRecord record = records[row_idx_usize];
    const size_t final_poly_len = 1ull << log_final_poly_len;
    assert(rows_per_proof > 0);
    const size_t proof_idx = row_idx_usize / rows_per_proof;
    const size_t row_in_proof = row_idx_usize - proof_idx * rows_per_proof;

    const size_t whir_round_idx =
        partition_point_leq(round_offsets + 1, num_whir_rounds, row_in_proof);
    const FpExt gamma = gammas[proof_idx * num_whir_rounds + whir_round_idx];
    const size_t num_in_domain_queries = num_queries_per_round[whir_round_idx];
    const size_t query_count = num_in_domain_queries + 1;
    const size_t round_start = round_offsets[whir_round_idx];
    const size_t row_in_round = row_in_proof - round_start;
    const size_t rows_per_round = round_offsets[whir_round_idx + 1] - round_start;
    assert(rows_per_round % query_count == 0);
    const size_t rows_per_query = rows_per_round / query_count;
    assert(rows_per_query >= final_poly_len);
    const size_t eq_phase_len = rows_per_query - final_poly_len;
    const size_t query_idx = row_in_round / rows_per_query;
    const size_t row_in_query = row_in_round % rows_per_query;

    const size_t phase_idx = (row_in_query < eq_phase_len) ? 0 : 1;
    const size_t eval_idx = row_in_query - ((row_in_query < eq_phase_len) ? 0 : eq_phase_len);
    const size_t num_alphas = eq_phase_len;

    const bool is_first_in_phase = eval_idx == 0;
    const bool is_first_in_query = is_first_in_phase && (phase_idx == 0 || num_alphas == 0);
    const bool is_first_in_round = is_first_in_query && query_idx == 0;
    const bool is_first_in_proof = is_first_in_round && whir_round_idx == 0;

    const size_t is_same_phase = eval_idx + 1 < ((row_in_query < eq_phase_len) ? num_alphas : final_poly_len);
    const bool is_same_query = is_same_phase || phase_idx == 0;
    const bool is_same_round = is_same_query || query_idx < num_in_domain_queries;
    const bool is_same_proof = is_same_round || whir_round_idx + 1 < num_whir_rounds;

    const bool is_q0_last = query_idx == 0 && (whir_round_idx + 1 == num_whir_rounds);
    const bool do_carry = (!is_same_query) && is_same_proof && !is_q0_last;

    COL_WRITE_VALUE(row, FinalPolyQueryEvalCols, is_enabled, Fp::one());
    COL_WRITE_VALUE(row, FinalPolyQueryEvalCols, proof_idx, proof_idx);
    COL_WRITE_VALUE(row, FinalPolyQueryEvalCols, whir_round, whir_round_idx);
    COL_WRITE_VALUE(row, FinalPolyQueryEvalCols, query_idx, query_idx);
    COL_WRITE_VALUE(row, FinalPolyQueryEvalCols, phase_idx, phase_idx);
    COL_WRITE_VALUE(row, FinalPolyQueryEvalCols, eval_idx, eval_idx);
    COL_WRITE_VALUE(row, FinalPolyQueryEvalCols, is_first_in_proof, is_first_in_proof);
    COL_WRITE_VALUE(row, FinalPolyQueryEvalCols, is_first_in_round, is_first_in_round);
    COL_WRITE_VALUE(row, FinalPolyQueryEvalCols, is_first_in_query, is_first_in_query);
    COL_WRITE_VALUE(row, FinalPolyQueryEvalCols, is_first_in_phase, is_first_in_phase);
    COL_WRITE_VALUE(
        row,
        FinalPolyQueryEvalCols,
        is_last_round,
        whir_round_idx + 1 == num_whir_rounds
    );
    COL_WRITE_VALUE(row, FinalPolyQueryEvalCols, is_query_zero, query_idx == 0);
    COL_WRITE_VALUE(row, FinalPolyQueryEvalCols, do_carry, do_carry);

    COL_WRITE_ARRAY(row, FinalPolyQueryEvalCols, query_pow, record.query_pow.elems);
    COL_WRITE_ARRAY(row, FinalPolyQueryEvalCols, alpha, record.alpha.elems);
    COL_WRITE_ARRAY(row, FinalPolyQueryEvalCols, gamma, gamma.elems);
    COL_WRITE_ARRAY(row, FinalPolyQueryEvalCols, gamma_pow, record.gamma_pow.elems);
    COL_WRITE_ARRAY(
        row,
        FinalPolyQueryEvalCols,
        final_poly_coeff,
        record.final_poly_coeff.elems
    );
    COL_WRITE_ARRAY(
        row,
        FinalPolyQueryEvalCols,
        final_value_acc,
        record.final_value_acc.elems
    );
    COL_WRITE_ARRAY(
        row,
        FinalPolyQueryEvalCols,
        gamma_eq_acc,
        record.gamma_eq_acc.elems
    );
    COL_WRITE_ARRAY(row, FinalPolyQueryEvalCols, horner_acc, record.horner_acc.elems);
}

extern "C" int _final_poly_query_eval_tracegen(
    Fp *trace_d,
    size_t num_valid_rows,
    size_t height,
    const FinalPolyQueryEvalRecord *records_d,
    const FpExt *gammas_d,
    size_t num_whir_rounds,
    size_t rows_per_proof,
    const size_t *round_offsets_d,
    size_t log_final_poly_len,
    const size_t *num_queries_per_round_d
) {
    assert((height & (height - 1)) == 0);
    auto [grid, block] = kernel_launch_params(height, 512);

    final_poly_query_eval_tracegen<<<grid, block>>>(
        trace_d,
        num_valid_rows,
        height,
        records_d,
        gammas_d,
        num_whir_rounds,
        rows_per_proof,
        round_offsets_d,
        log_final_poly_len,
        num_queries_per_round_d
    );
    return CHECK_KERNEL();
}
