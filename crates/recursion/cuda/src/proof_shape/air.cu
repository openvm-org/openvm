#include "checker.cuh"
#include "fp.h"
#include "launcher.cuh"
#include "primitives/encoder.cuh"
#include "primitives/trace_access.h"
#include "ptr_array.h"
#include "switch_macro.h"
#include "types.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

const size_t NUM_LIMBS = 4;
const size_t LIMB_BITS = 8;
typedef uint32_t Decomp[NUM_LIMBS];

struct ProofShapePerProof {
    size_t num_present;
    size_t n_max;
    size_t n_logup;
    size_t final_cidx;
    size_t final_total_interactions;
    Digest main_commit;
};

struct ProofShapeTracegenInputs {
    size_t num_airs;
    size_t l_skip;
    size_t max_cached;
    size_t min_cached_idx;
    Digest pre_hash;
    uint32_t *range_checker_8_ptr;
    uint32_t *range_checker_5_ptr;
    uint32_t *pow_checker_ptr;
};

template <typename T, size_t MAX_CACHED> struct ProofShapeCols {
    T proof_idx;
    T is_valid;
    T is_first;
    T is_last;

    T idx;
    T sorted_idx;
    T log_height;
    T n_sign_bit;

    T starting_tidx;
    T starting_cidx;

    T is_present;
    T height;

    T lifted_height_limbs[NUM_LIMBS];
    T num_interactions_limbs[NUM_LIMBS];
    T total_interactions_limbs[NUM_LIMBS];

    T n_max;
    T is_n_max_greater;

    T num_air_id_lookups;

    // T idx_flags[IDX_FLAGS];
    T cached_commits[MAX_CACHED][DIGEST_SIZE];
};

__device__ __forceinline__ void decompose(Decomp decomp, size_t value) {
    size_t mask = (1 << LIMB_BITS) - 1;
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        decomp[i] = (value >> (i * LIMB_BITS)) & mask;
    }
}

template <size_t MAX_CACHED> struct Cols {
    template <typename T> using Type = ProofShapeCols<T, MAX_CACHED>;
};

template <size_t MAX_CACHED>
__device__ __forceinline__ void fill_present_row(
    RowSlice row,
    AirData &air_data,
    TraceMetadata &trace_data,
    Digest *cached_commits,
    size_t l_skip,
    size_t cached_commits_idx,
    size_t min_cached_idx,
    RangeChecker &range_checker,
    PowerChecker<32> &pow_checker
) {
    size_t log_height = static_cast<size_t>(trace_data.log_height);
    int32_t n = static_cast<int32_t>(log_height) - static_cast<int32_t>(l_skip);
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, log_height, log_height);
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, n_sign_bit, n < 0 ? 1 : 0);

    COL_WRITE_VALUE(
        row, typename Cols<MAX_CACHED>::template Type, starting_cidx, trace_data.starting_cidx
    );

    size_t height = 1 << log_height;
    size_t lifted_height = max(height, (size_t)(1 << l_skip));
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, is_present, Fp::one());
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, height, height);
    COL_WRITE_VALUE(
        row,
        typename Cols<MAX_CACHED>::template Type,
        num_air_id_lookups,
        trace_data.num_air_id_lookups
    );

    Decomp lifted_height_decomp, num_interactions_decomp, total_interactions_decomp;
    decompose(lifted_height_decomp, lifted_height);
    decompose(num_interactions_decomp, lifted_height * air_data.num_interactions_per_row);
    decompose(total_interactions_decomp, trace_data.total_interactions);
    COL_WRITE_ARRAY(
        row, typename Cols<MAX_CACHED>::template Type, lifted_height_limbs, lifted_height_decomp
    );
    COL_WRITE_ARRAY(
        row,
        typename Cols<MAX_CACHED>::template Type,
        num_interactions_limbs,
        num_interactions_decomp
    );
    COL_WRITE_ARRAY(
        row,
        typename Cols<MAX_CACHED>::template Type,
        total_interactions_limbs,
        total_interactions_decomp
    );
    decompose(
        total_interactions_decomp,
        trace_data.total_interactions + lifted_height * air_data.num_interactions_per_row
    );

#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        range_checker.add_count(lifted_height_decomp[i]);
        range_checker.add_count(total_interactions_decomp[i]);
    }

    size_t non_zero_idx = 0;
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        if (lifted_height_decomp[i] != 0) {
            non_zero_idx = i;
            break;
        }
    }
    uint32_t height_limb = lifted_height_decomp[non_zero_idx];

    uint32_t carry = 0;
    const uint32_t MASK = (1 << LIMB_BITS) - 1;

#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS - 1; i++) {
        if (i < non_zero_idx) {
            range_checker.add_count(0);
        } else {
            uint32_t interactions_per_row_limb =
                static_cast<uint32_t>(
                    air_data.num_interactions_per_row >> ((i - non_zero_idx) * LIMB_BITS)
                ) &
                MASK;
            carry += height_limb * interactions_per_row_limb;
            carry = (carry - num_interactions_decomp[i]) >> LIMB_BITS;
            range_checker.add_count(carry);
        }
    }

    pow_checker.add_range_count(static_cast<uint32_t>(abs(n)));
    pow_checker.add_pow_count(log_height);

#pragma unroll
    for (size_t i = 0; i < MAX_CACHED; i++) {
        size_t commit_idx = cached_commits_idx + DIGEST_SIZE * i;
        if (i < air_data.num_cached) {
            row.write_array(commit_idx, DIGEST_SIZE, cached_commits[trace_data.cached_idx + i]);
        } else {
            if (i + 1 != MAX_CACHED || min_cached_idx != trace_data.air_idx) {
                row.fill_zero(commit_idx, DIGEST_SIZE);
            }
        }
    }
}

template <size_t MAX_CACHED>
__device__ __forceinline__ void fill_non_present_row(
    RowSlice row,
    size_t final_cidx,
    size_t final_total_interactions,
    size_t cached_commits_idx,
    RangeChecker &range_checker
) {
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, log_height, Fp::zero());
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, n_sign_bit, Fp::zero());
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, starting_cidx, final_cidx);
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, is_present, Fp::zero());
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, height, Fp::zero());
    row.fill_zero(
        COL_INDEX(typename Cols<MAX_CACHED>::template Type, lifted_height_limbs), NUM_LIMBS
    );
    row.fill_zero(
        COL_INDEX(typename Cols<MAX_CACHED>::template Type, num_interactions_limbs), NUM_LIMBS
    );
    row.fill_zero(cached_commits_idx, MAX_CACHED * DIGEST_SIZE);

    Decomp total_interactions;
    decompose(total_interactions, final_total_interactions);
    COL_WRITE_ARRAY(
        row, typename Cols<MAX_CACHED>::template Type, total_interactions_limbs, total_interactions
    );

#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        range_checker.add_count(total_interactions[i]);
    }

#pragma unroll
    for (size_t i = 0; i < 2 * NUM_LIMBS - 1; i++) {
        range_checker.add_count(0);
    }
}

template <size_t MAX_CACHED>
__device__ __forceinline__ void fill_summary_row(
    RowSlice row,
    size_t final_total_interactions,
    size_t cached_commits_idx,
    size_t n_max,
    size_t n_logup,
    Digest &pre_hash,
    RangeChecker &range_checker,
    PowerChecker<32> &pow_checker
) {
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, is_valid, Fp::zero());
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, is_last, Fp::one());
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, idx, Fp::zero());
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, sorted_idx, Fp::zero());
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, is_present, Fp::zero());
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, n_sign_bit, Fp::zero());
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, num_air_id_lookups, Fp::zero());
    row.fill_zero(cached_commits_idx, MAX_CACHED * DIGEST_SIZE);

    Decomp interaction_decomp, max_interaction_decomp;
    decompose(interaction_decomp, final_total_interactions);
    decompose(max_interaction_decomp, Fp::P);

    size_t nonzero_idx = 0;
    size_t diff_idx = 0;
#pragma unroll
    for (int i = NUM_LIMBS - 1; i >= 0; i--) {
        if (interaction_decomp[i] != 0 && nonzero_idx == 0) {
            nonzero_idx = i;
        }
        if (interaction_decomp[i] != max_interaction_decomp[i] && diff_idx == 0) {
            diff_idx = i;
        }
    }

    size_t msb_limb_zero_bits = 0;
    if (final_total_interactions > 0) {
        msb_limb_zero_bits = LIMB_BITS - (32 - __clz(interaction_decomp[nonzero_idx]));
    }

    // limb_to_range_check
    COL_WRITE_VALUE(
        row, typename Cols<MAX_CACHED>::template Type, height, interaction_decomp[nonzero_idx]
    );
    // msb_limb_zero_bits_exp
    COL_WRITE_VALUE(
        row, typename Cols<MAX_CACHED>::template Type, log_height, 1 << msb_limb_zero_bits
    );
#pragma unroll
    for (int i = 0; i < NUM_LIMBS; i++) {
        // non_zero_marker
        row.write(
            COL_INDEX(typename Cols<MAX_CACHED>::template Type, lifted_height_limbs) + i,
            i == nonzero_idx && final_total_interactions > 0
        );
        // diff_marker
        row.write(
            COL_INDEX(typename Cols<MAX_CACHED>::template Type, num_interactions_limbs) + i,
            i == diff_idx
        );
    }

    COL_WRITE_ARRAY(
        row, typename Cols<MAX_CACHED>::template Type, total_interactions_limbs, interaction_decomp
    );
    COL_WRITE_VALUE(
        row, typename Cols<MAX_CACHED>::template Type, is_n_max_greater, n_max > n_logup
    );

    // n_logup
    COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, starting_cidx, n_logup);

    range_checker.add_count(interaction_decomp[nonzero_idx] * (1 << msb_limb_zero_bits));
    range_checker.add_count(max_interaction_decomp[diff_idx] - interaction_decomp[diff_idx] - 1);
    pow_checker.add_pow_count(msb_limb_zero_bits);
    pow_checker.add_range_count(n_max > n_logup ? n_max - n_logup : n_logup - n_max);

    row.write_array(cached_commits_idx + DIGEST_SIZE * (MAX_CACHED - 1), DIGEST_SIZE, pre_hash);
}

template <size_t NUM_PROOFS, size_t MAX_CACHED>
__global__ void proof_shape_tracegen(
    Fp *trace,
    size_t height,
    AirData *air_data,
    PtrArray<size_t, NUM_PROOFS> per_row_tidx,
    PtrArray<TraceMetadata, NUM_PROOFS> sorted_trace_data,
    PtrArray<Digest, NUM_PROOFS> cached_commits,
    ProofShapePerProof *per_proof,
    ProofShapeTracegenInputs inputs
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + row_idx, height);

    if (row_idx < NUM_PROOFS * (inputs.num_airs + 1)) {
        size_t proof_idx = row_idx / (inputs.num_airs + 1);
        size_t record_idx = row_idx % (inputs.num_airs + 1);
        ProofShapePerProof proof_data = per_proof[proof_idx];

        COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, proof_idx, proof_idx);
        COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, is_first, record_idx == 0);
        COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, n_max, proof_data.n_max);

        Encoder encoder(inputs.num_airs, 2, true);
        size_t encoder_flags_idx =
            COL_INDEX(typename Cols<MAX_CACHED>::template Type, cached_commits);
        size_t cached_commits_idx = encoder_flags_idx + encoder.width();

        RangeChecker range_checker(inputs.range_checker_8_ptr, LIMB_BITS);
        PowerChecker<32> pow_checker(inputs.pow_checker_ptr, inputs.range_checker_5_ptr);

        if (record_idx == inputs.num_airs) {
            COL_WRITE_VALUE(
                row,
                typename Cols<MAX_CACHED>::template Type,
                starting_tidx,
                per_row_tidx[proof_idx][record_idx]
            );
            fill_summary_row<MAX_CACHED>(
                row,
                proof_data.final_total_interactions,
                cached_commits_idx,
                proof_data.n_max,
                proof_data.n_logup,
                inputs.pre_hash,
                range_checker,
                pow_checker
            );
            row.fill_zero(encoder_flags_idx, encoder.width());
        } else {
            TraceMetadata trace_data = sorted_trace_data[proof_idx][record_idx];

            COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, is_valid, Fp::one());
            COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, is_last, Fp::zero());
            COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, idx, trace_data.air_idx);
            COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, sorted_idx, record_idx);

            COL_WRITE_VALUE(
                row,
                typename Cols<MAX_CACHED>::template Type,
                starting_tidx,
                per_row_tidx[proof_idx][trace_data.air_idx]
            );

            COL_WRITE_VALUE(
                row, typename Cols<MAX_CACHED>::template Type, is_n_max_greater, Fp::zero()
            );

            encoder.write_flag_pt(row.slice_from(encoder_flags_idx), trace_data.air_idx);

            if (inputs.min_cached_idx == trace_data.air_idx) {
                row.write_array(
                    cached_commits_idx + DIGEST_SIZE * (MAX_CACHED - 1),
                    DIGEST_SIZE,
                    proof_data.main_commit
                );
            }

            if (record_idx + 1 < inputs.num_airs) {
                uint8_t current_log_height = trace_data.log_height;
                uint8_t next_log_height =
                    record_idx + 1 < proof_data.num_present
                        ? sorted_trace_data[proof_idx][record_idx + 1].log_height
                        : 0;
                pow_checker.add_range_count(
                    static_cast<uint32_t>(current_log_height - next_log_height)
                );
            }

            if (record_idx < proof_data.num_present) {
                fill_present_row<MAX_CACHED>(
                    row,
                    air_data[trace_data.air_idx],
                    trace_data,
                    cached_commits[proof_idx],
                    inputs.l_skip,
                    cached_commits_idx,
                    inputs.min_cached_idx,
                    range_checker,
                    pow_checker
                );
            } else {
                fill_non_present_row<MAX_CACHED>(
                    row,
                    proof_data.final_cidx,
                    proof_data.final_total_interactions,
                    cached_commits_idx,
                    range_checker
                );
            }
        }
    } else {
        COL_WRITE_VALUE(row, typename Cols<MAX_CACHED>::template Type, proof_idx, NUM_PROOFS);
        row.fill_zero(1, sizeof(ProofShapeCols<uint8_t, MAX_CACHED>));
    }
}

extern "C" int _proof_shape_tracegen(
    Fp *d_trace,
    size_t height,
    AirData *d_air_data,
    size_t **d_per_row_tidx,
    TraceMetadata **d_sorted_trace_data,
    Digest **d_cached_commits,
    ProofShapePerProof *d_per_proof,
    size_t num_proofs,
    ProofShapeTracegenInputs *inputs
) {
    assert((height & (height - 1)) == 0);
    auto [grid, block] = kernel_launch_params(height);
    SWITCH_BLOCK(
        num_proofs,
        NUM_PROOFS,
        (SWITCH_BLOCK(
            inputs->max_cached,
            MAX_CACHED,
            (proof_shape_tracegen<NUM_PROOFS, MAX_CACHED><<<grid, block>>>(
                 d_trace,
                 height,
                 d_air_data,
                 PtrArray<size_t, NUM_PROOFS>(d_per_row_tidx),
                 PtrArray<TraceMetadata, NUM_PROOFS>(d_sorted_trace_data),
                 PtrArray<Digest, NUM_PROOFS>(d_cached_commits),
                 d_per_proof,
                 *inputs
            );),
            1,
            2
        )),
        1,
        2,
        3,
        4,
        5
    )
    return CHECK_KERNEL();
}
