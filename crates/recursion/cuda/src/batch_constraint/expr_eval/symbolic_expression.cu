#include "dag_commit.cuh"
#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "primitives/encoder.cuh"
#include "primitives/trace_access.h"
#include "types.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

enum NodeKind : uint8_t {
    NODE_KIND_VAR_PREPROCESSED = 0,
    NODE_KIND_VAR_MAIN = 1,
    NODE_KIND_VAR_PUBLIC_VALUE = 2,
    NODE_KIND_IS_FIRST_ROW = 3,
    NODE_KIND_IS_LAST_ROW = 4,
    NODE_KIND_IS_TRANSITION = 5,
    NODE_KIND_CONSTANT = 6,
    NODE_KIND_ADD = 7,
    NODE_KIND_SUB = 8,
    NODE_KIND_NEG = 9,
    NODE_KIND_MUL = 10,
    NODE_KIND_INTERACTION_MULT = 11,
    NODE_KIND_INTERACTION_MSG_COMP = 12,
    NODE_KIND_INTERACTION_BUS_INDEX = 13,
    NODE_KIND_COUNT = 14,
};

struct FlatSymbolicConstraintNode {
    NodeKind kind;
    uint32_t data0;
    uint32_t data1;
    uint32_t data2;
    Fp constant;
};

struct FlatInteraction {
    uint32_t count;
    uint32_t message_start;
    uint32_t message_len;
    uint32_t bus_index;
    uint32_t count_weight;
};

struct FlatSymbolicVariable {
    uint8_t entry_kind;
    uint32_t index;
    uint32_t part_index;
    uint32_t offset;
};

struct CachedRecord {
    Fp poseidon2_input[WIDTH];
    bool is_constraint;
};

template <typename T> struct SymbolicExpressionCols {
    T is_present;
    T args[2 * D_EF];
    T sort_idx;
    T n_abs;
    T is_n_neg;
};

__device__ __forceinline__ FpExt pow_two_power(FpExt base, size_t exponent) {
    FpExt result = base;
    for (size_t i = 0; i < exponent; ++i) {
        result = result * result;
    }
    return result;
}

__device__ __forceinline__ FpExt eval_eq_uni_at_one_fp_ext(size_t l_skip, FpExt x) {
    FpExt res = FpExt(Fp::one());
    FpExt x_pow = x;
    const FpExt one_ext = FpExt(Fp::one());
    for (size_t i = 0; i < l_skip; ++i) {
        res *= x_pow + one_ext;
        x_pow *= x_pow;
    }

    return res * FpExt(Fp::one().mul_2exp_neg_n(l_skip));
}

__device__ __forceinline__ FpExt product_first_mle(const FpExt *rs_rest, size_t count) {
    FpExt acc = FpExt(Fp::one());
    const FpExt one_ext = FpExt(Fp::one());
    for (size_t i = 0; i < count; ++i) {
        acc *= one_ext - rs_rest[i];
    }
    return acc;
}

__device__ __forceinline__ FpExt product_last_mle(const FpExt *rs_rest, size_t count) {
    FpExt acc = FpExt(Fp::one());
    for (size_t i = 0; i < count; ++i) {
        acc *= rs_rest[i];
    }
    return acc;
}

__device__ __forceinline__ Fp two_adic_generator_at(size_t log_height) {
    size_t idx = log_height <= Fp::TWO_ADICITY ? log_height : Fp::TWO_ADICITY;
    return TWO_ADIC_GENERATORS[idx];
}

__device__ __forceinline__ void write_arg_first(RowSlice row, const FpExt &value) {
    row.write_array(COL_INDEX(SymbolicExpressionCols, args), D_EF, value.elems);
}

__device__ __forceinline__ void write_arg_second(RowSlice row, const FpExt &value) {
    row.write_array(COL_INDEX(SymbolicExpressionCols, args) + D_EF, D_EF, value.elems);
}

__global__ void symbolic_expression_tracegen(
    Fp *trace,
    size_t height,
    size_t l_skip,
    const size_t *log_heights,
    const size_t *sort_idx_by_air_idx,
    size_t num_airs,
    size_t num_proofs,
    size_t max_num_proofs,
    const FpExt *expr_evals,
    const size_t *expr_eval_bounds_0,
    const size_t *expr_eval_bounds_1,
    const FlatSymbolicConstraintNode *constraint_nodes,
    const size_t *constraint_nodes_bounds,
    const FlatInteraction *interactions,
    const size_t *interactions_bounds,
    const size_t *interaction_messages,
    const FlatSymbolicVariable *unused_variables,
    const size_t *unused_variables_bounds,
    const uint32_t *record_bounds,
    const uint32_t *air_ids_per_record,
    size_t num_records_per_proof,
    const FpExt *sumcheck_rnds,
    const size_t *sumcheck_bounds,
    const CachedRecord *cached_records
) {
    // Unused because expr_evals[i].len() is aleways the number of all airs,
    // plus 1 for unused variables
    (void)expr_eval_bounds_0;

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_slots = height * max_num_proofs;
    if (thread_idx >= total_slots) {
        return;
    }

    uint32_t row_idx = thread_idx / max_num_proofs;
    uint32_t proof_idx = thread_idx % max_num_proofs;

    RowSlice row(trace + row_idx, height);
    constexpr uint32_t SINGLE_WIDTH = sizeof(SymbolicExpressionCols<uint8_t>);
    constexpr uint32_t COMMIT_WIDTH = sizeof(DagCommitCols<uint8_t>);

    if (cached_records && proof_idx == 0) {
        CachedRecord record = cached_records[row_idx];
        write_dag_commit_poseidon2(row, record.poseidon2_input, record.is_constraint);
    }

    RowSlice proof_row =
        row.slice_from((cached_records ? COMMIT_WIDTH : 0) + proof_idx * SINGLE_WIDTH);
    proof_row.fill_zero(0, SINGLE_WIDTH);

    if (proof_idx >= num_proofs || row_idx >= num_records_per_proof) {
        return;
    }

    uint32_t air_idx = air_ids_per_record[row_idx];
    if (air_idx >= num_airs) {
        return;
    }
    uint32_t local_idx = row_idx - record_bounds[air_idx];

    uint32_t log_height = log_heights[proof_idx * num_airs + air_idx];
    uint32_t sort_idx = sort_idx_by_air_idx[proof_idx * num_airs + air_idx];

    uint32_t expr_bounds_base = proof_idx * (num_airs + 1) + air_idx;
    uint32_t expr_start = expr_eval_bounds_1[expr_bounds_base];
    uint32_t expr_end = expr_eval_bounds_1[expr_bounds_base + 1];
    if (expr_start >= expr_end) {
        return;
    }
    const FpExt *expr_per_air = expr_evals + expr_start;

    uint32_t sum_start = sumcheck_bounds[proof_idx];
    uint32_t sum_end = sumcheck_bounds[proof_idx + 1];
    if (sum_end <= sum_start) {
        return;
    }
    FpExt rs0 = sumcheck_rnds[sum_start];
    const FpExt *rs_rest = sumcheck_rnds + sum_start + 1;
    uint32_t rs_rest_len = sum_end - sum_start > 0 ? sum_end - sum_start - 1 : 0;

    uint32_t nodes_start = constraint_nodes_bounds[air_idx];
    uint32_t nodes_len = constraint_nodes_bounds[air_idx + 1] - nodes_start;
    const FlatSymbolicConstraintNode *nodes_per_air = constraint_nodes + nodes_start;

    uint32_t interactions_start = interactions_bounds[air_idx];
    uint32_t interactions_end = interactions_bounds[air_idx + 1];
    const FlatInteraction *interactions_per_air = interactions + interactions_start;

    uint32_t unused_start = unused_variables_bounds[air_idx];
    uint32_t unused_len = unused_variables_bounds[air_idx + 1] - unused_start;
    const FlatSymbolicVariable *unused_variables_per_air = unused_variables + unused_start;

    Fp sort_idx_fp = Fp(static_cast<uint32_t>(sort_idx));
    uint32_t n_abs = log_height < l_skip ? l_skip - log_height : log_height - l_skip;
    Fp n_abs_fp = Fp(static_cast<uint32_t>(n_abs));
    Fp is_n_neg_fp = log_height < l_skip ? Fp::one() : Fp::zero();

    COL_WRITE_VALUE(proof_row, SymbolicExpressionCols, is_present, Fp::one());
    COL_WRITE_VALUE(proof_row, SymbolicExpressionCols, sort_idx, sort_idx_fp);
    COL_WRITE_VALUE(proof_row, SymbolicExpressionCols, n_abs, n_abs_fp);
    COL_WRITE_VALUE(proof_row, SymbolicExpressionCols, is_n_neg, is_n_neg_fp);

    Encoder encoder(NodeKind::NODE_KIND_COUNT, 2, true);

    if (local_idx < nodes_len) {
        const FlatSymbolicConstraintNode node = nodes_per_air[local_idx];
        switch (node.kind) {
        case NODE_KIND_VAR_PREPROCESSED:
        case NODE_KIND_VAR_MAIN:
        case NODE_KIND_VAR_PUBLIC_VALUE: {
            write_arg_first(proof_row, expr_per_air[local_idx]);
            break;
        }
        case NODE_KIND_IS_FIRST_ROW: {
            uint32_t clamped = log_height < l_skip ? log_height : l_skip;
            uint32_t exponent = clamped <= l_skip ? l_skip - clamped : 0;
            FpExt r_pow = pow_two_power(rs0, exponent);
            FpExt eq_uni = eval_eq_uni_at_one_fp_ext(clamped, r_pow);
            uint32_t mle_idx = log_height > l_skip ? log_height - l_skip : 0;
            if (mle_idx > rs_rest_len) {
                mle_idx = rs_rest_len;
            }
            FpExt eq_mle = product_first_mle(rs_rest, mle_idx);
            write_arg_first(proof_row, eq_uni);
            write_arg_second(proof_row, eq_mle);
            break;
        }
        case NODE_KIND_IS_LAST_ROW:
        case NODE_KIND_IS_TRANSITION: {
            uint32_t clamped = log_height < l_skip ? log_height : l_skip;
            uint32_t exponent = clamped <= l_skip ? l_skip - clamped : 0;
            FpExt r_pow = pow_two_power(rs0, exponent);
            Fp generator = two_adic_generator_at(clamped);
            FpExt eq_uni = eval_eq_uni_at_one_fp_ext(clamped, r_pow * FpExt(generator));
            uint32_t mle_idx = log_height > l_skip ? log_height - l_skip : 0;
            if (mle_idx > rs_rest_len) {
                mle_idx = rs_rest_len;
            }
            FpExt eq_mle = product_last_mle(rs_rest, mle_idx);
            write_arg_first(proof_row, eq_uni);
            write_arg_second(proof_row, eq_mle);
            break;
        }
        case NODE_KIND_ADD:
        case NODE_KIND_SUB:
        case NODE_KIND_MUL: {
            write_arg_first(proof_row, expr_per_air[node.data0]);
            write_arg_second(proof_row, expr_per_air[node.data1]);
            break;
        }
        case NODE_KIND_NEG: {
            write_arg_first(proof_row, expr_per_air[node.data0]);
            break;
        }
        default:
            break;
        }
        if (cached_records && proof_idx == 0) {
            write_dag_commit_flags(row, encoder, node.kind);
        }
        return;
    }

    local_idx -= nodes_len;
    uint32_t interaction_idx = 0;
    while (interactions_start + interaction_idx < interactions_end) {
        const FlatInteraction interaction = interactions_per_air[interaction_idx];
        uint32_t block = 2 + interaction.message_len;
        if (local_idx < block) {
            if (local_idx == 0) {
                write_arg_first(proof_row, expr_per_air[interaction.count]);
            } else if (local_idx == interaction.message_len + 1) {
                write_arg_first(proof_row, FpExt(Fp(interaction.bus_index + 1)));
            } else {
                uint32_t msg_offset = local_idx - 1;
                uint32_t node_idx = interaction_messages[interaction.message_start + msg_offset];
                write_arg_first(proof_row, expr_per_air[node_idx]);
            }
            if (cached_records && proof_idx == 0) {
                write_dag_commit_flags(
                    row,
                    encoder,
                    local_idx == 0
                        ? NODE_KIND_INTERACTION_MULT
                        : (local_idx == interaction.message_len + 1
                               ? NODE_KIND_INTERACTION_BUS_INDEX
                               : NODE_KIND_INTERACTION_MSG_COMP)
                );
            }
            return;
        }
        local_idx -= block;
        ++interaction_idx;
    }

    if (local_idx < unused_len) {
        uint32_t eval_idx = nodes_len + local_idx;
        if (expr_start + eval_idx < expr_end) {
            write_arg_first(proof_row, expr_per_air[eval_idx]);
            if (cached_records && proof_idx == 0) {
                FlatSymbolicVariable unused = unused_variables_per_air[local_idx];
                write_dag_commit_flags(row, encoder, unused.entry_kind);
            }
        }
    }
}

extern "C" int _sym_expr_common_tracegen(
    Fp *d_trace,
    size_t height,
    size_t l_skip,
    const size_t *d_log_heights,
    const size_t *d_sort_idx_by_air_idx,
    size_t num_airs,
    size_t num_proofs,
    size_t max_num_proofs,
    const FpExt *d_expr_evals,
    const size_t *d_ee_bounds_0,
    const size_t *d_ee_bounds_1,
    const FlatSymbolicConstraintNode *d_constraint_nodes,
    const size_t *d_constraint_nodes_bounds,
    const FlatInteraction *d_interactions,
    const size_t *d_interactions_bounds,
    const size_t *d_interaction_messages,
    const FlatSymbolicVariable *d_unused_variables,
    const size_t *d_unused_variables_bounds,
    const uint32_t *d_record_bounds,
    const uint32_t *d_air_ids_per_record,
    size_t num_records_per_proof,
    const FpExt *d_sumcheck_rnds,
    const size_t *d_sumcheck_bounds,
    const CachedRecord *d_cached_records
) {
    // Unused because expr_evals[i].len() is always the number of all airs,
    // plus 1 for unused variables.
    // Will probably need it when we don't push anything for empty traces there
    (void)d_ee_bounds_0;

    assert((height & (height - 1)) == 0);
    size_t total_slots = height * max_num_proofs;
    auto [grid, block] = kernel_launch_params(total_slots, 512);
    symbolic_expression_tracegen<<<grid, block>>>(
        d_trace,
        height,
        l_skip,
        d_log_heights,
        d_sort_idx_by_air_idx,
        num_airs,
        num_proofs,
        max_num_proofs,
        d_expr_evals,
        d_ee_bounds_0,
        d_ee_bounds_1,
        d_constraint_nodes,
        d_constraint_nodes_bounds,
        d_interactions,
        d_interactions_bounds,
        d_interaction_messages,
        d_unused_variables,
        d_unused_variables_bounds,
        d_record_bounds,
        d_air_ids_per_record,
        num_records_per_proof,
        d_sumcheck_rnds,
        d_sumcheck_bounds,
        d_cached_records
    );
    return CHECK_KERNEL();
}
