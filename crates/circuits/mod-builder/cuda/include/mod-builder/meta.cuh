#pragma once

#include "symbolic_expr.cuh"
#include <stdint.h>

// Flat 128-bit encoded expression ops
typedef unsigned __int128 ExprOp;

typedef struct {
    uint32_t num_inputs;
    uint32_t num_u32_flags;
    uint32_t num_limbs;
    uint32_t limb_bits;
    uint32_t adapter_blocks;
    uint32_t adapter_width;
    uint32_t core_width;
    uint32_t trace_width;

    const uint32_t *local_opcode_idx;
    const uint32_t *opcode_flag_idx;
    const uint32_t *output_indices;

    uint32_t num_local_opcodes;
    uint32_t num_output_indices;

    uint32_t record_stride;
    uint32_t input_limbs_offset;

    const uint32_t *q_limb_counts;
    const uint32_t *carry_limb_counts;
    const ExprOp *compute_expr_ops;
    const uint32_t *compute_root_indices;
    const ExprOp *constraint_expr_ops;
    const uint32_t *constraint_root_indices;
    uint32_t max_q_count;

    ExprMeta expr_meta;

    uint32_t max_ast_depth;
} FieldExprMeta;