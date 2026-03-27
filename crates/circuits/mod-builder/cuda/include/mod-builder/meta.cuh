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
    uint32_t adapter_size;
    uint32_t adapter_width;
    uint32_t core_width;

    const uint32_t *local_opcode_idx;
    const uint32_t *opcode_flag_idx;

    uint32_t num_local_opcodes;
    const ExprOp *compute_expr_ops;
    const uint32_t *compute_root_indices;
    const uint32_t *compute_scratch_slots;
    uint32_t compute_pool_size;
    uint32_t compute_scratch_slot_count;
    const ExprOp *constraint_expr_ops;
    const uint32_t *constraint_root_indices;
    const uint32_t *constraint_scratch_slots;
    uint32_t constraint_pool_size;
    uint32_t constraint_scratch_slot_count;

    ExprMeta expr_meta;
} FieldExprMeta;
