#pragma once

#include <stdint.h>

typedef enum {
    EXPR_INPUT = 0,
    EXPR_VAR = 1,
    EXPR_CONST = 2,
    EXPR_ADD = 3,
    EXPR_SUB = 4,
    EXPR_MUL = 5,
    EXPR_DIV = 6,
    EXPR_INT_ADD = 7,
    EXPR_INT_MUL = 8,
    EXPR_SELECT = 9
} ExprType;

typedef struct {
    ExprType type;
    uint32_t data[3];
    // Input/Var/Const: data[0] = index
    // binary ops: data[0] = left_child_idx, data[1] = right_child_idx
    // IntAdd/IntMul: data[0] = child_idx, data[1] = integer_value (as uint32)
    // select: data[0] = flag_idx, data[1] = true_child_idx, data[2] = false_child_idx
} ExprNode;

typedef struct {
    const uint32_t *constants;
    const uint32_t *const_limb_counts;
    const uint32_t *q_limb_counts;
    const uint32_t *carry_limb_counts;

    uint32_t num_vars;
    uint32_t num_constants;
    uint32_t expr_pool_size;

    const uint32_t *prime_limbs;
    uint32_t prime_limb_count;
    uint32_t limb_bits;

    const uint8_t *barrett_mu;
} ExprMeta;