#pragma once

#include "bigint_ops.cuh"
#include "meta.cuh"
#include "overflow_ops.cuh"

// 128-bit encoded expression node: 8-bit kind + three 32-bit data fields
struct DecodedExpr {
    uint32_t kind;
    uint32_t data0;
    uint32_t data1;
    uint32_t data2;
};

__device__ __forceinline__ DecodedExpr decode_expr_op(ExprOp raw) {
    DecodedExpr d;
    d.kind = ((raw >> 0) & 0xFF);
    d.data0 = ((raw >> 8) & 0xFFFFFFFF);
    d.data1 = ((raw >> 40) & 0xFFFFFFFF);
    d.data2 = ((raw >> 72) & 0xFFFFFFFF);
    return d;
}

__device__ __forceinline__ uint32_t get_const_offset(const ExprMeta *meta, uint32_t const_idx) {
    if (meta->const_limb_offsets != nullptr) {
        return meta->const_limb_offsets[const_idx];
    }

    // Compatibility fallback while transitioning metadata.
    uint32_t offset = 0;
    for (uint32_t i = 0; i < const_idx; i++) {
        offset += meta->const_limb_counts[i];
    }
    return offset;
}

__device__ __forceinline__ void push_child_if_needed(
    uint32_t child_idx,
    uint32_t done_mark,
    uint32_t pending_mark,
    uint32_t pool_size,
    uint32_t *visit_epoch,
    uint32_t *node_stack,
    uint8_t *phase_stack,
    uint32_t &stack_len
) {
    assert(child_idx < pool_size);
    uint32_t state = visit_epoch[child_idx];
    if (state == done_mark || state == pending_mark) {
        return;
    }

    assert(stack_len < pool_size);
    visit_epoch[child_idx] = pending_mark;
    node_stack[stack_len] = child_idx;
    phase_stack[stack_len] = 0;
    stack_len++;
}

// Iterative compute evaluator with lazy SELECT semantics.
__device__ __noinline__ void compute_flat_lazy(
    uint32_t *result,
    const ExprOp *expr_ops,
    uint32_t root_idx,
    const ExprMeta *meta,
    const uint32_t *inputs,
    const uint32_t *vars,
    const bool *flags,
    uint32_t num_limbs,
    uint32_t limb_bits,
    const BigUintGpu &prime,
    BigUintGpu *scratch,
    uint32_t *visit_epoch,
    uint32_t current_epoch,
    uint32_t *node_stack,
    uint8_t *phase_stack,
    uint32_t pool_size
) {
    if (pool_size == 0) {
        for (uint32_t i = 0; i < num_limbs; i++) {
            result[i] = 0;
        }
        return;
    }

    assert(root_idx < pool_size);
    assert((current_epoch & 0x80000000u) == 0);

    uint32_t done_mark = current_epoch;
    uint32_t pending_mark = current_epoch | 0x80000000u;

    uint32_t stack_len = 0;
    visit_epoch[root_idx] = pending_mark;
    node_stack[stack_len] = root_idx;
    phase_stack[stack_len] = 0;
    stack_len++;

    while (stack_len > 0) {
        uint32_t idx = node_stack[stack_len - 1];
        uint8_t phase = phase_stack[stack_len - 1];

        if (visit_epoch[idx] == done_mark) {
            stack_len--;
            continue;
        }

        DecodedExpr node = decode_expr_op(expr_ops[idx]);

        if (phase == 0) {
            phase_stack[stack_len - 1] = 1;

            switch (node.kind) {
            case EXPR_ADD:
            case EXPR_SUB:
            case EXPR_MUL:
            case EXPR_DIV:
                // Push right then left to evaluate left first.
                push_child_if_needed(
                    node.data1,
                    done_mark,
                    pending_mark,
                    pool_size,
                    visit_epoch,
                    node_stack,
                    phase_stack,
                    stack_len
                );
                push_child_if_needed(
                    node.data0,
                    done_mark,
                    pending_mark,
                    pool_size,
                    visit_epoch,
                    node_stack,
                    phase_stack,
                    stack_len
                );
                break;
            case EXPR_INT_ADD:
            case EXPR_INT_MUL:
                push_child_if_needed(
                    node.data0,
                    done_mark,
                    pending_mark,
                    pool_size,
                    visit_epoch,
                    node_stack,
                    phase_stack,
                    stack_len
                );
                break;
            case EXPR_SELECT: {
                uint32_t child = flags[node.data0] ? node.data1 : node.data2;
                push_child_if_needed(
                    child,
                    done_mark,
                    pending_mark,
                    pool_size,
                    visit_epoch,
                    node_stack,
                    phase_stack,
                    stack_len
                );
                break;
            }
            default:
                break;
            }

            continue;
        }

        switch (node.kind) {
        case EXPR_INPUT:
            scratch[idx] =
                BigUintGpu(inputs + node.data0 * num_limbs, num_limbs, limb_bits)
                    .mod_reduce(prime, meta->barrett_mu);
            break;
        case EXPR_VAR:
            scratch[idx] = BigUintGpu(vars + node.data0 * num_limbs, num_limbs, limb_bits);
            break;
        case EXPR_CONST: {
            uint32_t offset = get_const_offset(meta, node.data0);
            scratch[idx] =
                BigUintGpu(meta->constants + offset, meta->const_limb_counts[node.data0], limb_bits);
            break;
        }
        case EXPR_ADD:
            scratch[idx] =
                (scratch[node.data0] + scratch[node.data1]).mod_reduce(prime, meta->barrett_mu);
            break;
        case EXPR_SUB:
            scratch[idx] = scratch[node.data0].mod_sub(scratch[node.data1], prime);
            break;
        case EXPR_MUL:
            scratch[idx] =
                (scratch[node.data0] * scratch[node.data1]).mod_reduce(prime, meta->barrett_mu);
            break;
        case EXPR_DIV:
            scratch[idx] =
                scratch[node.data0].mod_div(scratch[node.data1], prime, meta->barrett_mu);
            break;
        case EXPR_INT_ADD: {
            BigIntGpu a(scratch[node.data0], false);
            scratch[idx] =
                (a + BigIntGpu((int32_t)node.data1, limb_bits)).mag.mod_reduce(prime, meta->barrett_mu);
            break;
        }
        case EXPR_INT_MUL: {
            BigIntGpu a(scratch[node.data0], false);
            scratch[idx] =
                (a * BigIntGpu((int32_t)node.data1, limb_bits)).mag.mod_reduce(prime, meta->barrett_mu);
            break;
        }
        case EXPR_SELECT:
            scratch[idx] = flags[node.data0] ? scratch[node.data1] : scratch[node.data2];
            break;
        default:
            scratch[idx] = BigUintGpu(limb_bits);
            break;
        }

        visit_epoch[idx] = done_mark;
        stack_len--;
    }

    for (uint32_t i = 0; i < num_limbs; i++) {
        result[i] = (i < scratch[root_idx].num_limbs) ? scratch[root_idx].limbs[i] : 0;
    }
}

// Flat constraint evaluator over the entire topologically-sorted pool.
__device__ __noinline__ void evaluate_constraint_pool_flat(
    const ExprOp *expr_ops,
    uint32_t pool_size,
    const ExprMeta *meta,
    const uint32_t *inputs,
    const uint32_t *vars,
    const bool *flags,
    uint32_t num_limbs,
    uint32_t limb_bits,
    BigIntGpu *bigint_scratch,
    OverflowInt *overflow_scratch
) {
    for (uint32_t i = 0; i < pool_size; i++) {
        DecodedExpr node = decode_expr_op(expr_ops[i]);

        switch (node.kind) {
        case EXPR_INPUT:
            bigint_scratch[i] = BigIntGpu(inputs + node.data0 * num_limbs, num_limbs, limb_bits);
            overflow_scratch[i] =
                OverflowInt(inputs + node.data0 * num_limbs, num_limbs, limb_bits);
            break;
        case EXPR_VAR:
            bigint_scratch[i] = BigIntGpu(vars + node.data0 * num_limbs, num_limbs, limb_bits);
            overflow_scratch[i] = OverflowInt(vars + node.data0 * num_limbs, num_limbs, limb_bits);
            break;
        case EXPR_CONST: {
            uint32_t offset = get_const_offset(meta, node.data0);
            bigint_scratch[i] =
                BigIntGpu(meta->constants + offset, meta->const_limb_counts[node.data0], limb_bits);
            overflow_scratch[i] =
                OverflowInt(meta->constants + offset, meta->const_limb_counts[node.data0], limb_bits);
            break;
        }
        case EXPR_ADD:
            bigint_scratch[i] = bigint_scratch[node.data0] + bigint_scratch[node.data1];
            overflow_scratch[i] = overflow_scratch[node.data0] + overflow_scratch[node.data1];
            break;
        case EXPR_SUB:
            bigint_scratch[i] = bigint_scratch[node.data0] - bigint_scratch[node.data1];
            overflow_scratch[i] = overflow_scratch[node.data0] - overflow_scratch[node.data1];
            break;
        case EXPR_MUL:
            bigint_scratch[i] = bigint_scratch[node.data0] * bigint_scratch[node.data1];
            overflow_scratch[i] = overflow_scratch[node.data0] * overflow_scratch[node.data1];
            break;
        case EXPR_INT_ADD:
            bigint_scratch[i] = bigint_scratch[node.data0] + BigIntGpu((int32_t)node.data1, limb_bits);
            overflow_scratch[i] = overflow_scratch[node.data0] + (int32_t)node.data1;
            break;
        case EXPR_INT_MUL:
            bigint_scratch[i] = bigint_scratch[node.data0] * BigIntGpu((int32_t)node.data1, limb_bits);
            overflow_scratch[i] = overflow_scratch[node.data0] * (int32_t)node.data1;
            break;
        case EXPR_SELECT: {
            bool take_true = flags[node.data0];
            bigint_scratch[i] = take_true ? bigint_scratch[node.data1] : bigint_scratch[node.data2];

            OverflowInt true_expr = overflow_scratch[node.data1];
            OverflowInt false_expr = overflow_scratch[node.data2];
            OverflowInt selected = take_true ? true_expr : false_expr;
            selected.limb_max_abs = max(true_expr.limb_max_abs, false_expr.limb_max_abs);
            selected.max_overflow_bits =
                max(true_expr.max_overflow_bits, false_expr.max_overflow_bits);
            overflow_scratch[i] = selected;
            break;
        }
        default:
            // Constraint expressions should not contain division.
            bigint_scratch[i] = BigIntGpu(limb_bits);
            overflow_scratch[i] = OverflowInt(limb_bits);
            break;
        }
    }
}
