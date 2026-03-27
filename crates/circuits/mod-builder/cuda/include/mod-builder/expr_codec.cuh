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

__device__ __forceinline__ void push_compute_child_if_needed(
    uint32_t child_idx,
    uint32_t pool_size,
    uint8_t *live_nodes,
    uint32_t *node_stack,
    uint32_t &stack_len
) {
    assert(child_idx < pool_size);
    if (live_nodes[child_idx] != 0) {
        return;
    }

    assert(stack_len < pool_size);
    live_nodes[child_idx] = 1;
    node_stack[stack_len] = child_idx;
    stack_len++;
}

__device__ __forceinline__ BigUintGpu &compute_scratch_slot(
    BigUintGpu *scratch,
    uint32_t scratch_stride,
    uint32_t slot
) {
    return scratch[(size_t)slot * scratch_stride];
}

__device__ __forceinline__ const BigUintGpu &compute_scratch_slot(
    const BigUintGpu *scratch,
    uint32_t scratch_stride,
    uint32_t slot
) {
    return scratch[(size_t)slot * scratch_stride];
}

__device__ __forceinline__ BigIntGpu &constraint_bigint_slot(
    BigIntGpu *scratch,
    const uint32_t *scratch_slots,
    uint32_t idx
) {
    return scratch[scratch_slots[idx]];
}

__device__ __forceinline__ OverflowInt &constraint_overflow_slot(
    OverflowInt *scratch,
    const uint32_t *scratch_slots,
    uint32_t idx
) {
    return scratch[scratch_slots[idx]];
}

// Mark the live compute subgraph for the current row, preserving lazy SELECT semantics.
__device__ __noinline__ void mark_compute_live_nodes(
    const ExprOp *expr_ops,
    const bool *flags,
    const uint32_t *root_indices,
    uint32_t num_roots,
    uint8_t *live_nodes,
    uint32_t *node_stack,
    uint32_t pool_size
) {
    if (pool_size == 0) {
        return;
    }

    uint32_t stack_len = 0;
    for (uint32_t i = 0; i < num_roots; i++) {
        uint32_t root_idx = root_indices[i];
        assert(root_idx < pool_size);
        if (live_nodes[root_idx] == 0) {
            live_nodes[root_idx] = 1;
            node_stack[stack_len] = root_idx;
            stack_len++;
        }
    }

    while (stack_len > 0) {
        uint32_t idx = node_stack[--stack_len];

        DecodedExpr node = decode_expr_op(expr_ops[idx]);

        switch (node.kind) {
        case EXPR_ADD:
        case EXPR_SUB:
        case EXPR_MUL:
        case EXPR_DIV:
            push_compute_child_if_needed(node.data1, pool_size, live_nodes, node_stack, stack_len);
            push_compute_child_if_needed(node.data0, pool_size, live_nodes, node_stack, stack_len);
            break;
        case EXPR_INT_ADD:
        case EXPR_INT_MUL:
            push_compute_child_if_needed(node.data0, pool_size, live_nodes, node_stack, stack_len);
            break;
        case EXPR_SELECT: {
            uint32_t child = flags[node.data0] ? node.data1 : node.data2;
            push_compute_child_if_needed(child, pool_size, live_nodes, node_stack, stack_len);
            break;
        }
        default:
            break;
        }
    }
}

__device__ __noinline__ void evaluate_compute_pool_flat(
    const ExprOp *expr_ops,
    uint32_t pool_size,
    const ExprMeta *meta,
    const uint32_t *inputs,
    uint32_t *vars,
    const bool *flags,
    uint32_t num_limbs,
    uint32_t limb_bits,
    const BigUintGpu &prime,
    BigUintGpu *scratch,
    uint32_t scratch_stride,
    const uint32_t *scratch_slots,
    const uint8_t *live_nodes,
    const uint32_t *root_indices,
    uint32_t num_roots
) {
    for (uint32_t idx = 0; idx < pool_size; idx++) {
        if (live_nodes[idx] == 0) {
            continue;
        }

        DecodedExpr node = decode_expr_op(expr_ops[idx]);
        uint32_t slot = scratch_slots[idx];
        switch (node.kind) {
        case EXPR_INPUT:
            compute_scratch_slot(scratch, scratch_stride, slot) =
                BigUintGpu(inputs + node.data0 * num_limbs, num_limbs, limb_bits)
                    .mod_reduce(prime, meta->barrett_mu);
            break;
        case EXPR_VAR:
            assert(node.data0 < num_roots);
            compute_scratch_slot(scratch, scratch_stride, slot) = compute_scratch_slot(
                scratch, scratch_stride, scratch_slots[root_indices[node.data0]]
            );
            break;
        case EXPR_CONST: {
            uint32_t offset = get_const_offset(meta, node.data0);
            compute_scratch_slot(scratch, scratch_stride, slot) = BigUintGpu(
                meta->constants + offset, meta->const_limb_counts[node.data0], limb_bits
            );
            break;
        }
        case EXPR_ADD:
            compute_scratch_slot(scratch, scratch_stride, slot) =
                (compute_scratch_slot(scratch, scratch_stride, scratch_slots[node.data0]) +
                 compute_scratch_slot(scratch, scratch_stride, scratch_slots[node.data1]))
                    .mod_reduce(prime, meta->barrett_mu);
            break;
        case EXPR_SUB:
            compute_scratch_slot(scratch, scratch_stride, slot) =
                compute_scratch_slot(scratch, scratch_stride, scratch_slots[node.data0])
                    .mod_sub(
                        compute_scratch_slot(scratch, scratch_stride, scratch_slots[node.data1]),
                        prime
                    );
            break;
        case EXPR_MUL:
            compute_scratch_slot(scratch, scratch_stride, slot) =
                (compute_scratch_slot(scratch, scratch_stride, scratch_slots[node.data0]) *
                 compute_scratch_slot(scratch, scratch_stride, scratch_slots[node.data1]))
                    .mod_reduce(prime, meta->barrett_mu);
            break;
        case EXPR_DIV:
            compute_scratch_slot(scratch, scratch_stride, slot) =
                compute_scratch_slot(scratch, scratch_stride, scratch_slots[node.data0])
                    .mod_div(
                        compute_scratch_slot(scratch, scratch_stride, scratch_slots[node.data1]),
                        prime,
                        meta->barrett_mu
                    );
            break;
        case EXPR_INT_ADD: {
            BigIntGpu a(
                compute_scratch_slot(scratch, scratch_stride, scratch_slots[node.data0]), false
            );
            compute_scratch_slot(scratch, scratch_stride, slot) =
                (a + BigIntGpu((int32_t)node.data1, limb_bits))
                    .mag.mod_reduce(prime, meta->barrett_mu);
            break;
        }
        case EXPR_INT_MUL: {
            BigIntGpu a(
                compute_scratch_slot(scratch, scratch_stride, scratch_slots[node.data0]), false
            );
            compute_scratch_slot(scratch, scratch_stride, slot) =
                (a * BigIntGpu((int32_t)node.data1, limb_bits))
                    .mag.mod_reduce(prime, meta->barrett_mu);
            break;
        }
        case EXPR_SELECT:
            compute_scratch_slot(scratch, scratch_stride, slot) =
                flags[node.data0]
                    ? compute_scratch_slot(scratch, scratch_stride, scratch_slots[node.data1])
                    : compute_scratch_slot(scratch, scratch_stride, scratch_slots[node.data2]);
            break;
        default:
            compute_scratch_slot(scratch, scratch_stride, slot) = BigUintGpu(limb_bits);
            break;
        }
    }

    for (uint32_t var = 0; var < num_roots; var++) {
        uint32_t root_slot = scratch_slots[root_indices[var]];
        for (uint32_t limb = 0; limb < num_limbs; limb++) {
            vars[var * num_limbs + limb] =
                (limb < compute_scratch_slot(scratch, scratch_stride, root_slot).num_limbs)
                    ? compute_scratch_slot(scratch, scratch_stride, root_slot).limbs[limb]
                    : 0;
        }
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
    const uint32_t *scratch_slots,
    BigIntGpu *bigint_scratch,
    OverflowInt *overflow_scratch
) {
    for (uint32_t i = 0; i < pool_size; i++) {
        DecodedExpr node = decode_expr_op(expr_ops[i]);

        switch (node.kind) {
        case EXPR_INPUT:
            constraint_bigint_slot(bigint_scratch, scratch_slots, i) =
                BigIntGpu(inputs + node.data0 * num_limbs, num_limbs, limb_bits);
            constraint_overflow_slot(overflow_scratch, scratch_slots, i) =
                OverflowInt(inputs + node.data0 * num_limbs, num_limbs, limb_bits);
            break;
        case EXPR_VAR:
            constraint_bigint_slot(bigint_scratch, scratch_slots, i) =
                BigIntGpu(vars + node.data0 * num_limbs, num_limbs, limb_bits);
            constraint_overflow_slot(overflow_scratch, scratch_slots, i) =
                OverflowInt(vars + node.data0 * num_limbs, num_limbs, limb_bits);
            break;
        case EXPR_CONST: {
            uint32_t offset = get_const_offset(meta, node.data0);
            constraint_bigint_slot(bigint_scratch, scratch_slots, i) =
                BigIntGpu(meta->constants + offset, meta->const_limb_counts[node.data0], limb_bits);
            constraint_overflow_slot(overflow_scratch, scratch_slots, i) = OverflowInt(
                meta->constants + offset, meta->const_limb_counts[node.data0], limb_bits
            );
            break;
        }
        case EXPR_ADD:
            constraint_bigint_slot(bigint_scratch, scratch_slots, i) =
                constraint_bigint_slot(bigint_scratch, scratch_slots, node.data0) +
                constraint_bigint_slot(bigint_scratch, scratch_slots, node.data1);
            constraint_overflow_slot(overflow_scratch, scratch_slots, i) =
                constraint_overflow_slot(overflow_scratch, scratch_slots, node.data0) +
                constraint_overflow_slot(overflow_scratch, scratch_slots, node.data1);
            break;
        case EXPR_SUB:
            constraint_bigint_slot(bigint_scratch, scratch_slots, i) =
                constraint_bigint_slot(bigint_scratch, scratch_slots, node.data0) -
                constraint_bigint_slot(bigint_scratch, scratch_slots, node.data1);
            constraint_overflow_slot(overflow_scratch, scratch_slots, i) =
                constraint_overflow_slot(overflow_scratch, scratch_slots, node.data0) -
                constraint_overflow_slot(overflow_scratch, scratch_slots, node.data1);
            break;
        case EXPR_MUL:
            constraint_bigint_slot(bigint_scratch, scratch_slots, i) =
                constraint_bigint_slot(bigint_scratch, scratch_slots, node.data0) *
                constraint_bigint_slot(bigint_scratch, scratch_slots, node.data1);
            constraint_overflow_slot(overflow_scratch, scratch_slots, i) =
                constraint_overflow_slot(overflow_scratch, scratch_slots, node.data0) *
                constraint_overflow_slot(overflow_scratch, scratch_slots, node.data1);
            break;
        case EXPR_INT_ADD:
            constraint_bigint_slot(bigint_scratch, scratch_slots, i) =
                constraint_bigint_slot(bigint_scratch, scratch_slots, node.data0) +
                BigIntGpu((int32_t)node.data1, limb_bits);
            constraint_overflow_slot(overflow_scratch, scratch_slots, i) =
                constraint_overflow_slot(overflow_scratch, scratch_slots, node.data0) +
                (int32_t)node.data1;
            break;
        case EXPR_INT_MUL:
            constraint_bigint_slot(bigint_scratch, scratch_slots, i) =
                constraint_bigint_slot(bigint_scratch, scratch_slots, node.data0) *
                BigIntGpu((int32_t)node.data1, limb_bits);
            constraint_overflow_slot(overflow_scratch, scratch_slots, i) =
                constraint_overflow_slot(overflow_scratch, scratch_slots, node.data0) *
                (int32_t)node.data1;
            break;
        case EXPR_SELECT: {
            bool take_true = flags[node.data0];
            constraint_bigint_slot(bigint_scratch, scratch_slots, i) =
                take_true ? constraint_bigint_slot(bigint_scratch, scratch_slots, node.data1)
                          : constraint_bigint_slot(bigint_scratch, scratch_slots, node.data2);

            OverflowInt true_expr =
                constraint_overflow_slot(overflow_scratch, scratch_slots, node.data1);
            OverflowInt false_expr =
                constraint_overflow_slot(overflow_scratch, scratch_slots, node.data2);
            OverflowInt selected = take_true ? true_expr : false_expr;
            selected.limb_max_abs = max(true_expr.limb_max_abs, false_expr.limb_max_abs);
            selected.max_overflow_bits =
                max(true_expr.max_overflow_bits, false_expr.max_overflow_bits);
            constraint_overflow_slot(overflow_scratch, scratch_slots, i) = selected;
            break;
        }
        default:
            // Constraint expressions should not contain division.
            constraint_bigint_slot(bigint_scratch, scratch_slots, i) = BigIntGpu(limb_bits);
            constraint_overflow_slot(overflow_scratch, scratch_slots, i) = OverflowInt(limb_bits);
            break;
        }
    }
}
