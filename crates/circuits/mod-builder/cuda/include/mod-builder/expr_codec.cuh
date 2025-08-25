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

__device__ BigIntGpu evaluate_bigint(
    const ExprOp *expr_ops,
    uint32_t root_idx,
    const ExprMeta *expr_meta,
    const uint32_t *inputs,
    const uint32_t *vars,
    const bool *flags,
    uint32_t n,
    uint32_t limb_bits
) {
    DecodedExpr node = decode_expr_op(expr_ops[root_idx]);

    switch (node.kind) {
    case EXPR_INPUT: {
        uint32_t idx = node.data0;
        return BigIntGpu(inputs + idx * n, n, limb_bits);
    }
    case EXPR_VAR: {
        uint32_t idx = node.data0;
        return BigIntGpu(vars + idx * n, n, limb_bits);
    }
    case EXPR_CONST: {
        uint32_t const_idx = node.data0;
        const uint32_t *const_limbs = expr_meta->constants;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < const_idx; i++) {
            offset += expr_meta->const_limb_counts[i];
        }
        return BigIntGpu(const_limbs + offset, expr_meta->const_limb_counts[const_idx], limb_bits);
    }
    case EXPR_ADD: {
        return evaluate_bigint(expr_ops, node.data0, expr_meta, inputs, vars, flags, n, limb_bits) +
               evaluate_bigint(expr_ops, node.data1, expr_meta, inputs, vars, flags, n, limb_bits);
    }
    case EXPR_SUB: {
        return evaluate_bigint(expr_ops, node.data0, expr_meta, inputs, vars, flags, n, limb_bits) -
               evaluate_bigint(expr_ops, node.data1, expr_meta, inputs, vars, flags, n, limb_bits);
    }
    case EXPR_MUL: {
        return evaluate_bigint(expr_ops, node.data0, expr_meta, inputs, vars, flags, n, limb_bits) *
               evaluate_bigint(expr_ops, node.data1, expr_meta, inputs, vars, flags, n, limb_bits);
    }
    case EXPR_INT_ADD: {
        return evaluate_bigint(expr_ops, node.data0, expr_meta, inputs, vars, flags, n, limb_bits) +
               BigIntGpu((int32_t)node.data1, limb_bits);
    }
    case EXPR_INT_MUL: {
        return evaluate_bigint(expr_ops, node.data0, expr_meta, inputs, vars, flags, n, limb_bits) *
               BigIntGpu((int32_t)node.data1, limb_bits);
    }
    case EXPR_SELECT: {
        if (flags[node.data0]) {
            return evaluate_bigint(
                expr_ops, node.data1, expr_meta, inputs, vars, flags, n, limb_bits
            );
        } else {
            return evaluate_bigint(
                expr_ops, node.data2, expr_meta, inputs, vars, flags, n, limb_bits
            );
        }
    }
    default: {
        return BigIntGpu(limb_bits);
    }
    }
}

__device__ BigUintGpu compute_biguint(
    const ExprOp *expr_ops,
    uint32_t expr_idx,
    const ExprMeta *meta,
    const uint32_t *inputs,
    const uint32_t *vars,
    const bool *flags,
    uint32_t num_limbs,
    uint32_t limb_bits,
    BigUintGpu &prime
) {
    DecodedExpr e = decode_expr_op(expr_ops[expr_idx]);

    switch (e.kind) {
    case EXPR_INPUT: {
        const uint32_t *in_limbs = inputs + e.data0 * num_limbs;
        return BigUintGpu(in_limbs, num_limbs, limb_bits).mod_reduce(prime, meta->barrett_mu);
    }
    case EXPR_VAR: {
        const uint32_t *var_limbs = vars + e.data0 * num_limbs;
        return BigUintGpu(var_limbs, num_limbs, limb_bits);
    }
    case EXPR_CONST: {
        uint32_t idx = e.data0;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < idx; i++) {
            offset += meta->const_limb_counts[i];
        }
        return BigUintGpu(meta->constants + offset, meta->const_limb_counts[idx], limb_bits);
    }
    case EXPR_ADD: {
        BigUintGpu sum = compute_biguint(
            expr_ops, e.data0, meta, inputs, vars, flags, num_limbs, limb_bits, prime
        );
        sum += compute_biguint(
            expr_ops, e.data1, meta, inputs, vars, flags, num_limbs, limb_bits, prime
        );
        return sum.mod_reduce(prime, meta->barrett_mu);
    }
    case EXPR_SUB: {
        return compute_biguint(
                   expr_ops, e.data0, meta, inputs, vars, flags, num_limbs, limb_bits, prime
        )
            .mod_sub(
                compute_biguint(
                    expr_ops, e.data1, meta, inputs, vars, flags, num_limbs, limb_bits, prime
                ),
                prime
            );
    }
    case EXPR_MUL: {
        return (compute_biguint(
                    expr_ops, e.data0, meta, inputs, vars, flags, num_limbs, limb_bits, prime
                ) *
                compute_biguint(
                    expr_ops, e.data1, meta, inputs, vars, flags, num_limbs, limb_bits, prime
                ))
            .mod_reduce(prime, meta->barrett_mu);
    }
    case EXPR_DIV: {
        return compute_biguint(
                   expr_ops, e.data0, meta, inputs, vars, flags, num_limbs, limb_bits, prime
        )
            .mod_div(
                compute_biguint(
                    expr_ops, e.data1, meta, inputs, vars, flags, num_limbs, limb_bits, prime
                ),
                prime,
                meta->barrett_mu
            );
    }
    case EXPR_INT_ADD: {
        BigIntGpu a = BigIntGpu(
            compute_biguint(
                expr_ops, e.data0, meta, inputs, vars, flags, num_limbs, limb_bits, prime
            ),
            false
        );
        BigIntGpu sum = a + BigIntGpu((int32_t)e.data1, limb_bits);
        return sum.mag.mod_reduce(prime, meta->barrett_mu);
    }
    case EXPR_INT_MUL: {
        BigIntGpu a = BigIntGpu(
            compute_biguint(
                expr_ops, e.data0, meta, inputs, vars, flags, num_limbs, limb_bits, prime
            ),
            false
        );
        BigIntGpu prod = a * BigIntGpu((int32_t)e.data1, limb_bits);
        return prod.mag.mod_reduce(prime, meta->barrett_mu);
    }
    case EXPR_SELECT: {
        bool f = flags[e.data0];
        uint32_t idx = f ? e.data1 : e.data2;
        return compute_biguint(
            expr_ops, idx, meta, inputs, vars, flags, num_limbs, limb_bits, prime
        );
    }
    default: {
        return BigUintGpu(limb_bits);
    }
    }
}

__device__ void compute(
    uint32_t *result,
    const ExprOp *expr_ops,
    uint32_t expr_idx,
    const ExprMeta *meta,
    const uint32_t *inputs,
    const uint32_t *vars,
    const bool *flags,
    uint32_t num_limbs,
    uint32_t limb_bits,
    BigUintGpu &prime
) {
    BigUintGpu result_big =
        compute_biguint(expr_ops, expr_idx, meta, inputs, vars, flags, num_limbs, limb_bits, prime);

    for (uint32_t i = 0; i < num_limbs; i++) {
        result[i] = (i < result_big.num_limbs) ? result_big.limbs[i] : 0;
    }
}

// Raw evaluator: walks op stream, performs no modular reduction
__device__ OverflowInt evaluate_overflow_int(
    const ExprOp *expr_ops,
    uint32_t op_idx,
    const ExprMeta *meta,
    const uint32_t *inputs,
    const uint32_t *vars,
    const bool *flags,
    uint32_t num_limbs,
    uint32_t limb_bits
) {
    DecodedExpr d = decode_expr_op(expr_ops[op_idx]);

    switch (d.kind) {
    case EXPR_INPUT: {
        const uint32_t *in_limbs = inputs + d.data0 * num_limbs;
        return OverflowInt(in_limbs, num_limbs, limb_bits);
        break;
    }
    case EXPR_VAR: {
        const uint32_t *var_limbs = vars + d.data0 * num_limbs;
        return OverflowInt(var_limbs, num_limbs, limb_bits);
        break;
    }
    case EXPR_CONST: {
        uint32_t idx = d.data0;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < idx; i++) {
            offset += meta->const_limb_counts[i];
        }
        return OverflowInt(meta->constants + offset, meta->const_limb_counts[idx], limb_bits);
        break;
    }
    case EXPR_ADD: {
        OverflowInt a = evaluate_overflow_int(
            expr_ops, d.data0, meta, inputs, vars, flags, num_limbs, limb_bits
        );
        a += evaluate_overflow_int(
            expr_ops, d.data1, meta, inputs, vars, flags, num_limbs, limb_bits
        );
        return a;
    }
    case EXPR_SUB: {
        OverflowInt a = evaluate_overflow_int(
            expr_ops, d.data0, meta, inputs, vars, flags, num_limbs, limb_bits
        );
        a -= evaluate_overflow_int(
            expr_ops, d.data1, meta, inputs, vars, flags, num_limbs, limb_bits
        );
        return a;
    }
    case EXPR_MUL: {
        OverflowInt a = evaluate_overflow_int(
            expr_ops, d.data0, meta, inputs, vars, flags, num_limbs, limb_bits
        );
        a *= evaluate_overflow_int(
            expr_ops, d.data1, meta, inputs, vars, flags, num_limbs, limb_bits
        );
        return a;
    }
    case EXPR_SELECT: {
        OverflowInt a = evaluate_overflow_int(
            expr_ops, d.data1, meta, inputs, vars, flags, num_limbs, limb_bits
        );
        OverflowInt b = evaluate_overflow_int(
            expr_ops, d.data2, meta, inputs, vars, flags, num_limbs, limb_bits
        );
        if (flags[d.data0]) {
            a.limb_max_abs = max(a.limb_max_abs, b.limb_max_abs);
            a.max_overflow_bits = max(a.max_overflow_bits, b.max_overflow_bits);
            return a;
        } else {
            b.limb_max_abs = max(a.limb_max_abs, b.limb_max_abs);
            b.max_overflow_bits = max(a.max_overflow_bits, b.max_overflow_bits);
            return b;
        }
    }
    case EXPR_INT_ADD: {
        OverflowInt a = evaluate_overflow_int(
            expr_ops, d.data0, meta, inputs, vars, flags, num_limbs, limb_bits
        );
        int32_t scalar = (int32_t)d.data1;
        a += scalar;
        return a;
    }
    case EXPR_INT_MUL: {
        OverflowInt a = evaluate_overflow_int(
            expr_ops, d.data0, meta, inputs, vars, flags, num_limbs, limb_bits
        );
        int32_t scalar = (int32_t)d.data1;
        a *= scalar;
        return a;
    }
    default:
        return OverflowInt(limb_bits);
    }
}

__device__ void evaluate_overflow(
    uint32_t *result,
    const ExprOp *expr_ops,
    uint32_t op_idx,
    const ExprMeta *meta,
    const uint32_t *inputs,
    const uint32_t *vars,
    const bool *flags,
    uint32_t num_limbs,
    uint32_t limb_bits,
    uint32_t &max_limb_abs,
    uint32_t &max_overflow_bits,
    uint32_t *temp_storage
) {
    OverflowInt result_overflow =
        evaluate_overflow_int(expr_ops, op_idx, meta, inputs, vars, flags, num_limbs, limb_bits);

    for (uint32_t i = 0; i < 2 * num_limbs; i++) {
        result[i] = (i < result_overflow.num_limbs) ? result_overflow.limbs[i] : 0;
    }

    max_limb_abs = result_overflow.limb_max_abs;
    max_overflow_bits = result_overflow.max_overflow_bits;
}
