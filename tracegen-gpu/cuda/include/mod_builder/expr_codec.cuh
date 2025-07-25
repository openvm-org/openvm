#pragma once
#include "bigint_ops.cuh"
#include "fpext.h"
#include "limb_ops.cuh"
#include "meta.cuh"

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
    ExprOp op = expr_ops[root_idx];
    DecodedExpr node = decode_expr_op(op);
    BigIntGpu result;

    switch (node.kind) {
    case EXPR_INPUT: {
        uint32_t idx = node.data0;
        result = BigIntGpu(inputs + idx * n, n, limb_bits);
        break;
    }
    case EXPR_VAR: {
        uint32_t idx = node.data0;
        result = BigIntGpu(vars + idx * n, n, limb_bits);
        break;
    }
    case EXPR_CONST: {
        uint32_t const_idx = node.data0;
        const uint32_t *const_limbs = expr_meta->constants;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < const_idx; i++) {
            offset += expr_meta->const_limb_counts[i];
        }
        result =
            BigIntGpu(const_limbs + offset, expr_meta->const_limb_counts[const_idx], limb_bits);

        break;
    }
    case EXPR_ADD: {
        uint32_t left_idx = node.data0;
        uint32_t right_idx = node.data1;

        BigIntGpu left =
            evaluate_bigint(expr_ops, left_idx, expr_meta, inputs, vars, flags, n, limb_bits);
        BigIntGpu right =
            evaluate_bigint(expr_ops, right_idx, expr_meta, inputs, vars, flags, n, limb_bits);
        bigint_add(&result, &left, &right);
        break;
    }
    case EXPR_SUB: {
        uint32_t left_idx = node.data0;
        uint32_t right_idx = node.data1;
        BigIntGpu left =
            evaluate_bigint(expr_ops, left_idx, expr_meta, inputs, vars, flags, n, limb_bits);
        BigIntGpu right =
            evaluate_bigint(expr_ops, right_idx, expr_meta, inputs, vars, flags, n, limb_bits);
        bigint_sub(&result, &left, &right);

        break;
    }
    case EXPR_MUL: {
        uint32_t left_idx = node.data0;
        uint32_t right_idx = node.data1;
        BigIntGpu left =
            evaluate_bigint(expr_ops, left_idx, expr_meta, inputs, vars, flags, n, limb_bits);
        BigIntGpu right =
            evaluate_bigint(expr_ops, right_idx, expr_meta, inputs, vars, flags, n, limb_bits);
        bigint_mul(&result, &left, &right);
        break;
    }
    case EXPR_INT_ADD: {
        uint32_t child_idx = node.data0;
        int32_t scalar = (int32_t)node.data1;

        BigIntGpu child =
            evaluate_bigint(expr_ops, child_idx, expr_meta, inputs, vars, flags, n, limb_bits);

        BigIntGpu scalar_bigint(scalar, limb_bits);

        bigint_add(&result, &child, &scalar_bigint);
        break;
    }
    case EXPR_INT_MUL: {
        uint32_t child_idx = node.data0;
        int32_t scalar = (int32_t)node.data1;

        BigIntGpu child =
            evaluate_bigint(expr_ops, child_idx, expr_meta, inputs, vars, flags, n, limb_bits);

        BigIntGpu scalar_bigint(scalar, limb_bits);

        bigint_mul(&result, &child, &scalar_bigint);
        break;
    }
    case EXPR_SELECT: {
        uint32_t flag_idx = node.data0;
        uint32_t true_idx = node.data1;
        uint32_t false_idx = node.data2;

        if (flags[flag_idx]) {
            result =
                evaluate_bigint(expr_ops, true_idx, expr_meta, inputs, vars, flags, n, limb_bits);
        } else {
            result =
                evaluate_bigint(expr_ops, false_idx, expr_meta, inputs, vars, flags, n, limb_bits);
        }
        break;
    }
    default:
        result = BigIntGpu(limb_bits);
        break;
    }

    return result;
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
    uint32_t *temp_buf
) {
    DecodedExpr e = decode_expr_op(expr_ops[expr_idx]);

    switch (e.kind) {
    case EXPR_INPUT: {
        const uint32_t *in_limbs = inputs + e.data0 * num_limbs;
        for (uint32_t i = 0; i < num_limbs; i++) {
            result[i] = in_limbs[i];
        }
        break;
    }
    case EXPR_VAR: {
        const uint32_t *var_limbs = vars + e.data0 * num_limbs;
        for (uint32_t i = 0; i < num_limbs; i++) {
            result[i] = var_limbs[i];
        }
        break;
    }
    case EXPR_CONST: {
        uint32_t idx = e.data0;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < idx; i++) {
            offset += meta->const_limb_counts[i];
        }
        uint32_t count = meta->const_limb_counts[idx];
        for (uint32_t i = 0; i < num_limbs; i++) {
            result[i] = (i < count) ? meta->constants[offset + i] : 0;
        }
        break;
    }
    case EXPR_ADD: {
        uint32_t *a = temp_buf;
        uint32_t *b = temp_buf + num_limbs;
        uint32_t *sum = temp_buf + 2 * num_limbs;
        compute(a, expr_ops, e.data0, meta, inputs, vars, flags, num_limbs, limb_bits, sum);
        compute(b, expr_ops, e.data1, meta, inputs, vars, flags, num_limbs, limb_bits, sum);
        limb_add(sum, a, b, num_limbs, limb_bits);
        limb_mod_reduce(result, sum, meta->prime_limbs, num_limbs, limb_bits, meta->barrett_mu);
        break;
    }
    case EXPR_SUB: {
        uint32_t *a = temp_buf;
        uint32_t *b = temp_buf + num_limbs;
        compute(
            a,
            expr_ops,
            e.data0,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            temp_buf + 2 * num_limbs
        );
        compute(
            b,
            expr_ops,
            e.data1,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            temp_buf + 2 * num_limbs
        );
        limb_mod_sub(result, a, b, meta->prime_limbs, num_limbs, limb_bits);
        break;
    }
    case EXPR_MUL: {
        uint32_t *a = temp_buf;
        uint32_t *b = temp_buf + num_limbs;
        uint32_t *prod = temp_buf + 2 * num_limbs;
        compute(
            a,
            expr_ops,
            e.data0,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            temp_buf + 4 * num_limbs
        );
        compute(
            b,
            expr_ops,
            e.data1,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            temp_buf + 4 * num_limbs
        );
        limb_mul(prod, a, b, num_limbs, limb_bits);
        limb_mod_reduce(result, prod, meta->prime_limbs, num_limbs, limb_bits, meta->barrett_mu);
        break;
    }
    case EXPR_DIV: {
        uint32_t *a = temp_buf;
        uint32_t *b = temp_buf + num_limbs;
        compute(
            a,
            expr_ops,
            e.data0,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            temp_buf + 4 * num_limbs
        );
        compute(
            b,
            expr_ops,
            e.data1,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            temp_buf + 4 * num_limbs
        );
        limb_mod_div(
            result,
            a,
            b,
            meta->prime_limbs,
            num_limbs,
            limb_bits,
            meta->barrett_mu,
            temp_buf + 2 * num_limbs
        );
        break;
    }
    case EXPR_INT_ADD: {
        uint32_t *c = temp_buf;
        compute(
            c,
            expr_ops,
            e.data0,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            temp_buf + num_limbs
        );
        limb_int_add(c, c, (int32_t)e.data1, num_limbs, limb_bits);
        limb_mod_reduce(result, c, meta->prime_limbs, num_limbs, limb_bits, meta->barrett_mu);
        break;
    }
    case EXPR_INT_MUL: {
        uint32_t *c = temp_buf;
        compute(
            c,
            expr_ops,
            e.data0,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            temp_buf + 2 * num_limbs
        );
        limb_int_mul(c, c, (int32_t)e.data1, num_limbs, limb_bits, meta->prime_limbs);
        limb_mod_reduce(result, c, meta->prime_limbs, num_limbs, limb_bits, meta->barrett_mu);
        break;
    }
    case EXPR_SELECT: {
        bool f = flags[e.data0];
        uint32_t idx = f ? e.data1 : e.data2;
        compute(result, expr_ops, idx, meta, inputs, vars, flags, num_limbs, limb_bits, temp_buf);
        break;
    }
    default: {
        for (uint32_t i = 0; i < num_limbs; i++) {
            result[i] = 0;
        }
        break;
    }
    }
}

// Raw evaluator: walks op stream, performs no modular reduction
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
    uint32_t &real_num_limbs,
    uint32_t *temp_storage
) {
    DecodedExpr d = decode_expr_op(expr_ops[op_idx]);
    uint32_t *c1_res = temp_storage;
    uint32_t *c2_res = temp_storage + 2 * num_limbs;

    switch (d.kind) {
    case EXPR_INPUT:
        for (uint32_t i = 0; i < num_limbs; i++) {
            result[i] = inputs[d.data0 * num_limbs + i];
            result[i + num_limbs] = 0;
        }
        max_limb_abs = (1 << limb_bits) - 1;
        real_num_limbs = num_limbs;
        break;
    case EXPR_VAR:
        for (uint32_t i = 0; i < num_limbs; i++) {
            result[i] = vars[d.data0 * num_limbs + i];
            result[i + num_limbs] = 0;
        }
        max_limb_abs = (1 << limb_bits) - 1;
        real_num_limbs = num_limbs;
        break;
    case EXPR_CONST: {
        uint32_t idx = d.data0;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < idx; i++) {
            offset += meta->const_limb_counts[i];
        }
        uint32_t count = meta->const_limb_counts[idx];
        for (uint32_t i = 0; i < 2 * num_limbs; i++) {
            result[i] = (i < count) ? meta->constants[offset + i] : 0;
        }
        max_limb_abs = (1 << limb_bits) - 1;
        real_num_limbs = count;
        break;
    }
    case EXPR_ADD: {
        uint32_t max_limb_abs_0, max_limb_abs_1;
        uint32_t real_num_limbs_0, real_num_limbs_1;
        evaluate_overflow(
            c1_res,
            expr_ops,
            d.data0,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            max_limb_abs_0,
            real_num_limbs_0,
            temp_storage + 4 * num_limbs
        );
        evaluate_overflow(
            c2_res,
            expr_ops,
            d.data1,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            max_limb_abs_1,
            real_num_limbs_1,
            temp_storage + 4 * num_limbs
        );
        limb_add_raw(result, c1_res, c2_res, num_limbs);
        max_limb_abs = max_limb_abs_0 + max_limb_abs_1;
        real_num_limbs = max(real_num_limbs_0, real_num_limbs_1);
        break;
    }
    case EXPR_SUB: {
        uint32_t max_limb_abs_0, max_limb_abs_1;
        uint32_t real_num_limbs_0, real_num_limbs_1;
        evaluate_overflow(
            c1_res,
            expr_ops,
            d.data0,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            max_limb_abs_0,
            real_num_limbs_0,
            temp_storage + 4 * num_limbs
        );
        evaluate_overflow(
            c2_res,
            expr_ops,
            d.data1,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            max_limb_abs_1,
            real_num_limbs_1,
            temp_storage + 4 * num_limbs
        );
        limb_sub_raw(result, c1_res, c2_res, num_limbs);
        max_limb_abs = max_limb_abs_0 + max_limb_abs_1;
        real_num_limbs = max(real_num_limbs_0, real_num_limbs_1);
        break;
    }
    case EXPR_MUL: {
        uint32_t max_limb_abs_0, max_limb_abs_1;
        uint32_t real_num_limbs_0, real_num_limbs_1;
        evaluate_overflow(
            c1_res,
            expr_ops,
            d.data0,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            max_limb_abs_0,
            real_num_limbs_0,
            temp_storage + 4 * num_limbs
        );
        evaluate_overflow(
            c2_res,
            expr_ops,
            d.data1,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            max_limb_abs_1,
            real_num_limbs_1,
            temp_storage + 4 * num_limbs
        );
        limb_mul_raw(result, c1_res, c2_res, num_limbs);
        max_limb_abs = max_limb_abs_0 * max_limb_abs_1 * min(real_num_limbs_0, real_num_limbs_1);
        real_num_limbs = real_num_limbs_0 + real_num_limbs_1 - 1;
        break;
    }
    case EXPR_SELECT: {
        uint32_t max_limb_abs_0, max_limb_abs_1;
        uint32_t real_num_limbs_0, real_num_limbs_1;
        if (flags[d.data0]) {
            evaluate_overflow(
                result,
                expr_ops,
                d.data2,
                meta,
                inputs,
                vars,
                flags,
                num_limbs,
                limb_bits,
                max_limb_abs_1,
                real_num_limbs_1,
                temp_storage
            );
            evaluate_overflow(
                result,
                expr_ops,
                d.data1,
                meta,
                inputs,
                vars,
                flags,
                num_limbs,
                limb_bits,
                max_limb_abs_0,
                real_num_limbs_0,
                temp_storage
            );
        } else {
            evaluate_overflow(
                result,
                expr_ops,
                d.data1,
                meta,
                inputs,
                vars,
                flags,
                num_limbs,
                limb_bits,
                max_limb_abs_0,
                real_num_limbs_0,
                temp_storage
            );
            evaluate_overflow(
                result,
                expr_ops,
                d.data2,
                meta,
                inputs,
                vars,
                flags,
                num_limbs,
                limb_bits,
                max_limb_abs_1,
                real_num_limbs_1,
                temp_storage
            );
        }
        max_limb_abs = max(max_limb_abs_0, max_limb_abs_1);
        real_num_limbs = max(real_num_limbs_0, real_num_limbs_1);
        break;
    }
    case EXPR_INT_ADD: {
        uint32_t max_limb_abs_0, real_num_limbs_0;
        evaluate_overflow(
            c1_res,
            expr_ops,
            d.data0,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            max_limb_abs_0,
            real_num_limbs_0,
            temp_storage + 4 * num_limbs
        );
        for (uint32_t i = 0; i < 2 * num_limbs; i++) {
            result[i] = c1_res[i];
        }
        int32_t scalar = (int32_t)d.data1;
        result[0] = (uint32_t)((int32_t)result[0] + scalar);
        max_limb_abs = max_limb_abs_0 + abs(scalar);
        real_num_limbs = real_num_limbs_0;
        break;
    }
    case EXPR_INT_MUL: {
        uint32_t max_limb_abs_0, real_num_limbs_0;
        evaluate_overflow(
            c1_res,
            expr_ops,
            d.data0,
            meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            max_limb_abs_0,
            real_num_limbs_0,
            temp_storage + 4 * num_limbs
        );
        int32_t scalar = (int32_t)d.data1;
        for (uint32_t i = 0; i < 2 * num_limbs; i++) {
            result[i] = (uint32_t)((int32_t)c1_res[i] * scalar);
        }
        max_limb_abs = max_limb_abs_0 * abs(scalar);
        real_num_limbs = real_num_limbs_0;
        break;
    }
    default:
        assert(false);
        break;
    }
}