#pragma once

#include "algebra/ec_mul_projective.cuh"
#include "algebra/field_expr_core.cuh"
#include "riscv-adapters/ec_mul_columns.cuh"

#include <cstddef>
#include <cstdint>

// Trace generation for the `EC_MUL` chip: the passes driving the field-expression interpreter. The
// row layout they write around is in `ec_mul_columns.cuh`.

// The accumulator a setup row carries, mirroring `SETUP_ACC` in `mul/field_expr.rs`. The setup
// check does not pin it, but both backends must choose the same value or the memory argument will
// not balance.
static constexpr uint32_t EC_MUL_SETUP_ACC_X = 2;
static constexpr uint32_t EC_MUL_SETUP_ACC_Y = 1;

// Shape of the ladder-step expression declared in `mul/field_expr.rs`.
static constexpr uint32_t EC_MUL_EXPR_NUM_INPUTS = 4;
static constexpr uint32_t EC_MUL_EXPR_NUM_OUTPUTS = 2;

// Trace generation runs in two passes, split where the row dependency ends.
//
// A ladder row's inputs are the previous row's outputs, so the accumulator must be advanced
// sequentially within an instruction. `ec_mul_eval_instruction` does that projectively, one thread
// per instruction, recovering every row's affine accumulator with a single inversion. Each row is
// then self-contained, so `ec_mul_fill_row` writes it in full, one thread per row, doing that row's
// own divisions in parallel with every other row's.

// Checks that a blob describes this chip's ladder step before any row relies on its shape.
static __device__ bool ec_mul_program_matches(const FieldExprProg &s, uint32_t num_limbs) {
    return s.num_input == EC_MUL_EXPR_NUM_INPUTS && s.n_outputs == EC_MUL_EXPR_NUM_OUTPUTS &&
           s.num_flags == EC_MUL_SIGN_PATTERNS && s.num_limbs == num_limbs && s.needs_setup == 1 &&
           s.should_finalize == 0;
}

// A setup row's inputs, mirroring `setup_row_inputs`: the modulus, the setup values, then
// `EC_MUL_SETUP_ACC`.
static __device__ void ec_mul_setup_inputs(uint8_t *in_limbs, const FieldExprProg &s) {
    uint32_t nl = s.num_limbs;
    for (uint32_t byte = 0; byte < nl; byte++) {
        in_limbs[byte] = static_cast<uint8_t>(s.p[byte / 4] >> (8 * (byte % 4)));
    }
    for (uint32_t value = 0; value < s.n_setup_values; value++) {
        for (uint32_t byte = 0; byte < nl; byte++) {
            in_limbs[(value + 1) * nl + byte] =
                static_cast<uint8_t>(s.setup_values[value * nl + byte]);
        }
    }
    // Inputs the setup values do not cover stay zero.
    for (uint32_t byte = (s.n_setup_values + 1) * nl; byte < (s.num_input - 2) * nl; byte++) {
        in_limbs[byte] = 0;
    }
    uint8_t *acc = in_limbs + (s.num_input - 2) * nl;
    for (uint32_t byte = 0; byte < 2 * nl; byte++) {
        acc[byte] = 0;
    }
    acc[0] = static_cast<uint8_t>(EC_MUL_SETUP_ACC_X);
    acc[nl] = static_cast<uint8_t>(EC_MUL_SETUP_ACC_Y);
}

// A compute row's inputs: the base point, then the accumulator entering the row.
static __device__ void ec_mul_compute_inputs(
    uint8_t *in_limbs, uint32_t num_limbs, const uint8_t *point_bytes, const uint8_t *acc_bytes
) {
    for (uint32_t byte = 0; byte < 2 * num_limbs; byte++) {
        in_limbs[byte] = point_bytes[byte];
        in_limbs[2 * num_limbs + byte] = acc_bytes[byte];
    }
}

// The row mode for compute row `row_idx`.
static __device__ FieldExprRowMode ec_mul_row_mode(
    const uint8_t *scalar_bytes, size_t row_idx, bool is_setup
) {
    if (is_setup) {
        return FieldExprRowMode{0, true, false};
    }
    return FieldExprRowMode{uint32_t(1) << ec_mul_sign_pattern_for_row(scalar_bytes, row_idx),
                            false, false};
}

// Advances one instruction's ladder, writing the affine accumulator entering each compute row.
//
// A setup instruction has no ladder: every row carries `EC_MUL_SETUP_ACC`, which
// `ec_mul_setup_inputs` supplies directly, so nothing is written here.
//
// `ladder` is `EC_MUL_COMPUTE_ROWS * 4 * K` words of scratch: the Jacobian accumulators followed by
// the prefix products the batch inversion consumes.
template <uint32_t K, size_t BLOCKS>
static __device__ bool ec_mul_eval_instruction(
    const FieldExprProg &s,
    const EcMulTraceInput<BLOCKS> &input,
    uint8_t *affine_out,
    uint32_t *ladder,
    uint32_t *workspace,
    uint32_t *err
) {
    if (input.is_setup != 0) {
        return true;
    }
    auto *jacobian = reinterpret_cast<EcMulJacobian<K> *>(ladder);
    uint32_t *prefix = ladder + EC_MUL_COMPUTE_ROWS * 3 * K;
    const uint8_t *point_bytes = reinterpret_cast<const uint8_t *>(&input.point_blocks[0][0]);
    const uint8_t *scalar_bytes = reinterpret_cast<const uint8_t *>(&input.scalar_blocks[0][0]);

    return ec_mul_projective_accumulators<K>(
               s, point_bytes, scalar_bytes, jacobian, workspace, err
           ) &&
           ec_mul_affine_from_jacobian<K>(s, jacobian, prefix, affine_out, workspace, err);
}

// Writes one row of the trace.
//
// `dummy_expr` is the inactive expression witness that digest and padding rows carry, computed once
// per trace. It cannot be all zero: the curve's `a` coefficient is folded in as a constant, so on a
// zero row the lambda constraint evaluates to `-a` and the ungated carry recurrences are
// unsatisfiable whenever `a != 0`.
template <uint32_t K, size_t NUM_LIMBS, size_t BLOCKS>
static __device__ __noinline__ bool ec_mul_fill_row(
    const FieldExprProg &s,
    RowSlice row,
    size_t row_index,
    size_t used_rows,
    const EcMulTraceInput<BLOCKS> *projection,
    const uint8_t *affine,
    const Fp *dummy_expr,
    VariableRangeChecker range_checker,
    uint32_t timestamp_max_bits,
    uint32_t pointer_max_bits,
    uint32_t *aux,
    uint8_t *in_limbs,
    uint32_t *err
) {
    const size_t width = EC_MUL_HEADER_WIDTH + s.width + EC_MUL_DIGEST_WIDTH<NUM_LIMBS, BLOCKS>;
    const size_t expr_offset = EC_MUL_HEADER_WIDTH;
    const size_t digest_offset = expr_offset + s.width;
    row.fill_zero(0, width);

    // Padding rows carry the inactive witness with every selector clear.
    if (row_index >= used_rows) {
        row.write_array(expr_offset, s.width, dummy_expr);
        return true;
    }

    const size_t instruction = row_index / EC_MUL_TOTAL_ROWS;
    const size_t local_row = row_index % EC_MUL_TOTAL_ROWS;
    const EcMulTraceInput<BLOCKS> &input = projection[instruction];
    const bool is_setup = input.is_setup != 0;
    const uint8_t *scalar_bytes = reinterpret_cast<const uint8_t *>(&input.scalar_blocks[0][0]);

    fill_ec_mul_header(row, scalar_bytes, local_row, is_setup);

    if (local_row == EC_MUL_DIGEST_ROW_IDX) {
        row.write_array(expr_offset, s.width, dummy_expr);
        EcMulDigestFiller<NUM_LIMBS, BLOCKS> filler(
            range_checker, timestamp_max_bits, pointer_max_bits
        );
        return filler.fill(row.slice_from(digest_offset), input, err);
    }

    if (is_setup) {
        ec_mul_setup_inputs(in_limbs, s);
    } else {
        const size_t coordinates = 2 * static_cast<size_t>(s.num_limbs);
        const uint8_t *accumulator =
            affine + (instruction * EC_MUL_COMPUTE_ROWS + local_row) * coordinates;
        ec_mul_compute_inputs(
            in_limbs, s.num_limbs,
            reinterpret_cast<const uint8_t *>(&input.point_blocks[0][0]), accumulator
        );
    }

    FieldExprRowMode mode = ec_mul_row_mode(scalar_bytes, local_row, is_setup);
    return field_expr_fill_core_row<K>(
        s, row.slice_from(expr_offset), in_limbs, nullptr, mode, range_checker, aux, err
    );
}

// Builds the inactive expression witness that digest and padding rows carry.
//
// Built from the setup inputs rather than zeros, since the expression divides by `2*acc_y` without
// a guard. `is_valid` is then cleared, as `fill_dummy_core_row` does for the single-row chips. The
// AIR emits no range check when `is_valid` is zero, so the caller passes a throwaway histogram.
template <uint32_t K>
static __device__ bool ec_mul_build_dummy_expr(
    const FieldExprProg &s,
    RowSlice scratch_row,
    VariableRangeChecker discarded,
    uint32_t *aux,
    uint8_t *in_limbs,
    uint32_t *err
) {
    ec_mul_setup_inputs(in_limbs, s);
    FieldExprRowMode mode{0, true, false};
    if (!field_expr_fill_core_row<K>(
            s, scratch_row, in_limbs, nullptr, mode, discarded, aux, err
        )) {
        return false;
    }
    scratch_row.write(0, Fp(0u));
    return true;
}

// Checks the host's trace shape and buffer sizing against the blob before any row is written.
//
// The host derives the width and the buffer lengths from its own copies of these constants, so a
// mismatch means one side changed without the other, which would otherwise be an out-of-bounds
// write rather than a failure.
template <uint32_t K, size_t NUM_LIMBS, size_t BLOCKS>
static __device__ bool ec_mul_validate_trace_shape(
    const FieldExprProg &s,
    size_t width,
    size_t height,
    size_t num_instructions,
    size_t affine_bytes,
    size_t ladder_words
) {
    return ec_mul_program_matches(s, static_cast<uint32_t>(NUM_LIMBS)) && s.k == K &&
           width == EC_MUL_HEADER_WIDTH + s.width + EC_MUL_DIGEST_WIDTH<NUM_LIMBS, BLOCKS> &&
           num_instructions * EC_MUL_TOTAL_ROWS <= height &&
           affine_bytes >= num_instructions * EC_MUL_COMPUTE_ROWS * 2 * NUM_LIMBS &&
           ladder_words >= EC_MUL_COMPUTE_ROWS * 4 * K;
}
