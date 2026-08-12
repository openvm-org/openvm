#pragma once

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

// A ladder row's inputs are the previous row's outputs. The generic path uploads host-computed
// saved variables; supported device shapes replace the serial affine chain with projective
// preparation, batch normalization, and row-parallel evaluation. `ec_mul_fill_row` consumes either
// row-major host variables or variable-major device variables one thread per row. Witness
// generation depends only on a row's own variables and performs no modular inversions.

// Checks that a blob describes this chip's ladder step before any row relies on its shape.
static __device__ bool ec_mul_program_matches(const FieldExprProg &s, uint32_t num_limbs) {
    return s.num_input == EC_MUL_EXPR_NUM_INPUTS && s.n_outputs == EC_MUL_EXPR_NUM_OUTPUTS &&
           s.num_flags == EC_MUL_SIGN_PATTERNS && s.num_limbs == num_limbs && s.needs_setup == 1 &&
           s.should_finalize == 0;
}

// An evaluation's two outputs, as the little-endian bytes the next row reads as input.
static __device__ void ec_mul_output_bytes(
    uint8_t *out, const FieldExprProg &s, const uint32_t *var_canon, uint32_t k
) {
    for (uint32_t output = 0; output < EC_MUL_EXPR_NUM_OUTPUTS; output++) {
        const uint32_t *value = field_expr_var_limbs(var_canon, s.outputs[output], k);
        for (uint32_t byte = 0; byte < s.num_limbs; byte++) {
            out[output * s.num_limbs + byte] =
                static_cast<uint8_t>(value[byte / 4] >> (8 * (byte % 4)));
        }
    }
}

// An evaluation's two outputs from the row-major or variable-major saved-variable buffer.
static __device__ void ec_mul_output_bytes_from_layout(
    uint8_t *out,
    const FieldExprProg &s,
    const uint32_t *vars,
    size_t row,
    size_t total_rows,
    uint32_t k,
    bool vars_transposed
) {
    if (!vars_transposed) {
        ec_mul_output_bytes(out, s, vars + row * s.num_vars * k, k);
        return;
    }
    for (uint32_t output = 0; output < EC_MUL_EXPR_NUM_OUTPUTS; output++) {
        uint32_t var = s.outputs[output];
        for (uint32_t byte = 0; byte < s.num_limbs; byte++) {
            uint32_t word = vars[(var * k + byte / 4) * total_rows + row];
            out[output * s.num_limbs + byte] =
                static_cast<uint8_t>(word >> (8 * (byte % 4)));
        }
    }
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

// Writes one row of the trace.
//
// `dummy_expr` is the inactive expression witness that padding rows carry, computed once per
// trace. It cannot be all zero: the curve's `a` coefficient is folded in as a constant, so on a
// zero row the lambda constraint evaluates to `-a` and the ungated carry recurrences are
// unsatisfiable whenever `a != 0`.
template <uint32_t K, size_t NUM_LIMBS, size_t BLOCKS>
static __device__ __noinline__ bool ec_mul_fill_row(
    const FieldExprProg &s,
    RowSlice row,
    size_t row_index,
    size_t used_rows,
    const EcMulTraceInput<BLOCKS> *projection,
    const uint32_t *vars,
    bool vars_transposed,
    const Fp *dummy_expr,
    VariableRangeChecker range_checker,
    uint32_t timestamp_max_bits,
    uint32_t pointer_max_bits,
    uint32_t *aux,
    uint8_t *in_limbs,
    uint8_t *acc_bytes,
    uint32_t *err
) {
    const size_t width = EC_MUL_HEADER_WIDTH + s.width + EC_MUL_IO_WIDTH<NUM_LIMBS, BLOCKS>;
    const size_t expr_offset = EC_MUL_HEADER_WIDTH;
    const size_t io_offset = expr_offset + s.width;
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

    // The variables were computed on the host and uploaded; the witness needs the same inputs
    // that evaluation saw.
    const uint32_t vars_per_row = s.num_vars * K;
    const size_t vars_index = instruction * EC_MUL_COMPUTE_ROWS + local_row;
    const size_t vars_rows = (used_rows / EC_MUL_TOTAL_ROWS) * EC_MUL_COMPUTE_ROWS;
    for (uint32_t word = 0; word < vars_per_row; word++) {
        aux[word] = vars_transposed ? vars[word * vars_rows + vars_index]
                                    : vars[vars_index * vars_per_row + word];
    }

    if (is_setup) {
        ec_mul_setup_inputs(in_limbs, s);
    } else {
        if (local_row == 0) {
            for (uint32_t byte = 0; byte < 2 * s.num_limbs; byte++) {
                acc_bytes[byte] = ec_mul_block_byte(input.point_blocks, byte);
            }
        } else {
            ec_mul_output_bytes_from_layout(
                acc_bytes, s, vars, vars_index - 1, vars_rows, K, vars_transposed
            );
        }
        ec_mul_compute_inputs(
            in_limbs, s.num_limbs,
            reinterpret_cast<const uint8_t *>(&input.point_blocks[0][0]), acc_bytes
        );
    }

    FieldExprRowMode mode = ec_mul_row_mode(scalar_bytes, local_row, is_setup);
    if (!field_expr_fill_witness<K>(
            s, row.slice_from(expr_offset), in_limbs, mode, range_checker, aux, err
        )) {
        return false;
    }

    // The final compute row also carries the instruction's memory I/O.
    if (local_row == EC_MUL_FINAL_ROW_IDX) {
        EcMulIoFiller<NUM_LIMBS, BLOCKS> filler(
            range_checker, timestamp_max_bits, pointer_max_bits
        );
        return filler.fill(row.slice_from(io_offset), input, err);
    }
    return true;
}

// Builds the inactive expression witness that padding rows carry.
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
// The host derives the width and the variable-buffer length from its own copies of these constants,
// so a mismatch means one side changed without the other, which would otherwise be an
// out-of-bounds write rather than a failure.
template <uint32_t K, size_t NUM_LIMBS, size_t BLOCKS>
static __device__ bool ec_mul_validate_trace_shape(
    const FieldExprProg &s,
    size_t width,
    size_t height,
    size_t num_instructions,
    size_t vars_words
) {
    return ec_mul_program_matches(s, static_cast<uint32_t>(NUM_LIMBS)) && s.k == K &&
           width == EC_MUL_HEADER_WIDTH + s.width + EC_MUL_IO_WIDTH<NUM_LIMBS, BLOCKS> &&
           num_instructions * EC_MUL_TOTAL_ROWS <= height &&
           vars_words >= num_instructions * EC_MUL_COMPUTE_ROWS * s.num_vars * K;
}
