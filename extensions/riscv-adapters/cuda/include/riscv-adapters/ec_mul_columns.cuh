#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv-adapters/ec_mul_replay.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

#include <cstddef>
#include <cstdint>

using namespace riscv;

// Device mirror of the `EC_MUL` row layout: the per-row header and the digest row's memory
// witnesses. The expression-dependent passes are in `algebra/ec_mul_tracegen.cuh`.
//
// The column structs must match `extensions/ecc/circuit/src/weierstrass_chip/mul/columns.rs` field
// for field. `COL_INDEX` resolves a column by `offsetof` on the `uint8_t` instantiation, so a
// reordered field writes to the wrong column rather than failing to compile. The width asserts
// below and `ec_mul_column_widths_match_the_cuda_mirror` pin the same three numbers.

// Restated from `mul/mod.rs` so the column structs are fixed-size.
static constexpr size_t EC_MUL_SCALAR_BITS = 256;
static constexpr size_t EC_MUL_STEPS_PER_ROW = 2;
static constexpr size_t EC_MUL_SIGN_PATTERNS = size_t(1) << EC_MUL_STEPS_PER_ROW;
static constexpr size_t EC_MUL_COMPUTE_ROWS = EC_MUL_SCALAR_BITS / EC_MUL_STEPS_PER_ROW;
static constexpr size_t EC_MUL_TOTAL_ROWS = EC_MUL_COMPUTE_ROWS + 1;
static constexpr size_t EC_MUL_DIGEST_ROW_IDX = EC_MUL_COMPUTE_ROWS;
static constexpr size_t EC_MUL_SCALAR_LIMBS = EC_MUL_SCALAR_BITS / 8;
// Sized to one row's contribution, making the accumulator recurrence a shift.
static constexpr size_t EC_MUL_SCALAR_ACC_LIMBS = EC_MUL_SCALAR_BITS / EC_MUL_STEPS_PER_ROW;
static constexpr size_t EC_MUL_SCALAR_ACC_LIMBS_PER_BYTE = 8 / EC_MUL_STEPS_PER_ROW;

static_assert(EC_MUL_SCALAR_LIMBS == EC_MUL_SCALAR_BLOCKS * MEMORY_BLOCK_BYTES);
static_assert(EC_MUL_SCALAR_BITS % EC_MUL_STEPS_PER_ROW == 0);
static_assert(8 % EC_MUL_STEPS_PER_ROW == 0);

// `2B + 1` overflowed 256 bits, so the scalar was not below the curve order.
static constexpr uint32_t EC_MUL_SCALAR_OVERFLOW = 0x56020002;
// The expression blob does not describe this chip's ladder step.
static constexpr uint32_t EC_MUL_BAD_PROGRAM = 0x56020003;
// A ladder accumulator was the identity, which the preconditions exclude.
static constexpr uint32_t EC_MUL_IDENTITY_ACCUMULATOR = 0x56020004;
// Present on every row. `is_compute` doubles as the expression's `is_valid`.
template <typename T> struct EcMulHeaderCols {
    T is_compute;
    T is_digest;
    T is_first_compute;
    T is_setup;
    T is_ladder;
    T is_real_digest;
    T row_idx;
    T scalar_acc[EC_MUL_SCALAR_ACC_LIMBS];
};

// Present only on the digest row, which carries the instruction's memory I/O.
template <typename T, size_t NUM_LIMBS, size_t BLOCKS> struct EcMulDigestCols {
    ExecutionState<T> from_state;

    T rd_ptr;
    T rs1_ptr;
    T rs2_ptr;

    T rd_val[PTR_U16_LIMBS];
    T rs1_val[PTR_U16_LIMBS];
    T rs2_val[PTR_U16_LIMBS];

    MemoryReadAuxCols<T> rs_read_aux[EC_MUL_REGISTER_READS];

    T point_x[NUM_LIMBS];
    T point_y[NUM_LIMBS];
    MemoryReadAuxCols<T> point_read_aux[BLOCKS];

    T scalar_data[EC_MUL_SCALAR_LIMBS];
    MemoryReadAuxCols<T> scalar_read_aux[EC_MUL_SCALAR_BLOCKS];
    T scalar_carry[EC_MUL_SCALAR_LIMBS];

    T result_x[NUM_LIMBS];
    T result_y[NUM_LIMBS];
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> write_aux[BLOCKS];
};

static constexpr size_t EC_MUL_HEADER_WIDTH = sizeof(EcMulHeaderCols<uint8_t>);

template <size_t NUM_LIMBS, size_t BLOCKS>
static constexpr size_t EC_MUL_DIGEST_WIDTH = sizeof(EcMulDigestCols<uint8_t, NUM_LIMBS, BLOCKS>);

static_assert(EC_MUL_HEADER_WIDTH == 135);
static_assert(EC_MUL_DIGEST_WIDTH<32, 8> == 281);
static_assert(EC_MUL_DIGEST_WIDTH<48, 12> == 377);

// One byte of a projected memory operand, stored by the gather as `u16` cells.
template <size_t N>
static __device__ __forceinline__ uint8_t ec_mul_block_byte(
    const uint16_t (&blocks)[N][BLOCK_FE_WIDTH], size_t byte
) {
    uint16_t cell = blocks[byte / MEMORY_BLOCK_BYTES][(byte % MEMORY_BLOCK_BYTES) / 2];
    return static_cast<uint8_t>(byte % 2 == 0 ? cell : cell >> 8);
}

// The one-hot flag index for compute row `row`, digits most significant first.
//
// Digit `i` is bit `i + 1` of the scalar, since the multiplier is `2B + 1`. The most significant
// digit has no bit above it and is always negative, so the accumulator seeds itself from `P`.
static __device__ __forceinline__ uint32_t ec_mul_sign_pattern_for_row(
    const uint8_t *scalar, size_t row
) {
    uint32_t pattern = 0;
    for (size_t step = 0; step < EC_MUL_STEPS_PER_ROW; step++) {
        size_t digit = EC_MUL_SCALAR_BITS - 1 - (row * EC_MUL_STEPS_PER_ROW + step);
        size_t bit_index = digit + 1;
        bool bit = bit_index < EC_MUL_SCALAR_BITS &&
                   ((scalar[bit_index / 8] >> (bit_index % 8)) & 1) != 0;
        pattern |= uint32_t(bit) << (EC_MUL_STEPS_PER_ROW - 1 - step);
    }
    return pattern;
}

// Fills the header columns of one row. `row` points at the first header column.
//
// `scalar_acc` holds the accumulator entering the row, so on compute row `r` limb `j` is the sign
// pattern of row `r - 1 - j` for `j < r` and zero above. The host produces the same values by
// shifting a rolling array; writing them positionally costs the same and needs no carried state. A
// setup row never accumulates, so its limbs stay zero.
static __device__ void fill_ec_mul_header(
    RowSlice row, const uint8_t *scalar, size_t row_idx, bool is_setup
) {
    bool is_compute = row_idx < EC_MUL_COMPUTE_ROWS;
    bool is_digest = row_idx == EC_MUL_DIGEST_ROW_IDX;
    bool is_first_compute = is_compute && row_idx == 0;
    bool is_ladder = is_compute && !is_setup && row_idx != 0;

    COL_WRITE_VALUE(row, EcMulHeaderCols, is_compute, uint32_t(is_compute));
    COL_WRITE_VALUE(row, EcMulHeaderCols, is_digest, uint32_t(is_digest));
    COL_WRITE_VALUE(row, EcMulHeaderCols, is_first_compute, uint32_t(is_first_compute));
    COL_WRITE_VALUE(row, EcMulHeaderCols, is_setup, uint32_t(is_setup));
    COL_WRITE_VALUE(row, EcMulHeaderCols, is_ladder, uint32_t(is_ladder));
    COL_WRITE_VALUE(row, EcMulHeaderCols, is_real_digest, uint32_t(is_digest && !is_setup));
    COL_WRITE_VALUE(row, EcMulHeaderCols, row_idx, uint32_t(row_idx));

    size_t acc_base = COL_INDEX(EcMulHeaderCols, scalar_acc);
#pragma unroll 1
    for (size_t limb = 0; limb < EC_MUL_SCALAR_ACC_LIMBS; limb++) {
        uint32_t value = (!is_setup && limb < row_idx)
                             ? ec_mul_sign_pattern_for_row(scalar, row_idx - 1 - limb)
                             : 0;
        row.write(acc_base + limb, Fp(value));
    }
}

// Fills the digest row from one instruction's projection.
template <size_t NUM_LIMBS, size_t BLOCKS> struct EcMulDigestFiller {
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;
    uint32_t pointer_max_bits;

    __device__ EcMulDigestFiller(
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits,
        uint32_t pointer_max_bits
    )
        : range_checker(range_checker), mem_helper(range_checker, timestamp_max_bits),
          pointer_max_bits(pointer_max_bits) {}

    template <typename T> using Cols = EcMulDigestCols<T, NUM_LIMBS, BLOCKS>;

    static_assert(2 * NUM_LIMBS == BLOCKS * MEMORY_BLOCK_BYTES);

    // `row` points at the first digest column. Returns false, setting `err`, if the scalar does
    // not fit; the gather has already validated every other input.
    __device__ __noinline__ bool fill(
        RowSlice row, const EcMulTraceInput<BLOCKS> &input, uint32_t *err
    ) {
        bool is_setup = input.is_setup != 0;

        COL_WRITE_VALUE(row, Cols, from_state.pc, input.from_pc);
        COL_WRITE_VALUE(row, Cols, from_state.timestamp, input.from_timestamp);

        // `reg_ptrs` and `reg_vals` are in timestamp order, `[rs1, rs2, rd]`, not column order.
        COL_WRITE_VALUE(row, Cols, rs1_ptr, input.reg_ptrs[0]);
        COL_WRITE_VALUE(row, Cols, rs2_ptr, input.reg_ptrs[1]);
        COL_WRITE_VALUE(row, Cols, rd_ptr, input.reg_ptrs[2]);

        Fp limbs[PTR_U16_LIMBS];
        ptr_to_u16_limbs(limbs, input.reg_vals[0]);
        COL_WRITE_ARRAY(row, Cols, rs1_val, limbs);
        ptr_to_u16_limbs(limbs, input.reg_vals[1]);
        COL_WRITE_ARRAY(row, Cols, rs2_val, limbs);
        ptr_to_u16_limbs(limbs, input.reg_vals[2]);
        COL_WRITE_ARRAY(row, Cols, rd_val, limbs);

        for (size_t reg = 0; reg < EC_MUL_REGISTER_READS; reg++) {
            range_checker.add_count(
                ptr_bound_from_high_u16(
                    static_cast<uint16_t>(input.reg_vals[reg] >> U16_BITS), pointer_max_bits
                ),
                U16_BITS
            );
        }

        size_t point_x_base = COL_INDEX(Cols, point_x);
        size_t point_y_base = COL_INDEX(Cols, point_y);
        for (size_t byte = 0; byte < 2 * NUM_LIMBS; byte++) {
            uint8_t value = ec_mul_block_byte(input.point_blocks, byte);
            size_t column =
                byte < NUM_LIMBS ? point_x_base + byte : point_y_base + (byte - NUM_LIMBS);
            row.write(column, Fp(uint32_t(value)));
        }

        size_t result_x_base = COL_INDEX(Cols, result_x);
        size_t result_y_base = COL_INDEX(Cols, result_y);
        for (size_t byte = 0; byte < 2 * NUM_LIMBS; byte++) {
            uint8_t value = ec_mul_block_byte(input.write_blocks, byte);
            size_t column =
                byte < NUM_LIMBS ? result_x_base + byte : result_y_base + (byte - NUM_LIMBS);
            row.write(column, Fp(uint32_t(value)));
        }

        size_t scalar_base = COL_INDEX(Cols, scalar_data);
        for (size_t byte = 0; byte < EC_MUL_SCALAR_LIMBS; byte++) {
            uint32_t limb = ec_mul_block_byte(input.scalar_blocks, byte);
            row.write(scalar_base + byte, Fp(limb));
        }

        if (!fill_scalar_carries(row, input, is_setup, err)) {
            return false;
        }

        fill_memory_aux(row, input);
        return true;
    }

  private:
    // Carries for the `2B + 1 == scalar` check, byte by byte; byte 0's incoming carry is the `+1`.
    // A setup row leaves them zero, its accumulator having stayed zero and the check gated off.
    __device__ bool fill_scalar_carries(
        RowSlice row, const EcMulTraceInput<BLOCKS> &input, bool is_setup, uint32_t *err
    ) {
        if (is_setup) {
            return true;
        }
        const uint8_t *scalar_bytes = reinterpret_cast<const uint8_t *>(&input.scalar_blocks[0][0]);
        size_t carry_base = COL_INDEX(Cols, scalar_carry);
        uint32_t carry = 1;
        for (size_t byte = 0; byte < EC_MUL_SCALAR_LIMBS; byte++) {
            uint32_t accumulated = 0;
            for (size_t limb = 0; limb < EC_MUL_SCALAR_ACC_LIMBS_PER_BYTE; limb++) {
                size_t index = byte * EC_MUL_SCALAR_ACC_LIMBS_PER_BYTE + limb;
                // Limb `j` of the digest row's accumulator came from compute row
                // `EC_MUL_COMPUTE_ROWS - 1 - j`.
                uint32_t pattern = ec_mul_sign_pattern_for_row(
                    scalar_bytes, EC_MUL_COMPUTE_ROWS - 1 - index
                );
                accumulated += pattern << (limb * EC_MUL_STEPS_PER_ROW);
            }
            carry = (accumulated * 2 + carry) >> 8;
            row.write(carry_base + byte, Fp(carry));
        }
        if (carry != 0) {
            preflight_set_error(err, EC_MUL_SCALAR_OVERFLOW);
            return false;
        }
        return true;
    }

    // Timestamps run forward in the order the AIR consumes them: registers, point blocks, scalar
    // blocks, result writes.
    __device__ void fill_memory_aux(RowSlice row, const EcMulTraceInput<BLOCKS> &input) {
        uint32_t timestamp = input.from_timestamp;

        for (size_t reg = 0; reg < EC_MUL_REGISTER_READS; reg++) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Cols, rs_read_aux[reg])),
                input.reg_prev_timestamps[reg],
                timestamp++
            );
        }
        for (size_t block = 0; block < BLOCKS; block++) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Cols, point_read_aux[block])),
                input.point_prev_timestamps[block],
                timestamp++
            );
        }
        for (size_t block = 0; block < EC_MUL_SCALAR_BLOCKS; block++) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Cols, scalar_read_aux[block])),
                input.scalar_prev_timestamps[block],
                timestamp++
            );
        }
        for (size_t block = 0; block < BLOCKS; block++) {
            uint8_t previous[MEMORY_BLOCK_BYTES];
            for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
                uint16_t cell = input.write_predecessors[block][limb];
                previous[2 * limb] = static_cast<uint8_t>(cell);
                previous[2 * limb + 1] = static_cast<uint8_t>(cell >> 8);
            }
            Fp packed[BLOCK_FE_WIDTH];
            pack_u8_block_bytes(packed, previous);
            COL_WRITE_ARRAY(row, Cols, write_aux[block].prev_data, packed);
            mem_helper.fill(
                row.slice_from(COL_INDEX(Cols, write_aux[block])),
                input.write_prev_timestamps[block],
                timestamp++
            );
        }
    }
};
