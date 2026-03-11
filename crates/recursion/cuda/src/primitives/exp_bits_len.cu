#include "fp.h"
#include "launcher.cuh"
#include "primitives/trace_access.h"
#include "util.cuh"

#include <cassert>
#include <cstddef>
#include <cstdint>

constexpr uint8_t kExpBitsLenNumBitsMax = 31;
constexpr uint8_t kExpBitsLenNumRows = kExpBitsLenNumBitsMax + 1;
constexpr uint8_t kExpBitsLenLowBitsCount = 27;

struct ExpBitsLenRecord {
    uint8_t num_bits;
    Fp base;
    Fp bit_src;
    uint32_t row_offset;
    uint8_t shift_bits;
    uint32_t shift_mult;
};

template <typename T> struct ExpBitsLenCols {
    T is_valid;
    T is_first;
    T bit_idx;
    T base;
    T bit_src;
    T num_bits;
    T apply_bit;
    T low_bits_left;
    T in_low_region;
    T result;
    T result_multiplier;
    T bit_src_mod_2;
    T low_bits_are_zero;
    T high_bits_all_one;
    T bit_src_original;
    T shift_mult;
};

__global__ void exp_bits_len_tracegen_kernel(
    const ExpBitsLenRecord *records,
    size_t num_requests,
    Fp *trace,
    size_t height,
    size_t num_valid_rows
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_requests + (height - num_valid_rows)) {
        return;
    }

    if (idx >= num_requests) {
        RowSlice dummy_row(trace + num_valid_rows + (idx - num_requests), height);
        COL_WRITE_VALUE(dummy_row, ExpBitsLenCols, result, Fp::one());
        COL_WRITE_VALUE(dummy_row, ExpBitsLenCols, result_multiplier, Fp::one());
        return;
    }

    const ExpBitsLenRecord record = records[idx];
    assert(record.num_bits <= kExpBitsLenNumBitsMax);

    Fp bases[kExpBitsLenNumRows];
    Fp results[kExpBitsLenNumRows];

    bases[0] = record.base;
    for (uint8_t step = 1; step < kExpBitsLenNumRows; ++step) {
        bases[step] = bases[step - 1] * bases[step - 1];
    }

    for (uint8_t step = 0; step < kExpBitsLenNumRows; ++step) {
        results[step] = Fp::one();
    }

    const uint32_t bit_src_uint = record.bit_src.asUInt32();
    Fp acc = Fp::one();
    for (int step = kExpBitsLenNumBitsMax - 1; step >= 0; --step) {
        if (step < record.num_bits && ((bit_src_uint >> step) & 1) == 1) {
            acc = acc * bases[step];
        }
        results[step] = acc;
    }

    bool low_bits_are_zero = true;
    bool high_bits_all_one = false;
    for (uint8_t step = 0; step < kExpBitsLenNumRows; ++step) {
        if (step == kExpBitsLenLowBitsCount) {
            high_bits_all_one = true;
        }

        const uint32_t shifted = bit_src_uint >> step;
        const uint8_t num_bits = step < record.num_bits ? record.num_bits - step : 0;
        const uint8_t low_bits_left =
            step < kExpBitsLenLowBitsCount ? kExpBitsLenLowBitsCount - step : 0;
        RowSlice row(trace + record.row_offset + step, height);

        COL_WRITE_VALUE(row, ExpBitsLenCols, is_valid, Fp::one());
        COL_WRITE_VALUE(row, ExpBitsLenCols, is_first, bool_to_fp(step == 0));
        COL_WRITE_VALUE(row, ExpBitsLenCols, bit_idx, Fp(step));
        COL_WRITE_VALUE(row, ExpBitsLenCols, base, bases[step]);
        COL_WRITE_VALUE(row, ExpBitsLenCols, bit_src, Fp(shifted));
        COL_WRITE_VALUE(row, ExpBitsLenCols, num_bits, Fp(num_bits));
        COL_WRITE_VALUE(row, ExpBitsLenCols, apply_bit, bool_to_fp(num_bits != 0));
        COL_WRITE_VALUE(row, ExpBitsLenCols, low_bits_left, Fp(low_bits_left));
        COL_WRITE_VALUE(row, ExpBitsLenCols, in_low_region, bool_to_fp(low_bits_left != 0));
        COL_WRITE_VALUE(row, ExpBitsLenCols, result, results[step]);
        COL_WRITE_VALUE(
            row,
            ExpBitsLenCols,
            result_multiplier,
            num_bits != 0 && (shifted & 1) == 1 ? bases[step] : Fp::one()
        );
        COL_WRITE_VALUE(row, ExpBitsLenCols, bit_src_mod_2, Fp(shifted & 1));
        COL_WRITE_VALUE(row, ExpBitsLenCols, low_bits_are_zero, bool_to_fp(low_bits_are_zero));
        COL_WRITE_VALUE(row, ExpBitsLenCols, high_bits_all_one, bool_to_fp(high_bits_all_one));
        COL_WRITE_VALUE(row, ExpBitsLenCols, bit_src_original, Fp(bit_src_uint));
        COL_WRITE_VALUE(
            row,
            ExpBitsLenCols,
            shift_mult,
            step == record.shift_bits ? Fp(record.shift_mult) : Fp::zero()
        );

        if (step < kExpBitsLenLowBitsCount) {
            low_bits_are_zero = low_bits_are_zero && ((shifted & 1) == 0);
        } else if (step + 1 < kExpBitsLenNumRows) {
            high_bits_all_one = high_bits_all_one && ((shifted & 1) == 1);
        }
    }
}

extern "C" int _exp_bits_len_tracegen(
    const ExpBitsLenRecord *d_requests,
    size_t num_requests,
    Fp *d_trace,
    size_t height,
    size_t num_valid_rows
) {
    size_t total_jobs = num_requests + (height - num_valid_rows);
    if (total_jobs > 0) {
        auto [grid, block] = kernel_launch_params(total_jobs);
        exp_bits_len_tracegen_kernel<<<grid, block>>>(
            d_requests, num_requests, d_trace, height, num_valid_rows
        );
        int err = CHECK_KERNEL();
        if (err != 0) {
            return err;
        }
    }

    return 0;
}
