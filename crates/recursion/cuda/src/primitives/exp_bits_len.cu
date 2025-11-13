#include "fp.h"
#include "launcher.cuh"
#include "primitives/trace_access.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

constexpr uint8_t kExpBitsLenNumBitsMax = 31;

// Lookup table for inverses of all num_bits in [0, kExpBitsLenNumBitsMax]
__device__ __constant__ Fp kExpBitsLenNumBitsInv[kExpBitsLenNumBitsMax + 1] = {
    Fp(0x00000000),
    Fp(0x00000001),
    Fp(0x3c000001),
    Fp(0x50000001),
    Fp(0x5a000001),
    Fp(0x60000001),
    Fp(0x64000001),
    Fp(0x336db6dc),
    Fp(0x69000001),
    Fp(0x1aaaaaab),
    Fp(0x6c000001),
    Fp(0x20ba2e8c),
    Fp(0x6e000001),
    Fp(0x1bb13b14),
    Fp(0x19b6db6e),
    Fp(0x70000001),
    Fp(0x70800001),
    Fp(0x38787879),
    Fp(0x49555556),
    Fp(0x5ebca1b0),
    Fp(0x72000001),
    Fp(0x6124924a),
    Fp(0x105d1746),
    Fp(0x3e9bd37b),
    Fp(0x73000001),
    Fp(0x5b333334),
    Fp(0x0dd89d8a),
    Fp(0x08e38e39),
    Fp(0x0cdb6db7),
    Fp(0x14b08d3e),
    Fp(0x74000001),
    Fp(0x03def7be),
};

struct ExpBitsLenRecord {
    uint8_t num_bits;
    Fp base;
    Fp bit_src;
    uint32_t row_offset;
};

template <typename T>
struct ExpBitsLenCols {
    T is_valid;
    T base;
    T bit_src;
    T num_bits;
    T num_bits_inv;
    T result;
    T sub_result;
    T bit_src_div2;
    T bit_src_mod2;
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
        return;
    }

    const ExpBitsLenRecord record = records[idx];
    assert(record.num_bits <= kExpBitsLenNumBitsMax);

    Fp base_pow = record.base;
    for (uint8_t i = 0; i <= record.num_bits; i++) {
        RowSlice base_row(trace + record.row_offset + (record.num_bits - i), height);
        COL_WRITE_VALUE(base_row, ExpBitsLenCols, base, base_pow);
        base_pow = base_pow * base_pow;
    }

    uint32_t bit_src_uint = record.bit_src.asUInt32();

    // First iteration j = 0
    uint32_t shifted = bit_src_uint >> record.num_bits;
    RowSlice row(trace + record.row_offset, height);
    COL_WRITE_VALUE(row, ExpBitsLenCols, is_valid, Fp::one());
    COL_WRITE_VALUE(row, ExpBitsLenCols, bit_src, Fp(shifted));
    COL_WRITE_VALUE(row, ExpBitsLenCols, num_bits, Fp::zero());
    COL_WRITE_VALUE(row, ExpBitsLenCols, num_bits_inv, Fp::zero());
    COL_WRITE_VALUE(row, ExpBitsLenCols, result, Fp::one());
    COL_WRITE_VALUE(row, ExpBitsLenCols, sub_result, Fp::one());
    COL_WRITE_VALUE(row, ExpBitsLenCols, bit_src_div2, Fp(shifted >> 1));
    COL_WRITE_VALUE(row, ExpBitsLenCols, bit_src_mod2, Fp(shifted & 1));

    Fp acc = Fp::one();
    for (uint8_t j = 1; j <= record.num_bits; j++) {
        shifted = bit_src_uint >> (record.num_bits - j);

        row = RowSlice(trace + record.row_offset + j, height);
        COL_WRITE_VALUE(row, ExpBitsLenCols, is_valid, Fp::one());
        COL_WRITE_VALUE(row, ExpBitsLenCols, bit_src, Fp(shifted));
        COL_WRITE_VALUE(row, ExpBitsLenCols, num_bits, Fp(j));
        COL_WRITE_VALUE(row, ExpBitsLenCols, num_bits_inv, kExpBitsLenNumBitsInv[j]);
        COL_WRITE_VALUE(row, ExpBitsLenCols, sub_result, acc);
        COL_WRITE_VALUE(row, ExpBitsLenCols, bit_src_div2, Fp(shifted >> 1));
        COL_WRITE_VALUE(row, ExpBitsLenCols, bit_src_mod2, Fp(shifted & 1));

        if (shifted & 1) acc = acc * row[COL_INDEX(ExpBitsLenCols, base)];
        COL_WRITE_VALUE(row, ExpBitsLenCols, result, acc);
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
            d_requests,
            num_requests,
            d_trace,
            height,
            num_valid_rows
        );
        int err = CHECK_KERNEL();
        if (err != 0) {
            return err;
        }
    }

    return 0;
}
