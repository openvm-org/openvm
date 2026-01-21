#include "fp.h"
#include "launcher.cuh"
#include "primitives/trace_access.h"

template <typename T> struct BitwiseOperationLookupCols {
    T x_bits[8];
    T y_bits[8];
    T mult_range;
    T mult_xor;
};

__global__ void bitwise_op_lookup_tracegen(
    const uint32_t *count,
    const uint32_t *cpu_count,
    Fp *trace,
    uint32_t num_rows,
    uint32_t num_bits
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < num_rows) {
        uint32_t x = row_idx / (1U << num_bits);
        uint32_t y = row_idx % (1U << num_bits);

        Fp x_bits_array[8];
        Fp y_bits_array[8];
        for (uint32_t i = 0; i < num_bits; i++) {
            x_bits_array[i] = Fp((x >> i) & 1);
            y_bits_array[i] = Fp((y >> i) & 1);
        }

        uint32_t mult_range_val = count[row_idx] + (cpu_count ? cpu_count[row_idx] : 0);
        uint32_t mult_xor_val = count[row_idx + num_rows] + 
                                (cpu_count ? cpu_count[row_idx + num_rows] : 0);

        RowSlice row(trace + row_idx, num_rows);
        row.write_array(COL_INDEX(BitwiseOperationLookupCols, x_bits), num_bits, x_bits_array);
        row.write_array(COL_INDEX(BitwiseOperationLookupCols, y_bits), num_bits, y_bits_array);
        COL_WRITE_VALUE(row, BitwiseOperationLookupCols, mult_range, mult_range_val);
        COL_WRITE_VALUE(row, BitwiseOperationLookupCols, mult_xor, mult_xor_val);
    }
}

extern "C" int _bitwise_op_lookup_tracegen(
    const uint32_t *d_count,
    const uint32_t *d_cpu_count,
    Fp *d_trace,
    uint32_t num_bits
) {
    uint32_t num_rows = 1 << (2 * num_bits);
    auto [grid, block] = kernel_launch_params(num_rows);
    bitwise_op_lookup_tracegen<<<grid, block>>>(d_count, d_cpu_count, d_trace, num_rows, num_bits);
    return CHECK_KERNEL();
}
