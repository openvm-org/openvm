#include "adapters/mul.cuh"
#include "constants.h"
#include "histogram.cuh"
#include "launcher.cuh"
#include "trace_access.h"

using namespace riscv;

struct Rv32MultiplicationCoreRecord {
    uint8_t b[RV32_REGISTER_NUM_LIMBS];
    uint8_t c[RV32_REGISTER_NUM_LIMBS];
};

template <typename T> struct Rv32MultiplicationCoreCols {
    T a[RV32_REGISTER_NUM_LIMBS];
    T b[RV32_REGISTER_NUM_LIMBS];
    T c[RV32_REGISTER_NUM_LIMBS];
    T is_valid;
};

__device__ void run_mul(const uint8_t *x, const uint8_t *y, uint8_t *out_a, uint32_t *carry) {
    for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
        uint32_t res = (i > 0) ? carry[i - 1] : 0;
        for (size_t j = 0; j <= i; j++) {
            res += static_cast<uint32_t>(x[j]) * static_cast<uint32_t>(y[i - j]);
        }
        carry[i] = res >> RV32_CELL_BITS;
        out_a[i] = static_cast<uint8_t>(res & ((1u << RV32_CELL_BITS) - 1));
    }
}

struct Rv32MultiplicationCore {
    __device__ Rv32MultiplicationCore() {}
    __device__ void fill_trace_row(RowSlice row, Rv32MultiplicationCoreRecord record) {
        uint8_t a[RV32_REGISTER_NUM_LIMBS];
        uint32_t carry_buf[RV32_REGISTER_NUM_LIMBS];
        run_mul(record.b, record.c, a, carry_buf);

        COL_WRITE_ARRAY(row, Rv32MultiplicationCoreCols, b, record.b);
        COL_WRITE_ARRAY(row, Rv32MultiplicationCoreCols, c, record.c);
        COL_WRITE_ARRAY(row, Rv32MultiplicationCoreCols, a, a);
        COL_WRITE_VALUE(row, Rv32MultiplicationCoreCols, is_valid, 1);
    }
};

template <typename T> struct Rv32MultiplicationCols {
    Rv32MultAdapterCols<T> adapter;
    Rv32MultiplicationCoreCols<T> core;
};

struct Rv32MultiplicationRecord {
    Rv32MultAdapterRecord adapter;
    Rv32MultiplicationCoreRecord core;
};

__global__ void mul_tracegen(
    Fp *d_trace,
    size_t height,
    uint8_t *d_records,
    size_t num_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < num_records) {
        auto rec = reinterpret_cast<Rv32MultiplicationRecord *>(d_records)[idx];

        Rv32MultAdapter adapter(VariableRangeChecker(d_range_checker_ptr, range_checker_bins));
        adapter.fill_trace_row(row, rec.adapter);

        Rv32MultiplicationCore core;
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv32MultiplicationCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv32MultiplicationCols<uint8_t>));
    }
}

extern "C" int _mul_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t *d_records,
    size_t record_len,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins
) {
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(Rv32MultiplicationRecord) >= record_len);
    assert(width == sizeof(Rv32MultiplicationCols<uint8_t>));
    size_t num_records = record_len / sizeof(Rv32MultiplicationRecord);
    auto [grid, block] = kernel_launch_params(height);
    mul_tracegen<<<grid, block>>>(
        d_trace, height, d_records, num_records, d_range_checker_ptr, range_checker_bins
    );
    return cudaGetLastError();
}