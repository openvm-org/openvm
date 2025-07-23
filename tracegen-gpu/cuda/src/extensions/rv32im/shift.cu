#include "adapters/alu.cuh"
#include "constants.h"
#include "cores/shift.cuh"
#include "histogram.cuh"
#include "launcher.cuh"
#include "trace_access.h"

using namespace riscv;
using namespace program;

// Concrete type aliases for 32-bit
using Rv32ShiftCoreRecord = ShiftCoreRecord<RV32_REGISTER_NUM_LIMBS>;
using Rv32ShiftCore = ShiftCore<RV32_REGISTER_NUM_LIMBS>;
template <typename T> using Rv32ShiftCoreCols = ShiftCoreCols<T, RV32_REGISTER_NUM_LIMBS>;

template <typename T> struct ShiftCols {
    Rv32BaseAluAdapterCols<T> adapter;
    Rv32ShiftCoreCols<T> core;
};

struct ShiftRecord {
    Rv32BaseAluAdapterRecord adapter;
    Rv32ShiftCoreRecord core;
};

__global__ void rv32_shift_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    uint8_t *records,
    size_t num_records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t *lookup_ptr,
    uint32_t lookup_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < num_records) {
        auto rec = reinterpret_cast<ShiftRecord *>(records)[idx];
        auto adapter = Rv32BaseAluAdapter(VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Rv32ShiftCore(
            BitwiseOperationLookup(lookup_ptr, lookup_bits),
            VariableRangeChecker(range_ptr, range_bins)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(ShiftCols, core)), rec.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv32_shift_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t *d_records,
    size_t record_len,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(ShiftRecord) >= record_len);
    assert(width == sizeof(ShiftCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);

    rv32_shift_tracegen<<<grid, block>>>(
        d_trace,
        height,
        width,
        d_records,
        record_len / sizeof(ShiftRecord),
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return cudaGetLastError();
}