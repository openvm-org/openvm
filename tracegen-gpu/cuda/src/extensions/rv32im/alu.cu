#include "adapters/alu.cuh"
#include "constants.h"
#include "cores/alu.cuh"
#include "histogram.cuh"
#include "launcher.cuh"
#include "trace_access.h"

using namespace riscv;

// Concrete type aliases for 32-bit
using Rv32BaseAluCoreRecord = BaseAluCoreRecord<RV32_REGISTER_NUM_LIMBS>;
using Rv32BaseAluCore = BaseAluCore<RV32_REGISTER_NUM_LIMBS>;
template <typename T> using Rv32BaseAluCoreCols = BaseAluCoreCols<T, RV32_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv32BaseAluCols {
    Rv32BaseAluAdapterCols<T> adapter;
    Rv32BaseAluCoreCols<T> core;
};

struct Rv32BaseAluRecord {
    Rv32BaseAluAdapterRecord adapter;
    Rv32BaseAluCoreRecord core;
};

__global__ void alu_tracegen(
    Fp *d_trace,
    size_t height,
    uint8_t *d_records,
    size_t num_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < num_records) {
        auto rec = reinterpret_cast<Rv32BaseAluRecord *>(d_records)[idx];

        Rv32BaseAluAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        Rv32BaseAluCore core(BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv32BaseAluCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv32BaseAluCols<uint8_t>));
    }
}

extern "C" int _alu_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t *d_records,
    size_t record_len,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(Rv32BaseAluRecord) >= record_len);
    assert(width == sizeof(Rv32BaseAluCols<uint8_t>));
    size_t num_records = record_len / sizeof(Rv32BaseAluRecord);
    auto [grid, block] = kernel_launch_params(height);
    alu_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        num_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return cudaGetLastError();
}