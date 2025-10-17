#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "rv32im/adapters/alu.cuh"
#include "rv32im/cores/alu.cuh"

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
    DeviceBufferConstView<Rv32BaseAluRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

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
    DeviceBufferConstView<Rv32BaseAluRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(Rv32BaseAluCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    alu_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}