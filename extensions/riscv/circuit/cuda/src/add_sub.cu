#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_u16.cuh"
#include "riscv/cores/add_sub.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// Concrete type aliases for RV64
using Rv64AddSubCoreRecord = AddSubCoreRecord<BLOCK_FE_WIDTH>;
using Rv64AddSubCore = AddSubCore<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T> using Rv64AddSubCoreCols = AddSubCoreCols<T, BLOCK_FE_WIDTH>;

template <typename T> struct Rv64AddSubCols {
    Rv64BaseAluU16AdapterCols<T> adapter;
    Rv64AddSubCoreCols<T> core;
};

struct Rv64AddSubRecord {
    Rv64BaseAluU16AdapterRecord adapter;
    Rv64AddSubCoreRecord core;
};

__global__ void add_sub_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64AddSubRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        auto adapter = Rv64BaseAluU16Adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        auto core =
            Rv64AddSubCore(VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64AddSubCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64AddSubCols<uint8_t>));
    }
}

extern "C" int _add_sub_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64AddSubRecord> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddSubCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    add_sub_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_range_checker, range_checker_num_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
