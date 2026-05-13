#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu.cuh"
#include "riscv/cores/add_sub.cuh"

using namespace riscv;

using Rv64AddSubCoreRecord = AddSubCoreRecord<RV64_REGISTER_NUM_LIMBS>;
using Rv64AddSubCore = AddSubCore<RV64_REGISTER_NUM_LIMBS>;
template <typename T> using Rv64AddSubCoreCols = AddSubCoreCols<T, RV64_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv64AddSubCols {
    Rv64BaseAluAdapterCols<T> adapter;
    Rv64AddSubCoreCols<T> core;
};

struct Rv64AddSubRecord {
    Rv64BaseAluAdapterRecord adapter;
    Rv64AddSubCoreRecord core;
};

__global__ void add_sub_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64AddSubRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Rv64BaseAluAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        Rv64AddSubCore core{BitwiseOperationLookup(d_bitwise_lookup_ptr)};
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
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddSubCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    add_sub_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
