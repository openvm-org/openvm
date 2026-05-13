#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w.cuh"
#include "riscv/cores/add_sub.cuh"

using namespace riscv;

using Rv64AddSubWCoreRecord = AddSubCoreRecord<RV64_WORD_NUM_LIMBS>;
using Rv64AddSubWCore = AddSubCore<RV64_WORD_NUM_LIMBS>;
template <typename T> using Rv64AddSubWCoreCols = AddSubCoreCols<T, RV64_WORD_NUM_LIMBS>;

template <typename T> struct Rv64AddSubWCols {
    Rv64BaseAluWAdapterCols<T> adapter;
    Rv64AddSubWCoreCols<T> core;
};

struct Rv64AddSubWRecord {
    Rv64BaseAluWAdapterRecord adapter;
    Rv64AddSubWCoreRecord core;
};

__global__ void add_sub_w_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64AddSubWRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Rv64BaseAluWAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        Rv64AddSubWCore core{BitwiseOperationLookup(d_bitwise_lookup_ptr)};
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64AddSubWCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64AddSubWCols<uint8_t>));
    }
}

extern "C" int _add_sub_w_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64AddSubWRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddSubWCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    add_sub_w_tracegen<<<grid, block, 0, stream>>>(
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
