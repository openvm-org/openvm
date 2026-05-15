#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu.cuh"
#include "riscv/cores/and.cuh"

using namespace riscv;

using Rv64AndCoreRecord = AndCoreRecord<RV64_REGISTER_NUM_LIMBS>;
using Rv64AndCore = AndCore<RV64_REGISTER_NUM_LIMBS>;
template <typename T> using Rv64AndCoreCols = AndCoreCols<T, RV64_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv64AndCols {
    Rv64BaseAluAdapterCols<T> adapter;
    Rv64AndCoreCols<T> core;
};

struct Rv64AndRecord {
    Rv64BaseAluAdapterRecord adapter;
    Rv64AndCoreRecord core;
};

__global__ void and_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64AndRecord> d_records,
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

        Rv64AndCore core{BitwiseOperationLookup(d_bitwise_lookup_ptr)};
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64AndCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64AndCols<uint8_t>));
    }
}

extern "C" int _and_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64AndRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AndCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    and_tracegen<<<grid, block, 0, stream>>>(
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
