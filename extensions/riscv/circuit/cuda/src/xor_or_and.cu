#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu.cuh"
#include "riscv/cores/xor_or_and.cuh"

using namespace riscv;

// Concrete type aliases for RV64
using Rv64XorOrAndCoreRecord = XorOrAndCoreRecord<RV64_REGISTER_NUM_LIMBS>;
using Rv64XorOrAndCore = XorOrAndCore<RV64_REGISTER_NUM_LIMBS>;
template <typename T> using Rv64XorOrAndCoreCols = XorOrAndCoreCols<T, RV64_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv64XorOrAndCols {
    Rv64BaseAluAdapterCols<T> adapter;
    Rv64XorOrAndCoreCols<T> core;
};

struct Rv64XorOrAndRecord {
    Rv64BaseAluAdapterRecord adapter;
    Rv64XorOrAndCoreRecord core;
};

__global__ void xor_or_and_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64XorOrAndRecord> d_records,
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

        Rv64XorOrAndCore core{BitwiseOperationLookup(d_bitwise_lookup_ptr)};
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64XorOrAndCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64XorOrAndCols<uint8_t>));
    }
}

extern "C" int _xor_or_and_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64XorOrAndRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64XorOrAndCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    xor_or_and_tracegen<<<grid, block, 0, stream>>>(
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
