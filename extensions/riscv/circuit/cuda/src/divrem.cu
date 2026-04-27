#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/mul.cuh"
#include "riscv/cores/divrem.cuh"

using namespace riscv;

template <typename T> struct Rv64DivRemCols {
    Rv64MultAdapterCols<T> adapter;
    DivRemCoreCols<T, RV64_REGISTER_NUM_LIMBS> core;
};

struct Rv64DivRemRecord {
    Rv64MultAdapterRecord adapter;
    DivRemCoreRecords<RV64_REGISTER_NUM_LIMBS> core;
};

__global__ void rv64_div_rem_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64DivRemRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_bits,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t bitwise_lookup_bits,
    uint32_t *d_range_tuple_checker_ptr,
    uint2 range_tuple_checker_sizes,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);

    if (idx < d_records.len()) {
        auto const &record = d_records[idx];

        Rv64MultAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bits), timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        DivRemCore<RV64_REGISTER_NUM_LIMBS> core(
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_lookup_bits),
            RangeTupleChecker<2>(
                d_range_tuple_checker_ptr,
                (uint32_t[2]){range_tuple_checker_sizes.x, range_tuple_checker_sizes.y}
            )
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64DivRemCols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(Rv64DivRemCols<uint8_t>));
    }
}

extern "C" int _rv64_div_rem_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64DivRemRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t *d_range_tuple_checker_ptr,
    uint2 range_tuple_checker_sizes,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64DivRemCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_div_rem_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_num_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        d_range_tuple_checker_ptr,
        range_tuple_checker_sizes,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
