#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_reg.cuh"
#include "riscv/cores/bitwise_logic.cuh"

using namespace riscv;

// Concrete type aliases for RV64
using Rv64BitwiseLogicCoreRecord = BitwiseLogicCoreRecord<RV64_REGISTER_NUM_LIMBS>;
using Rv64BitwiseLogicCore = BitwiseLogicCore<RV64_REGISTER_NUM_LIMBS>;
template <typename T> using Rv64BitwiseLogicCoreCols = BitwiseLogicCoreCols<T, RV64_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv64BitwiseLogicCols {
    Rv64BaseAluRegAdapterCols<T> adapter;
    Rv64BitwiseLogicCoreCols<T> core;
};

struct Rv64BitwiseLogicRecord {
    Rv64BaseAluRegAdapterRecord adapter;
    Rv64BitwiseLogicCoreRecord core;
};

__global__ void bitwise_logic_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64BitwiseLogicRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Rv64BaseAluRegAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        Rv64BitwiseLogicCore core{BitwiseOperationLookup(d_bitwise_lookup_ptr)};
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64BitwiseLogicCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64BitwiseLogicCols<uint8_t>));
    }
}

extern "C" int _bitwise_logic_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64BitwiseLogicRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64BitwiseLogicCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    bitwise_logic_tracegen<<<grid, block, 0, stream>>>(
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
