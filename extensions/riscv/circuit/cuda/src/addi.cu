#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm_u16.cuh"
#include "riscv/cores/addi.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// Concrete type aliases for RV64
using Rv64AddICoreRecord = AddICoreRecord<BLOCK_FE_WIDTH>;
using Rv64AddICore = AddICore<BLOCK_FE_WIDTH, U16_BITS, true>;
template <typename T> using Rv64AddICoreCols = AddICoreCols<T, BLOCK_FE_WIDTH>;

template <typename T> struct Rv64AddICols {
    Rv64BaseAluImmU16AdapterCols<T> adapter;
    Rv64AddICoreCols<T> core;
};

struct Rv64AddIRecord {
    Rv64BaseAluImmU16AdapterRecord adapter;
    Rv64AddICoreRecord core;
};

__global__ void addi_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64AddIRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        auto adapter = Rv64BaseAluImmU16Adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        auto core =
            Rv64AddICore(VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64AddICols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64AddICols<uint8_t>));
    }
}

extern "C" int _addi_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64AddIRecord> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddICols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    addi_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_range_checker, range_checker_num_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
