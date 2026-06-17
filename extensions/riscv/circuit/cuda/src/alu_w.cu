#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_u16.cuh"
#include "riscv/cores/add_sub.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// Concrete type aliases for the 32-bit word variant on RV64. The low word is two u16 limbs and
// reuses the add_sub core; the adapter rebuilds the sign-extended 64-bit register write.
using Rv64BaseAluWCoreRecord = AddSubCoreRecord<RV64_WORD_U16_LIMBS>;
using Rv64BaseAluWCore = AddSubCore<RV64_WORD_U16_LIMBS, U16_BITS>;
template <typename T> using Rv64BaseAluWCoreCols = AddSubCoreCols<T, RV64_WORD_U16_LIMBS>;

template <typename T> struct Rv64BaseAluWCols {
    Rv64BaseAluWU16AdapterCols<T> adapter;
    Rv64BaseAluWCoreCols<T> core;
};

struct Rv64BaseAluWRecord {
    Rv64BaseAluWU16AdapterRecord adapter;
    Rv64BaseAluWCoreRecord core;
};

__global__ void rv64_alu_w_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64BaseAluWRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Rv64BaseAluWU16Adapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        Rv64BaseAluWCore core(VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64BaseAluWCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64BaseAluWCols<uint8_t>));
    }
}

extern "C" int _rv64_alu_w_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64BaseAluWRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64BaseAluWCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_alu_w_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_range_checker_ptr, range_checker_num_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
