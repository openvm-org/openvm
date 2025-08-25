#include "launcher.cuh"
#include "native/adapters/branch_native_adapter.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

using namespace riscv;
using namespace program;
using namespace native;

template <typename T> struct NativeBranchEqualCoreCols {
    T a;
    T b;

    T cmp_result;
    T imm;

    T opcode_beq_flag;
    T opcode_bne_flag;

    T diff_inv_marker;
};

template <typename F> struct NativeBranchEqualCoreRecord {
    F a;
    F b;
    F imm;
    bool is_beq;
};

struct NativeBranchEqual {

    __device__ void fill_trace_row(RowSlice row, NativeBranchEqualCoreRecord<Fp> record) {
        COL_WRITE_VALUE(row, NativeBranchEqualCoreCols, a, record.a);
        COL_WRITE_VALUE(row, NativeBranchEqualCoreCols, b, record.b);
        bool cmp_result = (record.a != record.b) ^ record.is_beq;
        COL_WRITE_VALUE(row, NativeBranchEqualCoreCols, cmp_result, cmp_result);
        COL_WRITE_VALUE(row, NativeBranchEqualCoreCols, imm, record.imm);
        COL_WRITE_VALUE(row, NativeBranchEqualCoreCols, opcode_beq_flag, record.is_beq);
        COL_WRITE_VALUE(row, NativeBranchEqualCoreCols, opcode_bne_flag, !record.is_beq);
        Fp diff_inv_marker = (record.a == record.b) ? Fp::zero() : inv(record.a - record.b);
        COL_WRITE_VALUE(row, NativeBranchEqualCoreCols, diff_inv_marker, diff_inv_marker);
    }
};

// [Adapter + Core] columns and record
template <typename T> struct NativeBranchEqualCols {
    BranchNativeAdapterCols<T> adapter;
    NativeBranchEqualCoreCols<T> core;
};

template <typename F> struct NativeBranchEqualRecord {
    BranchNativeAdapterRecord<F> adapter;
    NativeBranchEqualCoreRecord<F> core;
};

__global__ void native_branch_eq_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<NativeBranchEqualRecord<Fp>> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = BranchNativeAdapter<Fp>(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);
        auto core = NativeBranchEqual();
        core.fill_trace_row(row.slice_from(COL_INDEX(NativeBranchEqualCols, core)), record.core);
    } else {
        // Fill with 0s
        row.fill_zero(0, width);
    }
}

extern "C" int _native_branch_eq_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<NativeBranchEqualRecord<Fp>> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(width == sizeof(NativeBranchEqualCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    native_branch_eq_tracegen<<<grid, block>>>(
        d_trace,
        height,
        width,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return cudaGetLastError();
}
