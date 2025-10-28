#include "launcher.cuh"
#include "native/adapters/convert_adapter.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"

using namespace riscv;
using namespace program;
using namespace native;

const uint32_t FINAL_LIMB_BITS = 6;

template <typename T> struct CastFCoreCols {
    T in_val;
    T out_val[RV32_REGISTER_NUM_LIMBS];
    T is_valid;
};

struct CastFCoreRecord {
    uint32_t val;
};

struct CastF {
    VariableRangeChecker range_checker;

    __device__ CastF(VariableRangeChecker range_checker) : range_checker(range_checker) {}

    __device__ void fill_trace_row(RowSlice row, CastFCoreRecord record) {
        uint8_t *out_val = reinterpret_cast<uint8_t *>(&record.val);
#pragma unroll
        for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS - 1; i++) {
            range_checker.add_count(out_val[i], RV32_CELL_BITS);
        }
        range_checker.add_count(out_val[RV32_REGISTER_NUM_LIMBS - 1], FINAL_LIMB_BITS);
        COL_WRITE_VALUE(row, CastFCoreCols, in_val, record.val);
        COL_WRITE_ARRAY(row, CastFCoreCols, out_val, out_val);
        COL_WRITE_VALUE(row, CastFCoreCols, is_valid, 1);
    }
};

// [Adapter + Core] columns and record
template <typename T> struct CastFCols {
    ConvertAdapterCols<T, RV32_REGISTER_NUM_LIMBS> adapter;
    CastFCoreCols<T> core;
};

template <typename F> struct CastFRecord {
    ConvertAdapterRecord<F, RV32_REGISTER_NUM_LIMBS> adapter;
    CastFCoreRecord core;
};

__global__ void castf_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<CastFRecord<Fp>> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = ConvertAdapter<Fp, RV32_REGISTER_NUM_LIMBS>(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core = CastF(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(CastFCols, core)), record.core);
    } else {
        // Fill with 0s
        row.fill_zero(0, width);
    }
}

extern "C" int _castf_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<CastFRecord<Fp>> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(width == sizeof(CastFCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    castf_tracegen<<<grid, block>>>(
        d_trace,
        height,
        width,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
