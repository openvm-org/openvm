#include "launcher.cuh"
#include "native/adapters/native_vectorized_adapter.cuh"
#include "native/field_ext_operations.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

using namespace riscv;
using namespace program;
using namespace native;

template <typename T> struct FieldExtensionCoreCols {
    T x[EXT_DEG];
    T y[EXT_DEG];
    T z[EXT_DEG];

    T is_add;
    T is_sub;
    T is_mul;
    T is_div;
    /// `divisor_inv` is z.inverse() when opcode is FDIV and zero otherwise.
    T divisor_inv[EXT_DEG];
};

template <typename F> struct FieldExtensionCoreRecord {
    F y[EXT_DEG];
    F z[EXT_DEG];
    uint8_t local_opcode;
};

enum class FieldExtensionOpcode {
    FE4ADD,
    FE4SUB,
    BBE4MUL,
    BBE4DIV,
};

struct FieldExtension {
    VariableRangeChecker range_checker;

    __device__ FieldExtension(VariableRangeChecker range_checker) : range_checker(range_checker) {}

    __device__ void fill_trace_row(RowSlice row, FieldExtensionCoreRecord<Fp> const &record) {
        COL_WRITE_ARRAY(row, FieldExtensionCoreCols, y, record.y);
        COL_WRITE_ARRAY(row, FieldExtensionCoreCols, z, record.z);
        FieldExtensionOpcode opcode = static_cast<FieldExtensionOpcode>(record.local_opcode);
        COL_WRITE_VALUE(
            row, FieldExtensionCoreCols, is_add, opcode == FieldExtensionOpcode::FE4ADD
        );
        COL_WRITE_VALUE(
            row, FieldExtensionCoreCols, is_sub, opcode == FieldExtensionOpcode::FE4SUB
        );
        COL_WRITE_VALUE(
            row, FieldExtensionCoreCols, is_mul, opcode == FieldExtensionOpcode::BBE4MUL
        );
        COL_WRITE_VALUE(
            row, FieldExtensionCoreCols, is_div, opcode == FieldExtensionOpcode::BBE4DIV
        );
        FieldExtElement<Fp> y{record.y}, z{record.z};

        switch (opcode) {
        case FieldExtensionOpcode::FE4ADD:
            COL_FILL_ZERO(row, FieldExtensionCoreCols, divisor_inv);
            COL_WRITE_ARRAY(row, FieldExtensionCoreCols, x, FieldExtOperations::add(y, z).el);
            break;
        case FieldExtensionOpcode::FE4SUB:
            COL_FILL_ZERO(row, FieldExtensionCoreCols, divisor_inv);
            COL_WRITE_ARRAY(row, FieldExtensionCoreCols, x, FieldExtOperations::subtract(y, z).el);
            break;
        case FieldExtensionOpcode::BBE4MUL:
            COL_FILL_ZERO(row, FieldExtensionCoreCols, divisor_inv);
            COL_WRITE_ARRAY(row, FieldExtensionCoreCols, x, FieldExtOperations::multiply(y, z).el);
            break;
        case FieldExtensionOpcode::BBE4DIV: {
            auto inv_z = FieldExtOperations::invert(z);
            COL_WRITE_ARRAY(row, FieldExtensionCoreCols, divisor_inv, inv_z.el);
            COL_WRITE_ARRAY(
                row, FieldExtensionCoreCols, x, FieldExtOperations::multiply(y, inv_z).el
            );
            break;
        }
        default:
            assert(false);
        }
    }
};

// [Adapter + Core] columns and record
template <typename T> struct FieldExtensionCols {
    NativeVectorizedAdapterCols<T, EXT_DEG> adapter;
    FieldExtensionCoreCols<T> core;
};

template <typename F> struct FieldExtensionRecord {
    NativeVectorizedAdapterRecord<F, EXT_DEG> adapter;
    FieldExtensionCoreRecord<F> core;
};

__global__ void field_extension_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<FieldExtensionRecord<Fp>> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = NativeVectorizedAdapter<Fp, EXT_DEG>(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core = FieldExtension(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(FieldExtensionCols, core)), record.core);
    } else {
        // Fill with 0s
        row.fill_zero(0, width);
    }
}

extern "C" int _field_extension_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<FieldExtensionRecord<Fp>> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(width == sizeof(FieldExtensionCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    field_extension_tracegen<<<grid, block>>>(
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
