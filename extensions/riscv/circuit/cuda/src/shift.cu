#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu.cuh"
#include "riscv/cores/shift.cuh"

using namespace riscv;
using namespace program;

// Concrete type aliases for 64-bit
using Rv64ShiftSplitRecord = ShiftSplitRecord<RV64_REGISTER_NUM_LIMBS>;
using Rv64SllCore = LogicalShiftCore<RV64_REGISTER_NUM_LIMBS, true>;
using Rv64SrlCore = LogicalShiftCore<RV64_REGISTER_NUM_LIMBS, false>;
using Rv64SraCore = ArithShiftCore<RV64_REGISTER_NUM_LIMBS>;

// Row layouts: SLL and SRL share `ShiftCols`; SRA has the extra `b_sign` column.
template <typename T> struct LogicalShiftRowCols {
    Rv64BaseAluAdapterCols<T> adapter;
    ShiftCols<T, RV64_REGISTER_NUM_LIMBS> core;
};
template <typename T> struct ArithShiftRowCols {
    Rv64BaseAluAdapterCols<T> adapter;
    ShiftSraCols<T, RV64_REGISTER_NUM_LIMBS> core;
};

struct ShiftSplitRecord_ {
    Rv64BaseAluAdapterRecord adapter;
    Rv64ShiftSplitRecord core;
};

template <typename Core>
__device__ void shift_split_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftSplitRecord_> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t *lookup_ptr,
    uint32_t timestamp_max_bits,
    size_t core_offset
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluAdapter(
            VariableRangeChecker(range_ptr, range_bins),
            BitwiseOperationLookup(lookup_ptr),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Core(
            BitwiseOperationLookup(lookup_ptr),
            VariableRangeChecker(range_ptr, range_bins)
        );
        core.fill_trace_row(row.slice_from(core_offset), rec.core);
    } else {
        row.fill_zero(0, width);
    }
}

__global__ void rv64_sll_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftSplitRecord_> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t *lookup_ptr,
    uint32_t timestamp_max_bits
) {
    shift_split_tracegen<Rv64SllCore>(
        trace, height, width, records, range_ptr, range_bins, lookup_ptr, timestamp_max_bits,
        COL_INDEX(LogicalShiftRowCols, core)
    );
}

__global__ void rv64_srl_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftSplitRecord_> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t *lookup_ptr,
    uint32_t timestamp_max_bits
) {
    shift_split_tracegen<Rv64SrlCore>(
        trace, height, width, records, range_ptr, range_bins, lookup_ptr, timestamp_max_bits,
        COL_INDEX(LogicalShiftRowCols, core)
    );
}

__global__ void rv64_sra_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftSplitRecord_> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t *lookup_ptr,
    uint32_t timestamp_max_bits
) {
    shift_split_tracegen<Rv64SraCore>(
        trace, height, width, records, range_ptr, range_bins, lookup_ptr, timestamp_max_bits,
        COL_INDEX(ArithShiftRowCols, core)
    );
}

#define DEFINE_SHIFT_LAUNCHER(NAME, KERNEL, ROW_COLS)                                              \
    extern "C" int NAME(                                                                           \
        Fp *__restrict__ d_trace,                                                                  \
        size_t height,                                                                             \
        size_t width,                                                                              \
        DeviceBufferConstView<ShiftSplitRecord_> d_records,                                        \
        uint32_t *__restrict__ d_range_checker,                                                    \
        uint32_t range_checker_num_bins,                                                           \
        uint32_t *__restrict__ d_bitwise_lookup,                                                   \
        uint32_t timestamp_max_bits,                                                               \
        cudaStream_t stream                                                                        \
    ) {                                                                                            \
        assert(width == sizeof(ROW_COLS<uint8_t>));                                                \
        auto [grid, block] = kernel_launch_params(height, 512);                                    \
        KERNEL<<<grid, block, 0, stream>>>(                                                        \
            d_trace,                                                                               \
            height,                                                                                \
            width,                                                                                 \
            d_records,                                                                             \
            d_range_checker,                                                                       \
            range_checker_num_bins,                                                                \
            d_bitwise_lookup,                                                                      \
            timestamp_max_bits                                                                     \
        );                                                                                         \
        return CHECK_KERNEL();                                                                     \
    }

DEFINE_SHIFT_LAUNCHER(_rv64_sll_tracegen, rv64_sll_tracegen, LogicalShiftRowCols)
DEFINE_SHIFT_LAUNCHER(_rv64_srl_tracegen, rv64_srl_tracegen, LogicalShiftRowCols)
DEFINE_SHIFT_LAUNCHER(_rv64_sra_tracegen, rv64_sra_tracegen, ArithShiftRowCols)
