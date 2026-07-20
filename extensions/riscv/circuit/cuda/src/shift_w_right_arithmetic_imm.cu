#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_imm_u16.cuh"
#include "riscv/cores/shift_right_arithmetic_imm.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

using Rv64ShiftWRightArithmeticImmCore = ShiftRightArithmeticImmCore<RV64_WORD_U16_LIMBS, U16_BITS>;
using Rv64ShiftWRightArithmeticImmCoreRecord =
    ShiftRightArithmeticImmCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>;
template <typename T>
using Rv64ShiftWRightArithmeticImmCoreCols =
    ShiftRightArithmeticImmCoreCols<T, RV64_WORD_U16_LIMBS, U16_BITS>;

template <typename T> struct Rv64ShiftWRightArithmeticImmCols {
    Rv64BaseAluWImmU16AdapterCols<T> adapter;
    Rv64ShiftWRightArithmeticImmCoreCols<T> core;
};

struct Rv64ShiftWRightArithmeticImmRecord {
    Rv64BaseAluWImmU16AdapterRecord adapter;
    Rv64ShiftWRightArithmeticImmCoreRecord core;
};

static_assert(sizeof(Rv64ShiftWRightArithmeticImmRecord) == 48);
static_assert(offsetof(Rv64ShiftWRightArithmeticImmRecord, core) == 40);

__global__ void shift_w_right_arithmetic_imm_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64ShiftWRightArithmeticImmRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluWImmU16Adapter(
            VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Rv64ShiftWRightArithmeticImmCore(VariableRangeChecker(range_ptr, range_bins));
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64ShiftWRightArithmeticImmCols, core)), rec.core
        );
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _shift_w_right_arithmetic_imm_tracegen(
    Fp *__restrict__ trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64ShiftWRightArithmeticImmRecord> records,
    uint32_t *__restrict__ range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64ShiftWRightArithmeticImmCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    shift_w_right_arithmetic_imm_tracegen<<<grid, block, 0, stream>>>(
        trace, height, width, records, range_ptr, range_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
