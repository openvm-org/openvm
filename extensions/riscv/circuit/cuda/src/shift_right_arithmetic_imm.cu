#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm_u16.cuh"
#include "riscv/cores/shift_right_arithmetic_imm.cuh"
#include "riscv/rvr_compact.cuh"
#include "riscv/rvr_g2_trace.cuh"
#include "system/memory/params.cuh"

using namespace riscv;
using namespace program;

// SRAI uses u16 limbs (4 limbs of 16 bits) and the immediate-only u16 ALU adapter.
using Rv64ShiftRightArithmeticImmCoreRecord =
    ShiftRightArithmeticImmCoreRecord<BLOCK_FE_WIDTH, U16_BITS>;
using Rv64ShiftRightArithmeticImmCore =
    ShiftRightArithmeticImmCore<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64ShiftRightArithmeticImmCoreCols =
    ShiftRightArithmeticImmCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct ShiftRightArithmeticImmCols {
    Rv64BaseAluImmU16AdapterCols<T> adapter;
    Rv64ShiftRightArithmeticImmCoreCols<T> core;
};

struct ShiftRightArithmeticImmRecord {
    Rv64BaseAluImmU16AdapterRecord adapter;
    Rv64ShiftRightArithmeticImmCoreRecord core;
};

static_assert(sizeof(ShiftRightArithmeticImmRecord) == 44);
static_assert(offsetof(ShiftRightArithmeticImmRecord, core) == 32);

__global__ void shift_right_arithmetic_imm_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftRightArithmeticImmRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluImmU16Adapter(
            VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Rv64ShiftRightArithmeticImmCore(
            VariableRangeChecker(range_ptr, range_bins)
        );
        core.fill_trace_row(
            row.slice_from(COL_INDEX(ShiftRightArithmeticImmCols, core)), rec.core
        );
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _shift_right_arithmetic_imm_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftRightArithmeticImmRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(ShiftRightArithmeticImmCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    shift_right_arithmetic_imm_tracegen<<<grid, block, 0, stream>>>(
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

template <typename RecordView>
__global__ void shift_right_arithmetic_imm_tracegen_compact(
    Fp *trace, size_t height, RecordView records, RvrOperandEntry const *operand_table,
    uint32_t pc_base, uint32_t *range_checker, uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        RvrAlu3Compact const rec = records[idx];
        RvrOperandEntry const entry = rvr_operand_entry(operand_table, pc_base, rec.from_pc);
        ShiftRightArithmeticImmRecord full;
        full.adapter = rvr_decode_alu3_alu_imm_u16(rec, entry);
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; ++i)
            full.core.b[i] = rvr_u16_limb(rec.b, i);
        full.core.shamt = uint8_t(entry.c);
        Rv64BaseAluImmU16Adapter(
            VariableRangeChecker(range_checker, range_bins), timestamp_max_bits
        ).fill_trace_row(row, full.adapter);
        Rv64ShiftRightArithmeticImmCore(VariableRangeChecker(range_checker, range_bins))
            .fill_trace_row(
                row.slice_from(COL_INDEX(ShiftRightArithmeticImmCols, core)), full.core
            );
    } else {
        row.fill_zero(0, sizeof(ShiftRightArithmeticImmCols<uint8_t>));
    }
}

extern "C" int _shift_right_arithmetic_imm_tracegen_compact(
    Fp *trace, size_t height, size_t width, DeviceBufferConstView<RvrAlu3Compact> records,
    RvrOperandEntry const *operand_table, uint32_t pc_base, uint32_t *range_checker,
    uint32_t range_bins, uint32_t timestamp_max_bits, cudaStream_t stream
) {
    assert(width == sizeof(ShiftRightArithmeticImmCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    shift_right_arithmetic_imm_tracegen_compact<<<grid, block, 0, stream>>>(
        trace, height, records, operand_table, pc_base, range_checker, range_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

DEFINE_RVR_G2_TRACEGEN_LAUNCHER(
    _shift_right_arithmetic_imm_tracegen_g2, ShiftRightArithmeticImmCols,
    shift_right_arithmetic_imm_tracegen_compact, RvrAlu3Compact, 512, operand_table,
    pc_base, range_checker, range_checker_num_bins, timestamp_max_bits
)
