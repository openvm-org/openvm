#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_reg_u16.cuh"
#include "riscv/cores/shift_right_arithmetic.cuh"
#include "riscv/rvr_compact.cuh"
#include "riscv/rvr_g2_trace.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// SRAW uses the u16 shift-right-arithmetic core (RV64_WORD_U16_LIMBS limbs of 16 bits) over the low
// 32-bit word and the u16 W adapter.
using Rv64ShiftWRightArithmeticCoreRecord =
    ShiftRightArithmeticCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>;
using Rv64ShiftWRightArithmeticCore = ShiftRightArithmeticCore<RV64_WORD_U16_LIMBS, U16_BITS>;
template <typename T>
using Rv64ShiftWRightArithmeticCoreCols =
    ShiftRightArithmeticCoreCols<T, RV64_WORD_U16_LIMBS, U16_BITS>;

template <typename T> struct ShiftWRightArithmeticCols {
    Rv64BaseAluWRegU16AdapterCols<T> adapter;
    Rv64ShiftWRightArithmeticCoreCols<T> core;
};

struct ShiftWRightArithmeticRecord {
    Rv64BaseAluWRegU16AdapterRecord adapter;
    Rv64ShiftWRightArithmeticCoreRecord core;
};

__global__ void rv64_shift_w_right_arithmetic_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<ShiftWRightArithmeticRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluWRegU16Adapter(
            VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Rv64ShiftWRightArithmeticCore(VariableRangeChecker(range_ptr, range_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(ShiftWRightArithmeticCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(ShiftWRightArithmeticCols<uint8_t>));
    }
}

extern "C" int _rv64_shift_w_right_arithmetic_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftWRightArithmeticRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(ShiftWRightArithmeticCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_shift_w_right_arithmetic_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_range_checker, range_checker_num_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

template <typename RecordView>
__global__ void rv64_shift_w_right_arithmetic_tracegen_compact(
    Fp *trace,
    size_t height,
    RecordView records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        RvrAlu3Compact const rec = records[idx];
        RvrOperandEntry const entry = rvr_operand_entry(operand_table, pc_base, rec.from_pc);
        uint32_t const result_word =
            (uint32_t)((int32_t)rec.b[0] >> (rec.c[0] & 31u));
        ShiftWRightArithmeticRecord full;
        full.adapter = rvr_decode_alu3_alu_w_u16(rec, entry, result_word);
#pragma unroll
        for (size_t i = 0; i < RV64_WORD_U16_LIMBS; i++) {
            full.core.b[i] = rvr_u16_limb(rec.b, i);
            full.core.c[i] = rvr_u16_limb(rec.c, i);
        }

        auto adapter =
            Rv64BaseAluWU16Adapter(VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, full.adapter);
        auto core = Rv64ShiftWRightArithmeticCore(VariableRangeChecker(range_ptr, range_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(ShiftWRightArithmeticCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(ShiftWRightArithmeticCols<uint8_t>));
    }
}

extern "C" int _rv64_shift_w_right_arithmetic_tracegen_compact(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrAlu3Compact> d_records,
    RvrOperandEntry const *d_operand_table,
    uint32_t pc_base,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(ShiftWRightArithmeticCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_shift_w_right_arithmetic_tracegen_compact<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_operand_table,
        pc_base,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

DEFINE_RVR_G2_TRACEGEN_LAUNCHER(
    _rv64_shift_w_right_arithmetic_tracegen_g2, ShiftWRightArithmeticCols,
    rv64_shift_w_right_arithmetic_tracegen_compact, RvrAlu3Compact, 512, operand_table,
    pc_base, range_checker, range_checker_num_bins, timestamp_max_bits
)
