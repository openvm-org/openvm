#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_imm_u16.cuh"
#include "riscv/cores/addi.cuh"
#include "riscv/rvr_compact.cuh"
#include "riscv/rvr_g2_trace.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

using Rv64AddIWCoreRecord = AddICoreRecord<RV64_WORD_U16_LIMBS>;
using Rv64AddIWCore = AddICore<RV64_WORD_U16_LIMBS, U16_BITS, false>;
template <typename T> using Rv64AddIWCoreCols = AddICoreCols<T, RV64_WORD_U16_LIMBS>;

template <typename T> struct Rv64AddIWCols {
    Rv64BaseAluWImmU16AdapterCols<T> adapter;
    Rv64AddIWCoreCols<T> core;
};

struct Rv64AddIWRecord {
    Rv64BaseAluWImmU16AdapterRecord adapter;
    Rv64AddIWCoreRecord core;
};

static_assert(sizeof(Rv64AddIWRecord) == 48);
static_assert(offsetof(Rv64AddIWRecord, core) == 40);

__global__ void addi_w_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64AddIWRecord> records,
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
        auto core = Rv64AddIWCore(VariableRangeChecker(range_ptr, range_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64AddIWCols, core)), rec.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _addi_w_tracegen(
    Fp *__restrict__ trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64AddIWRecord> records,
    uint32_t *__restrict__ range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddIWCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    addi_w_tracegen<<<grid, block, 0, stream>>>(
        trace, height, width, records, range_ptr, range_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

template <typename RecordView>
__global__ void addi_w_tracegen_compact(
    Fp *trace, size_t height, RecordView records, RvrOperandEntry const *operand_table,
    uint32_t pc_base, uint32_t *range_checker, uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        RvrAlu3Compact const rec = records[idx];
        RvrOperandEntry const entry = rvr_operand_entry(operand_table, pc_base, rec.from_pc);
        uint32_t source = rec.b[0];
        uint32_t immediate = uint32_t(int32_t(entry.c << 20) >> 20);
        Rv64AddIWRecord full;
        full.adapter = rvr_decode_alu3_alu_w_imm_u16(rec, entry, source + immediate);
#pragma unroll
        for (size_t i = 0; i < RV64_WORD_U16_LIMBS; ++i)
            full.core.rs1[i] = rvr_u16_limb(rec.b, i);
        full.core.imm_low11 = uint16_t(entry.c & 0x7ffu);
        full.core.imm_sign = uint16_t((entry.c >> 11) & 1u);
        Rv64BaseAluWImmU16Adapter(
            VariableRangeChecker(range_checker, range_bins), timestamp_max_bits
        ).fill_trace_row(row, full.adapter);
        Rv64AddIWCore(VariableRangeChecker(range_checker, range_bins))
            .fill_trace_row(row.slice_from(COL_INDEX(Rv64AddIWCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(Rv64AddIWCols<uint8_t>));
    }
}

extern "C" int _addi_w_tracegen_compact(
    Fp *trace, size_t height, size_t width, DeviceBufferConstView<RvrAlu3Compact> records,
    RvrOperandEntry const *operand_table, uint32_t pc_base, uint32_t *range_checker,
    uint32_t range_bins, uint32_t timestamp_max_bits, cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddIWCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    addi_w_tracegen_compact<<<grid, block, 0, stream>>>(
        trace, height, records, operand_table, pc_base, range_checker, range_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

DEFINE_RVR_G2_TRACEGEN_LAUNCHER(
    _addi_w_tracegen_g2, Rv64AddIWCols, addi_w_tracegen_compact, RvrAlu3Compact, 512,
    operand_table, pc_base, range_checker, range_checker_num_bins, timestamp_max_bits
)
