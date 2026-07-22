#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm.cuh"
#include "riscv/cores/bitwise_logic_imm.cuh"
#include "riscv/rvr_compact.cuh"
#include "riscv/rvr_g2_trace.cuh"

using namespace riscv;

// XORI/ORI/ANDI use byte limbs and the immediate-only byte ALU adapter.
using Rv64BitwiseLogicImmCoreRecord = BitwiseLogicImmCoreRecord<RV64_REGISTER_NUM_LIMBS>;
using Rv64BitwiseLogicImmCore = BitwiseLogicImmCore<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>;
template <typename T>
using Rv64BitwiseLogicImmCoreCols = BitwiseLogicImmCoreCols<T, RV64_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv64BitwiseLogicImmCols {
    Rv64BaseAluImmAdapterCols<T> adapter;
    Rv64BitwiseLogicImmCoreCols<T> core;
};

struct Rv64BitwiseLogicImmRecord {
    Rv64BaseAluImmAdapterRecord adapter;
    Rv64BitwiseLogicImmCoreRecord core;
};

static_assert(sizeof(Rv64BitwiseLogicImmRecord) == 44);
static_assert(offsetof(Rv64BitwiseLogicImmRecord, core) == 32);

__global__ void bitwise_logic_imm_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64BitwiseLogicImmRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        auto adapter = Rv64BaseAluImmAdapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        auto core = Rv64BitwiseLogicImmCore(BitwiseOperationLookup(d_bitwise_lookup_ptr));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64BitwiseLogicImmCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64BitwiseLogicImmCols<uint8_t>));
    }
}

extern "C" int _bitwise_logic_imm_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64BitwiseLogicImmRecord> d_records,
    uint32_t *d_range_checker,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64BitwiseLogicImmCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    bitwise_logic_imm_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker,
        range_checker_bins,
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

template <typename RecordView>
__global__ void bitwise_logic_imm_tracegen_compact(
    Fp *trace,
    size_t height,
    RecordView records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    uint32_t *range_checker,
    size_t range_bins,
    uint32_t *bitwise_lookup,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        RvrAlu3Compact const rec = records[idx];
        RvrOperandEntry const entry = rvr_operand_entry(operand_table, pc_base, rec.from_pc);
        Rv64BitwiseLogicImmRecord full;
        full.adapter = rvr_decode_alu3_alu_imm_bytes(rec, entry);
#pragma unroll
        for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; ++i)
            full.core.b[i] = rvr_u8_limb(rec.b, i);
        full.core.c_low[0] = uint8_t(entry.c);
        full.core.c_low[1] = uint8_t((entry.c >> 8) & 7u);
        full.core.imm_sign = uint8_t((entry.c >> 11) & 1u);
        full.core.local_opcode = entry.local_opcode;
        Rv64BaseAluImmAdapter(
            VariableRangeChecker(range_checker, range_bins), timestamp_max_bits
        ).fill_trace_row(row, full.adapter);
        Rv64BitwiseLogicImmCore(BitwiseOperationLookup(bitwise_lookup))
            .fill_trace_row(row.slice_from(COL_INDEX(Rv64BitwiseLogicImmCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(Rv64BitwiseLogicImmCols<uint8_t>));
    }
}

extern "C" int _bitwise_logic_imm_tracegen_compact(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrAlu3Compact> records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    uint32_t *range_checker,
    size_t range_bins,
    uint32_t *bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64BitwiseLogicImmCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    bitwise_logic_imm_tracegen_compact<<<grid, block, 0, stream>>>(
        trace, height, records, operand_table, pc_base, range_checker, range_bins,
        bitwise_lookup, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

DEFINE_RVR_G2_TRACEGEN_LAUNCHER(
    _bitwise_logic_imm_tracegen_g2,
    Rv64BitwiseLogicImmCols,
    bitwise_logic_imm_tracegen_compact,
    RvrAlu3Compact,
    512,
    operand_table,
    pc_base,
    range_checker,
    range_checker_num_bins,
    bitwise_lookup,
    timestamp_max_bits
)
