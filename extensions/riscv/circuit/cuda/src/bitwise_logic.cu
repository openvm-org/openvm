#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_reg.cuh"
#include "riscv/cores/bitwise_logic.cuh"
#include "riscv/rvr_compact.cuh"
#include "riscv/rvr_g2_trace.cuh"

using namespace riscv;

// Concrete type aliases for RV64
using Rv64BitwiseLogicCoreRecord = BitwiseLogicCoreRecord<RV64_REGISTER_NUM_LIMBS>;
using Rv64BitwiseLogicCore = BitwiseLogicCore<RV64_REGISTER_NUM_LIMBS>;
template <typename T> using Rv64BitwiseLogicCoreCols = BitwiseLogicCoreCols<T, RV64_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv64BitwiseLogicCols {
    Rv64BaseAluRegAdapterCols<T> adapter;
    Rv64BitwiseLogicCoreCols<T> core;
};

struct Rv64BitwiseLogicRecord {
    Rv64BaseAluRegAdapterRecord adapter;
    Rv64BitwiseLogicCoreRecord core;
};

__global__ void bitwise_logic_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64BitwiseLogicRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Rv64BaseAluRegAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        Rv64BitwiseLogicCore core{BitwiseOperationLookup(d_bitwise_lookup_ptr)};
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64BitwiseLogicCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64BitwiseLogicCols<uint8_t>));
    }
}

extern "C" int _bitwise_logic_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64BitwiseLogicRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64BitwiseLogicCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    bitwise_logic_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

// M-GPUDEC (G2): compact-wire twin of the kernel above; decodes in registers
// and calls the SAME fill methods.
template <typename RecordView>
__global__ void bitwise_logic_tracegen_compact(
    Fp *d_trace,
    size_t height,
    RecordView d_records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        RvrAlu3Compact const rec_c = d_records[idx];
        RvrOperandEntry const entry = rvr_operand_entry(operand_table, pc_base, rec_c.from_pc);
        Rv64BitwiseLogicRecord full;
        full.adapter = rvr_decode_alu3_bytes(rec_c, entry);
#pragma unroll
        for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
            full.core.b[i] = rvr_u8_limb(rec_c.b, i);
            full.core.c[i] = rvr_u8_limb(rec_c.c, i);
        }
        full.core.local_opcode = entry.local_opcode;

        Rv64BaseAluRegAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, full.adapter);

        Rv64BitwiseLogicCore core{BitwiseOperationLookup(d_bitwise_lookup_ptr)};
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64BitwiseLogicCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(Rv64BitwiseLogicCols<uint8_t>));
    }
}

extern "C" int _bitwise_logic_tracegen_compact(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrAlu3Compact> d_records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
#ifdef OPENVM_RVR_CUDA_G2_ONLY
    return int(cudaErrorNotSupported);
#else
    assert(width == sizeof(Rv64BitwiseLogicCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    bitwise_logic_tracegen_compact<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        operand_table,
        pc_base,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
#endif
}

DEFINE_RVR_G2_TRACEGEN_LAUNCHER(
    _bitwise_logic_tracegen_g2, Rv64BitwiseLogicCols, bitwise_logic_tracegen_compact,
    RvrAlu3Compact, 256, operand_table, pc_base, range_checker,
    size_t(range_checker_num_bins), bitwise_lookup, timestamp_max_bits
)
