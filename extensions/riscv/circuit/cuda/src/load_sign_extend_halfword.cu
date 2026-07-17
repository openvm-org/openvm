#include "riscv/cores/load_sign_extend.cuh"
#include "riscv/rvr_compact.cuh"

using LoadSignExtendHalfwordCore = LoadSignExtendWidthCore<HALFWORD_ACCESS_WIDTH>;

template <typename T> struct Rv64LoadSignExtendHalfwordCols {
    Rv64LoadMultiByteAdapterCols<T> adapter;
    LoadSignExtendWidthCoreCols<T, HALFWORD_ACCESS_WIDTH> core;
};

__global__ void rv64_load_sign_extend_halfword_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadSignExtendRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];
        auto adapter = Rv64LoadAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);
        auto core = LoadSignExtendHalfwordCore(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            BitwiseOperationLookup(bitwise_lookup_ptr)
        );
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64LoadSignExtendHalfwordCols, core)),
            record.core,
            rv64_load_shift_amount(record.adapter)
        );
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_load_sign_extend_halfword_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadSignExtendRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64LoadSignExtendHalfwordCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_sign_extend_halfword_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

__global__ void rv64_load_sign_extend_halfword_tracegen_compact(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrAlu3Compact> records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const rec = records[idx];
        auto const entry = rvr_operand_entry(operand_table, pc_base, rec.from_pc);
        Rv64LoadSignExtendRecord full;
        full.adapter = rvr_decode_alu3_load_multi(rec, entry);
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            full.core.read_data[0][i] = rvr_u16_limb(rec.c, i);
            full.core.read_data[1][i] = 0;
        }
        auto adapter = Rv64LoadAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, full.adapter);
        auto core = LoadSignExtendHalfwordCore(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            BitwiseOperationLookup(bitwise_lookup_ptr)
        );
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64LoadSignExtendHalfwordCols, core)),
            full.core,
            rv64_load_shift_amount(full.adapter)
        );
    } else {
        row.fill_zero(0, sizeof(Rv64LoadSignExtendHalfwordCols<uint8_t>));
    }
}

extern "C" int _rv64_load_sign_extend_halfword_tracegen_compact(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrAlu3Compact> records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    size_t pointer_max_bits,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64LoadSignExtendHalfwordCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_sign_extend_halfword_tracegen_compact<<<grid, block, 0, stream>>>(
        trace,
        height,
        records,
        operand_table,
        pc_base,
        pointer_max_bits,
        range_checker,
        range_checker_num_bins,
        bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
