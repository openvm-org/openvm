#include "riscv/cores/load_sign_extend.cuh"
#include "riscv/rvr_compact.cuh"

template <typename T> struct Rv64LoadSignExtendWordCols {
    Rv64LoadAdapterCols<T> adapter;
    LoadSignExtendWidthAlignedCoreCols<T, LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH> core;
};

struct LoadSignExtendWordCore {
    VariableRangeChecker range_checker;

    __device__ LoadSignExtendWordCore(VariableRangeChecker range_checker)
        : range_checker(range_checker) {}

    __device__ void fill_trace_row(RowSlice row, LoadSignExtendRecord record, uint8_t shift) {
        uint32_t case_idx = shift >> 2;
        uint16_t sign_cell = record.read_data[(shift >> 1) + 1];
        uint16_t sign_bit = sign_cell & SIGN_U16;
        range_checker.add_count(sign_cell - sign_bit, U16_BITS - 1);

        Encoder encoder(
            LOAD_SIGN_EXTEND_WORD_CASES,
            LOAD_SIGN_EXTEND_SELECTOR_MAX_DEGREE,
            true,
            LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH
        );
        encoder.write_flag_pt(row, case_idx);
        row[LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH] = 1;
        row[LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH + 1] = sign_bit != 0;
        row.write_array(LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH + 2, BLOCK_FE_WIDTH, record.read_data);
    }
};

__global__ void rv64_load_sign_extend_word_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadSignExtendRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
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

        auto core =
            LoadSignExtendWordCore(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64LoadSignExtendWordCols, core)),
            record.core,
            rv64_load_shift_amount(record.adapter)
        );
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_load_sign_extend_word_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadSignExtendRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64LoadSignExtendWordCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_sign_extend_word_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

__global__ void rv64_load_sign_extend_word_tracegen_compact(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrAlu3Compact> records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const rec = records[idx];
        auto const entry = rvr_operand_entry(operand_table, pc_base, rec.from_pc);
        Rv64LoadSignExtendRecord full;
        full.adapter = rvr_decode_alu3_load(rec, entry);
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            full.core.read_data[i] = rvr_u16_limb(rec.c, i);
        }
        auto adapter = Rv64LoadAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, full.adapter);
        auto core =
            LoadSignExtendWordCore(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64LoadSignExtendWordCols, core)),
            full.core,
            rv64_load_shift_amount(full.adapter)
        );
    } else {
        row.fill_zero(0, sizeof(Rv64LoadSignExtendWordCols<uint8_t>));
    }
}

extern "C" int _rv64_load_sign_extend_word_tracegen_compact(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrAlu3Compact> records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    size_t pointer_max_bits,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64LoadSignExtendWordCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_sign_extend_word_tracegen_compact<<<grid, block, 0, stream>>>(
        trace,
        height,
        records,
        operand_table,
        pc_base,
        pointer_max_bits,
        range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
