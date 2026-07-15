#include "riscv/cores/load.cuh"
#include "riscv/rvr_compact.cuh"

template <typename T> struct LoadWordCoreCols {
    T selector[LOAD_WORD_SELECTOR_WIDTH];
    T is_valid;
    T read_data[BLOCK_FE_WIDTH];
};

template <typename T> struct Rv64LoadWordCols {
    Rv64LoadAdapterCols<T> adapter;
    LoadWordCoreCols<T> core;
};

struct LoadWordCore {
    __device__ void fill_trace_row(RowSlice row, LoadRecord record, uint8_t shift) {
        uint32_t case_idx = shift >> 2;

        Encoder encoder(LOAD_WORD_CASES, LOAD_SELECTOR_MAX_DEGREE, true, LOAD_WORD_SELECTOR_WIDTH);
        encoder.write_flag_pt(row, case_idx);
        row[LOAD_WORD_SELECTOR_WIDTH] = 1;
        row.write_array(LOAD_WORD_SELECTOR_WIDTH + 1, BLOCK_FE_WIDTH, record.read_data);
    }
};

__global__ void rv64_load_word_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadRecord> records,
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
        LoadWordCore core;
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64LoadWordCols, core)),
            record.core,
            rv64_load_shift_amount(record.adapter)
        );
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_load_word_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64LoadWordCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_word_tracegen<<<grid, block, 0, stream>>>(
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

__global__ void rv64_load_word_tracegen_compact(
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
        Rv64LoadRecord full;
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
        LoadWordCore core;
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64LoadWordCols, core)),
            full.core,
            rv64_load_shift_amount(full.adapter)
        );
    } else {
        row.fill_zero(0, sizeof(Rv64LoadWordCols<uint8_t>));
    }
}

extern "C" int _rv64_load_word_tracegen_compact(
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
    assert(width == sizeof(Rv64LoadWordCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_word_tracegen_compact<<<grid, block, 0, stream>>>(
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
