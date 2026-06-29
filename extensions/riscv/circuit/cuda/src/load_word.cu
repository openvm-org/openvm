#include "riscv/cores/load.cuh"

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
    __device__ void fill_trace_row(RowSlice row, LoadRecord record) {
        assert(record.local_opcode == LOADWU);
        uint8_t shift = record.shift_amount;
        uint32_t case_idx = shift >> 2;

        Encoder encoder(LOAD_WORD_CASES, LOAD_SELECTOR_MAX_DEGREE, true, LOAD_WORD_SELECTOR_WIDTH);
        encoder.write_flag_pt(row, case_idx);
        row[LOAD_WORD_SELECTOR_WIDTH] = 1;
        row.write_array(LOAD_WORD_SELECTOR_WIDTH + 1, BLOCK_FE_WIDTH, record.read_data);
    }
};

__global__ void rv64_load_word_tracegen_kernel(
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
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64LoadWordCols, core)), record.core);
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
    rv64_load_word_tracegen_kernel<<<grid, block, 0, stream>>>(
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
