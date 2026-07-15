#include "riscv/cores/load.cuh"

template <typename T> struct LoadDoublewordCoreCols {
    T selector[LOAD_DOUBLEWORD_SELECTOR_WIDTH];
    T is_valid;
    T read_data[BLOCK_FE_WIDTH];
};

template <typename T> struct Rv64LoadDoublewordCols {
    Rv64LoadAdapterCols<T> adapter;
    LoadDoublewordCoreCols<T> core;
};

struct LoadDoublewordCore {
    __device__ void fill_trace_row(RowSlice row, LoadRecord record, uint8_t shift) {
        assert(shift == 0);

        Encoder encoder(
            LOAD_DOUBLEWORD_CASES, LOAD_SELECTOR_MAX_DEGREE, true, LOAD_DOUBLEWORD_SELECTOR_WIDTH
        );
        encoder.write_flag_pt(row, 0);
        row[LOAD_DOUBLEWORD_SELECTOR_WIDTH] = 1;
        row.write_array(LOAD_DOUBLEWORD_SELECTOR_WIDTH + 1, BLOCK_FE_WIDTH, record.read_data);
    }
};

__global__ void rv64_load_doubleword_tracegen(
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
        LoadDoublewordCore core;
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64LoadDoublewordCols, core)),
            record.core,
            rv64_load_shift_amount(record.adapter)
        );
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_load_doubleword_tracegen(
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
    assert(width == sizeof(Rv64LoadDoublewordCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_doubleword_tracegen<<<grid, block, 0, stream>>>(
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
