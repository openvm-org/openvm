#include "riscv/cores/store.cuh"

template <typename T> struct StoreDoublewordCoreCols {
    T selector[STORE_DOUBLEWORD_SELECTOR_WIDTH];
    T is_valid;
    T read_data[BLOCK_FE_WIDTH];
    T prev_data[BLOCK_FE_WIDTH];
};

template <typename T> struct Rv64StoreDoublewordCols {
    Rv64StoreAdapterCols<T> adapter;
    StoreDoublewordCoreCols<T> core;
};

struct StoreDoublewordCore {
    __device__ void fill_trace_row(RowSlice row, StoreRecord record) {
        assert(record.local_opcode == STORED);
        assert(record.shift_amount == 0);

        Encoder encoder(
            STORE_DOUBLEWORD_CASES, STORE_SELECTOR_MAX_DEGREE, true, STORE_DOUBLEWORD_SELECTOR_WIDTH
        );
        encoder.write_flag_pt(row, 0);
        row[STORE_DOUBLEWORD_SELECTOR_WIDTH] = 1;
        row.write_array(STORE_DOUBLEWORD_SELECTOR_WIDTH + 1, BLOCK_FE_WIDTH, record.read_data);
        row.write_array(
            STORE_DOUBLEWORD_SELECTOR_WIDTH + 1 + BLOCK_FE_WIDTH,
            BLOCK_FE_WIDTH,
            record.prev_data
        );
    }
};

__global__ void rv64_store_doubleword_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64StoreRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];
        auto adapter = Rv64StoreAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);
        StoreDoublewordCore core;
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64StoreDoublewordCols, core)), record.core
        );
    } else {
        row.fill_zero(0, width);
        COL_WRITE_VALUE(row, Rv64StoreDoublewordCols, adapter.mem_as, 2);
    }
}

extern "C" int _rv64_store_doubleword_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64StoreRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64StoreDoublewordCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_store_doubleword_tracegen_kernel<<<grid, block, 0, stream>>>(
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
