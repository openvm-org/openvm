#include "riscv/cores/store.cuh"

template <typename T> struct StoreHalfwordCoreCols {
    T selector[STORE_HALFWORD_SELECTOR_WIDTH];
    T is_valid;
    T read_data[BLOCK_FE_WIDTH];
    T prev_data[BLOCK_FE_WIDTH];
};

template <typename T> struct Rv64StoreHalfwordCols {
    Rv64StoreAdapterCols<T> adapter;
    StoreHalfwordCoreCols<T> core;
};

struct StoreHalfwordCore {
    __device__ void fill_trace_row(RowSlice row, StoreRecord record, uint8_t shift) {
        uint32_t case_idx = shift >> 1;

        Encoder encoder(
            STORE_HALFWORD_CASES, STORE_SELECTOR_MAX_DEGREE, true, STORE_HALFWORD_SELECTOR_WIDTH
        );
        encoder.write_flag_pt(row, case_idx);
        row[STORE_HALFWORD_SELECTOR_WIDTH] = 1;
        row.write_array(STORE_HALFWORD_SELECTOR_WIDTH + 1, BLOCK_FE_WIDTH, record.read_data);
        row.write_array(
            STORE_HALFWORD_SELECTOR_WIDTH + 1 + BLOCK_FE_WIDTH,
            BLOCK_FE_WIDTH,
            record.prev_data
        );
    }
};

__global__ void rv64_store_halfword_tracegen(
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
        StoreHalfwordCore core;
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64StoreHalfwordCols, core)),
            record.core,
            rv64_store_shift_amount(record.adapter)
        );
    } else {
        row.fill_zero(0, width);
        COL_WRITE_VALUE(row, Rv64StoreHalfwordCols, adapter.mem_as, 2);
    }
}

extern "C" int _rv64_store_halfword_tracegen(
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
    assert(width == sizeof(Rv64StoreHalfwordCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_store_halfword_tracegen<<<grid, block, 0, stream>>>(
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
