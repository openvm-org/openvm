#include "riscv/cores/store.cuh"

// Doubleword store with the column-reduced layout (see the Rust `StoreDoublewordCoreAir`): the
// whole 8-byte value is the source register, so `rs2` is committed only as `value_bytes` and
// reconstructed for the register read, dropping the `read_data` cells.
static constexpr size_t STORE_DOUBLEWORD_WIDTH_CELLS = 4;

template <typename T> struct StoreDoublewordCoreCols {
    T selector[STORE_DOUBLEWORD_SELECTOR_WIDTH];
    T prev_data[2][BLOCK_FE_WIDTH];
    T value_bytes[STORE_DOUBLEWORD_WIDTH_CELLS][2];
    T prev_bound_bytes[2];
};

template <typename T> struct Rv64StoreDoublewordCols {
    Rv64StoreAdapterCols<T> adapter;
    StoreDoublewordCoreCols<T> core;
};

struct StoreDoublewordCore {
    using Cols = StoreDoublewordCoreCols<uint8_t>;

    BitwiseOperationLookup bitwise_lookup;

    __device__ StoreDoublewordCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, StoreRecord record, uint8_t shift) {
        Encoder encoder(
            STORE_DOUBLEWORD_CASES, STORE_SELECTOR_MAX_DEGREE, true, STORE_DOUBLEWORD_SELECTOR_WIDTH
        );
        encoder.write_flag_pt(row.slice_from(offsetof(Cols, selector)), shift);
        row.write_array(offsetof(Cols, prev_data), 2 * BLOCK_FE_WIDTH, &record.prev_data[0][0]);

        // The value bytes are the unconditional decomposition of the four rs2 cells; they feed both
        // the register-read reconstruction and (on odd shifts) the write splice.
        uint16_t value_bytes[STORE_DOUBLEWORD_WIDTH_CELLS][2];
        for (size_t i = 0; i < STORE_DOUBLEWORD_WIDTH_CELLS; i++) {
            value_bytes[i][0] = store_byte_from_cell(record.read_data[i], 0);
            value_bytes[i][1] = store_byte_from_cell(record.read_data[i], 1);
            bitwise_lookup.add_range(value_bytes[i][0], value_bytes[i][1]);
        }
        row.write_array(
            offsetof(Cols, value_bytes), STORE_DOUBLEWORD_WIDTH_CELLS * 2, &value_bytes[0][0]
        );

        uint16_t prev_bound_cells[2][2] = {};
        if (shift & 1) {
            uint32_t c0 = shift >> 1;
            for (size_t which = 0; which < 2; which++) {
                uint16_t cell =
                    store_prev_full_cell(record, c0 + which * STORE_DOUBLEWORD_WIDTH_CELLS);
                prev_bound_cells[which][0] = store_byte_from_cell(cell, 0);
                prev_bound_cells[which][1] = store_byte_from_cell(cell, 1);
            }
        }
        for (size_t which = 0; which < 2; which++) {
            bitwise_lookup.add_range(prev_bound_cells[which][0], prev_bound_cells[which][1]);
        }
        uint16_t prev_bound_bytes[2] = {prev_bound_cells[0][0], prev_bound_cells[1][1]};
        row.write_array(offsetof(Cols, prev_bound_bytes), 2, prev_bound_bytes);
    }
};

__global__ void rv64_store_doubleword_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64StoreRecord> records,
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
        auto adapter = Rv64StoreAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);
        auto core = StoreDoublewordCore(BitwiseOperationLookup(bitwise_lookup_ptr));
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64StoreDoublewordCols, core)),
            record.core,
            rv64_store_shift_amount(record.adapter)
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
    uint32_t *d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64StoreDoublewordCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_store_doubleword_tracegen<<<grid, block, 0, stream>>>(
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
