#include "riscv/cores/load.cuh"

// Doubleword load with the column-reduced layout (see the Rust `LoadDoublewordCoreAir`): the two
// read blocks are reconstructed for the bus from the `NUM_SLOTS` window-cell decompositions plus
// the `NUM_NONOVERLAP` cells outside the window, so `read_data` is not stored.
static constexpr size_t LOAD_DOUBLEWORD_NUM_SLOTS = 5;
static constexpr size_t LOAD_DOUBLEWORD_NUM_NONOVERLAP = 3;

template <typename T> struct LoadDoublewordCoreCols {
    T selector[LOAD_DOUBLEWORD_SELECTOR_WIDTH];
    T read_nonoverlap[LOAD_DOUBLEWORD_NUM_NONOVERLAP];
    T cell_bytes[LOAD_DOUBLEWORD_NUM_SLOTS][2];
};

template <typename T> struct Rv64LoadDoublewordCols {
    Rv64LoadAdapterCols<T> adapter;
    LoadDoublewordCoreCols<T> core;
};

struct LoadDoublewordCore {
    using Cols = LoadDoublewordCoreCols<uint8_t>;

    BitwiseOperationLookup bitwise_lookup;

    __device__ LoadDoublewordCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, LoadRecord record, uint8_t shift) {
        Encoder encoder(
            LOAD_DOUBLEWORD_CASES, LOAD_SELECTOR_MAX_DEGREE, true, LOAD_DOUBLEWORD_SELECTOR_WIDTH
        );
        encoder.write_flag_pt(row.slice_from(offsetof(Cols, selector)), shift);

        uint32_t c0 = shift >> 1;
        uint16_t cell_bytes[LOAD_DOUBLEWORD_NUM_SLOTS][2];
        for (size_t j = 0; j < LOAD_DOUBLEWORD_NUM_SLOTS; j++) {
            uint16_t cell = load_read_full_cell(record, c0 + j);
            cell_bytes[j][0] = load_byte_from_cell(cell, 0);
            cell_bytes[j][1] = load_byte_from_cell(cell, 1);
            bitwise_lookup.add_range(cell_bytes[j][0], cell_bytes[j][1]);
        }
        row.write_array(offsetof(Cols, cell_bytes), LOAD_DOUBLEWORD_NUM_SLOTS * 2, &cell_bytes[0][0]);

        uint16_t read_nonoverlap[LOAD_DOUBLEWORD_NUM_NONOVERLAP];
        for (size_t k = 0; k < LOAD_DOUBLEWORD_NUM_NONOVERLAP; k++) {
            uint32_t p = (k < c0) ? k : k + LOAD_DOUBLEWORD_NUM_SLOTS;
            read_nonoverlap[k] = load_read_full_cell(record, p);
        }
        row.write_array(
            offsetof(Cols, read_nonoverlap), LOAD_DOUBLEWORD_NUM_NONOVERLAP, read_nonoverlap
        );
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
        auto core = LoadDoublewordCore(BitwiseOperationLookup(bitwise_lookup_ptr));
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
    uint32_t *d_bitwise_lookup,
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
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
