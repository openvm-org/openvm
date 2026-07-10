#include "riscv/cores/load.cuh"

template <typename T> struct LoadByteCoreCols {
    T selector[LOAD_BYTE_SELECTOR_WIDTH];
    T read_cell_bytes[2];
    T read_data[BLOCK_FE_WIDTH];
};

template <typename T> struct Rv64LoadByteCols {
    Rv64LoadByteAdapterCols<T> adapter;
    LoadByteCoreCols<T> core;
};

struct LoadByteCore {
    BitwiseOperationLookup bitwise_lookup;

    __device__ LoadByteCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, LoadRecord record, uint8_t shift) {
        uint16_t read_cell = record.read_data[0][shift >> 1];
        uint16_t read_cell_bytes[2] = {
            load_byte_from_cell(read_cell, 0),
            load_byte_from_cell(read_cell, 1),
        };
        bitwise_lookup.add_range(read_cell_bytes[0], read_cell_bytes[1]);

        Encoder encoder(LOAD_BYTE_CASES, LOAD_SELECTOR_MAX_DEGREE, true, LOAD_BYTE_SELECTOR_WIDTH);
        encoder.write_flag_pt(row.slice_from(COL_INDEX(LoadByteCoreCols, selector)), shift);
        COL_WRITE_ARRAY(row, LoadByteCoreCols, read_cell_bytes, read_cell_bytes);
        COL_WRITE_ARRAY(row, LoadByteCoreCols, read_data, record.read_data[0]);
    }
};

__global__ void rv64_load_byte_tracegen(
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
        auto adapter = Rv64LoadByteAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);
        auto core = LoadByteCore(BitwiseOperationLookup(bitwise_lookup_ptr));
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64LoadByteCols, core)),
            record.core,
            rv64_load_shift_amount(record.adapter)
        );
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_load_byte_tracegen(
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
    assert(width == sizeof(Rv64LoadByteCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_byte_tracegen<<<grid, block, 0, stream>>>(
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
