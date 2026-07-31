#include "riscv/cores/load.cuh"

template <typename T> struct LoadByteCoreCols {
    T selector[BYTE_SHIFT_SELECTOR_WIDTH];
    T read_cell_lo_byte;
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

    __device__ void fill_trace_row(
        RowSlice row, uint16_t const (&read_data)[BLOCK_FE_WIDTH], uint8_t shift
    ) {
        uint16_t read_cell = read_data[shift >> 1];
        uint16_t read_cell_bytes[2] = {
            load_byte_from_cell(read_cell, 0),
            load_byte_from_cell(read_cell, 1),
        };
        bitwise_lookup.add_range(read_cell_bytes[0], read_cell_bytes[1]);

        Encoder encoder = shift_encoder();
        encoder.write_flag_pt(row.slice_from(COL_INDEX(LoadByteCoreCols, selector)), shift);
        COL_WRITE_VALUE(row, LoadByteCoreCols, read_cell_lo_byte, read_cell_bytes[0]);
        COL_WRITE_ARRAY(row, LoadByteCoreCols, read_data, read_data);
    }
};

#include "../rvr/src/load_byte.inc.cuh"
