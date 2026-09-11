#include "riscv/cores/store.cuh"

template <typename T> struct StoreByteCoreCols {
    T selector[BYTE_SHIFT_SELECTOR_WIDTH];
    // Low byte of the first source register cell; the high byte is derived in the AIR.
    T read_lo_byte;
    // Low byte of the selected previous memory cell; the high byte is derived in the AIR.
    T prev_cell_lo_byte;
    T read_data[BLOCK_FE_WIDTH];
    T prev_data[BLOCK_FE_WIDTH];
};

template <typename T> struct StoreByteCols {
    StoreByteAdapterCols<T> adapter;
    StoreByteCoreCols<T> core;
};

struct StoreByteCore {
    BitwiseOperationLookup bitwise_lookup;

    __device__ StoreByteCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(
        RowSlice row,
        uint16_t const (&read_data)[BLOCK_FE_WIDTH],
        uint16_t const (&prev_data)[BLOCK_FE_WIDTH],
        uint8_t shift
    ) {
        uint8_t cell_shift = shift >> 1;

        uint16_t read_lo_byte = store_byte_from_cell(read_data[0], 0);
        uint16_t prev_cell_bytes[2] = {
            store_byte_from_cell(prev_data[cell_shift], 0),
            store_byte_from_cell(prev_data[cell_shift], 1),
        };
        bitwise_lookup.add_range(read_lo_byte, store_byte_from_cell(read_data[0], 1));
        bitwise_lookup.add_range(prev_cell_bytes[0], prev_cell_bytes[1]);

        Encoder encoder = shift_encoder();
        encoder.write_flag_pt(row.slice_from(COL_INDEX(StoreByteCoreCols, selector)), shift);
        COL_WRITE_VALUE(row, StoreByteCoreCols, read_lo_byte, read_lo_byte);
        COL_WRITE_VALUE(row, StoreByteCoreCols, prev_cell_lo_byte, prev_cell_bytes[0]);
        COL_WRITE_ARRAY(row, StoreByteCoreCols, read_data, read_data);
        COL_WRITE_ARRAY(row, StoreByteCoreCols, prev_data, prev_data);
    }
};

#include "../rvr/src/store_byte.inc.cuh"
