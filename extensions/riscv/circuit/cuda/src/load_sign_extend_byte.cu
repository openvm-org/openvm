#include "riscv/cores/load_sign_extend.cuh"

template <typename T> struct LoadSignExtendByteCoreCols {
    T selector[BYTE_SHIFT_SELECTOR_WIDTH];
    T data_most_sig_bit;
    T read_cell_lo_byte;
    T read_data[BLOCK_FE_WIDTH];
};

template <typename T> struct LoadSignExtendByteCols {
    LoadByteAdapterCols<T> adapter;
    LoadSignExtendByteCoreCols<T> core;
};

struct LoadSignExtendByteCore {
    VariableRangeChecker range_checker;
    BitwiseOperationLookup bitwise_lookup;

    __device__ LoadSignExtendByteCore(
        VariableRangeChecker range_checker,
        BitwiseOperationLookup bitwise_lookup
    )
        : range_checker(range_checker), bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(
        RowSlice row, uint16_t const (&read_data)[BLOCK_FE_WIDTH], uint8_t shift
    ) {
        uint16_t read_cell = read_data[shift >> 1];
        uint16_t read_cell_bytes[2] = {
            load_sign_extend_byte_from_cell(read_cell, 0),
            load_sign_extend_byte_from_cell(read_cell, 1),
        };
        uint16_t selected_byte = read_cell_bytes[shift & 1];
        uint16_t sign_bit = selected_byte & SIGN_BYTE;

        bitwise_lookup.add_range(read_cell_bytes[0], read_cell_bytes[1]);
        range_checker.add_count(selected_byte - sign_bit, BYTE_BITS - 1);

        Encoder encoder = shift_encoder();
        encoder.write_flag_pt(
            row.slice_from(COL_INDEX(LoadSignExtendByteCoreCols, selector)),
            shift
        );
        COL_WRITE_VALUE(row, LoadSignExtendByteCoreCols, data_most_sig_bit, sign_bit != 0);
        COL_WRITE_VALUE(
            row, LoadSignExtendByteCoreCols, read_cell_lo_byte, read_cell_bytes[0]
        );
        COL_WRITE_ARRAY(row, LoadSignExtendByteCoreCols, read_data, read_data);
    }
};

#include "../rvr/src/load_sign_extend_byte.inc.cuh"
