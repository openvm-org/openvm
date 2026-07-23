#pragma once

#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/encoder.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/cores/load.cuh"

using namespace riscv;
using namespace program;

constexpr uint16_t SIGN_BYTE = 1 << (RV64_BYTE_BITS - 1);
constexpr uint16_t SIGN_U16 = 1 << (U16_BITS - 1);

using LoadSignExtendRecord = LoadRecord;

template <typename T, size_t WIDTH_BYTES> struct LoadSignExtendWidthCoreCols {
    static_assert(WIDTH_BYTES == HALFWORD_ACCESS_WIDTH || WIDTH_BYTES == WORD_ACCESS_WIDTH);
    static constexpr size_t NUM_OVERLAP_CELLS = WIDTH_BYTES / 2 + 1;

    T selector[BYTE_SHIFT_SELECTOR_WIDTH];
    T data_most_sig_bit;
    T read_data[2][BLOCK_FE_WIDTH];
    T overlap_lo_bytes[NUM_OVERLAP_CELLS];
};

struct Rv64LoadSignExtendRecord {
    Rv64LoadMultiByteAdapterRecord adapter;
    LoadSignExtendRecord core;
};

static_assert(sizeof(Rv64LoadMultiByteAdapterRecord) == 44);
static_assert(sizeof(LoadSignExtendRecord) == 16);
static_assert(sizeof(Rv64LoadSignExtendRecord) == 60);
static_assert(offsetof(LoadSignExtendRecord, read_data) == 0);
static_assert(offsetof(Rv64LoadSignExtendRecord, core) == 44);

static __device__ __forceinline__ uint16_t load_sign_extend_byte_from_cell(
    uint16_t cell,
    uint8_t byte_idx
) {
    return (cell >> (RV64_BYTE_BITS * byte_idx)) & RV64_BYTE_MASK;
}

// Shared tracegen for the halfword/word signed load cores.
template <size_t WIDTH_BYTES> struct LoadSignExtendWidthCore {
    using Cols = LoadSignExtendWidthCoreCols<uint8_t, WIDTH_BYTES>;
    static constexpr size_t NUM_OVERLAP_CELLS = Cols::NUM_OVERLAP_CELLS;

    VariableRangeChecker range_checker;
    BitwiseOperationLookup bitwise_lookup;

    __device__ LoadSignExtendWidthCore(
        VariableRangeChecker range_checker,
        BitwiseOperationLookup bitwise_lookup
    )
        : range_checker(range_checker), bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(
        RowSlice row,
        uint16_t const (&read_data)[2][BLOCK_FE_WIDTH],
        uint8_t shift
    ) {
        constexpr size_t WIDTH_CELLS = WIDTH_BYTES / 2;

        Encoder encoder = shift_encoder();
        encoder.write_flag_pt(row.slice_from(offsetof(Cols, selector)), shift);
        row.write_array(offsetof(Cols, read_data), 2 * BLOCK_FE_WIDTH, &read_data[0][0]);

        // On odd shifts, slot `j` holds the low byte of overlapped cell `j`. The high bytes are
        // derived in the AIR and only range checked here.
        uint16_t overlap_lo_bytes[NUM_OVERLAP_CELLS] = {};
        uint16_t overlap_hi_bytes[NUM_OVERLAP_CELLS] = {};
        if (shift & 1) {
            for (size_t j = 0; j < NUM_OVERLAP_CELLS; j++) {
                uint16_t cell = load_read_full_cell(read_data, (shift >> 1) + j);
                overlap_lo_bytes[j] = load_sign_extend_byte_from_cell(cell, 0);
                overlap_hi_bytes[j] = load_sign_extend_byte_from_cell(cell, 1);
            }
        }
        for (size_t j = 0; j < NUM_OVERLAP_CELLS; j++) {
            bitwise_lookup.add_range(overlap_lo_bytes[j], overlap_hi_bytes[j]);
        }
        row.write_array(offsetof(Cols, overlap_lo_bytes), NUM_OVERLAP_CELLS, overlap_lo_bytes);

        uint16_t sign_bit;
        if (shift & 1) {
            // The top loaded byte is the last overlapped cell's low byte. Shift the checked
            // value into the high byte so odd and even shifts use the same 15-bit range check.
            uint16_t sign_byte = overlap_lo_bytes[NUM_OVERLAP_CELLS - 1];
            sign_bit = sign_byte & SIGN_BYTE;
            range_checker.add_count((sign_byte - sign_bit) << RV64_BYTE_BITS, U16_BITS - 1);
        } else {
            // The top loaded byte is the high byte of the top cell.
            uint16_t sign_cell =
                load_read_full_cell(read_data, (shift >> 1) + WIDTH_CELLS - 1);
            sign_bit = sign_cell & SIGN_U16;
            range_checker.add_count(sign_cell - sign_bit, U16_BITS - 1);
        }
        row[offsetof(Cols, data_most_sig_bit)] = sign_bit != 0;
    }

    __device__ void fill_trace_row(RowSlice row, LoadSignExtendRecord record, uint8_t shift) {
        fill_trace_row(row, record.read_data, shift);
    }
};
