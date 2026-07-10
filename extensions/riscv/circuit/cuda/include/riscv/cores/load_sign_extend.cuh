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

constexpr size_t LOAD_SIGN_EXTEND_BYTE_SELECTOR_WIDTH = 3;
constexpr uint32_t LOAD_SIGN_EXTEND_BYTE_CASES = 8;
constexpr size_t LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH = 3;
constexpr uint32_t LOAD_SIGN_EXTEND_HALFWORD_CASES = 8;
constexpr size_t LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH = 3;
constexpr uint32_t LOAD_SIGN_EXTEND_WORD_CASES = 8;
constexpr uint32_t LOAD_SIGN_EXTEND_SELECTOR_MAX_DEGREE = 2;
constexpr uint16_t SIGN_BYTE = 1 << (RV64_BYTE_BITS - 1);
constexpr uint16_t SIGN_U16 = 1 << (U16_BITS - 1);

using LoadSignExtendRecord = LoadRecord;

template <typename T, size_t SELECTOR_WIDTH, size_t NUM_TOUCHED_CELLS>
struct LoadSignExtendWidthCoreCols {
    T selector[SELECTOR_WIDTH];
    T is_valid;
    T cross;
    T data_most_sig_bit;
    T read_data[2][BLOCK_FE_WIDTH];
    T touched_cell_bytes[NUM_TOUCHED_CELLS][2];
};

struct Rv64LoadSignExtendRecord {
    Rv64LoadAdapterRecord adapter;
    LoadSignExtendRecord core;
};

static_assert(sizeof(Rv64LoadAdapterRecord) == 48);
static_assert(sizeof(LoadSignExtendRecord) == 16);
static_assert(sizeof(Rv64LoadSignExtendRecord) == 64);
static_assert(offsetof(LoadSignExtendRecord, read_data) == 0);
static_assert(offsetof(Rv64LoadSignExtendRecord, core) == 48);

static __device__ __forceinline__ uint16_t load_sign_extend_byte_from_cell(
    uint16_t cell,
    uint8_t byte_idx
) {
    return (cell >> (RV64_BYTE_BITS * byte_idx)) & 0xff;
}

// Shared tracegen for the halfword/word signed load cores. `WIDTH_BYTES` is the access width;
// `NUM_TOUCHED_CELLS` must equal `WIDTH_BYTES / 2 + 1`.
template <size_t SELECTOR_WIDTH, size_t NUM_TOUCHED_CELLS, uint32_t CASES, size_t WIDTH_BYTES>
struct LoadSignExtendWidthCore {
    using Cols = LoadSignExtendWidthCoreCols<uint8_t, SELECTOR_WIDTH, NUM_TOUCHED_CELLS>;

    VariableRangeChecker range_checker;
    BitwiseOperationLookup bitwise_lookup;

    __device__ LoadSignExtendWidthCore(
        VariableRangeChecker range_checker,
        BitwiseOperationLookup bitwise_lookup
    )
        : range_checker(range_checker), bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, LoadSignExtendRecord record, uint8_t shift) {
        constexpr size_t WIDTH_CELLS = WIDTH_BYTES / 2;

        Encoder encoder(CASES, LOAD_SIGN_EXTEND_SELECTOR_MAX_DEGREE, true, SELECTOR_WIDTH);
        encoder.write_flag_pt(row.slice_from(offsetof(Cols, selector)), shift);
        row[offsetof(Cols, is_valid)] = 1;
        row[offsetof(Cols, cross)] = (shift + WIDTH_BYTES > 2 * BLOCK_FE_WIDTH) ? 1 : 0;
        row.write_array(offsetof(Cols, read_data), 2 * BLOCK_FE_WIDTH, &record.read_data[0][0]);

        uint16_t touched_cell_bytes[NUM_TOUCHED_CELLS][2] = {};
        if (shift & 1) {
            for (size_t j = 0; j < NUM_TOUCHED_CELLS; j++) {
                uint16_t cell = load_read_full_cell(record, (shift >> 1) + j);
                touched_cell_bytes[j][0] = load_sign_extend_byte_from_cell(cell, 0);
                touched_cell_bytes[j][1] = load_sign_extend_byte_from_cell(cell, 1);
            }
        }
        for (size_t j = 0; j < NUM_TOUCHED_CELLS; j++) {
            bitwise_lookup.add_range(touched_cell_bytes[j][0], touched_cell_bytes[j][1]);
        }
        row.write_array(
            offsetof(Cols, touched_cell_bytes),
            NUM_TOUCHED_CELLS * 2,
            &touched_cell_bytes[0][0]
        );

        uint16_t sign_bit;
        if (shift & 1) {
            // The top loaded byte is the low byte of the last touched cell.
            uint16_t sign_byte = touched_cell_bytes[NUM_TOUCHED_CELLS - 1][0];
            sign_bit = sign_byte & SIGN_BYTE;
            range_checker.add_count(sign_byte - sign_bit, RV64_BYTE_BITS - 1);
        } else {
            // The top loaded byte is the high byte of the top cell.
            uint16_t sign_cell = load_read_full_cell(record, (shift >> 1) + WIDTH_CELLS - 1);
            sign_bit = sign_cell & SIGN_U16;
            range_checker.add_count(sign_cell - sign_bit, U16_BITS - 1);
        }
        row[offsetof(Cols, data_most_sig_bit)] = sign_bit != 0;
    }
};
