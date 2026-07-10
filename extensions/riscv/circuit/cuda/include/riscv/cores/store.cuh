#pragma once

#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/encoder.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/store.cuh"

using namespace riscv;
using namespace program;

constexpr size_t STORE_BYTE_SELECTOR_WIDTH = 3;
constexpr uint32_t STORE_BYTE_CASES = 8;
constexpr size_t STORE_HALFWORD_SELECTOR_WIDTH = 3;
constexpr uint32_t STORE_HALFWORD_CASES = 8;
constexpr size_t STORE_WORD_SELECTOR_WIDTH = 3;
constexpr uint32_t STORE_WORD_CASES = 8;
constexpr size_t STORE_DOUBLEWORD_SELECTOR_WIDTH = 3;
constexpr uint32_t STORE_DOUBLEWORD_CASES = 8;
constexpr uint32_t STORE_SELECTOR_MAX_DEGREE = 2;

struct StoreRecord {
    uint16_t read_data[BLOCK_FE_WIDTH];
    // Previous contents of the touched block followed by the next block, which is all-zero
    // unless the access crosses a block boundary.
    uint16_t prev_data[2][BLOCK_FE_WIDTH];
};

struct Rv64StoreRecord {
    Rv64StoreAdapterRecord adapter;
    StoreRecord core;
};

static_assert(sizeof(Rv64StoreAdapterRecord) == 40);
static_assert(sizeof(StoreRecord) == 24);
static_assert(sizeof(Rv64StoreRecord) == 64);
static_assert(offsetof(StoreRecord, read_data) == 0);
static_assert(offsetof(StoreRecord, prev_data) == 8);
static_assert(offsetof(Rv64StoreRecord, core) == 40);

static __device__ __forceinline__ uint16_t store_byte_from_cell(uint16_t cell, uint8_t byte_idx) {
    return (cell >> (RV64_BYTE_BITS * byte_idx)) & 0xff;
}

static __device__ __forceinline__ uint16_t
store_prev_full_cell(StoreRecord const &record, uint32_t cell) {
    return record.prev_data[cell / BLOCK_FE_WIDTH][cell % BLOCK_FE_WIDTH];
}

template <typename T, size_t SELECTOR_WIDTH, size_t NUM_VALUE_CELLS> struct StoreWidthCoreCols {
    T selector[SELECTOR_WIDTH];
    T read_data[BLOCK_FE_WIDTH];
    T prev_data[2][BLOCK_FE_WIDTH];
    T value_bytes[NUM_VALUE_CELLS][2];
    T prev_bound_bytes[2];
};

// Shared tracegen for the halfword/word/doubleword store cores. `WIDTH_BYTES` is the access
// width; `NUM_VALUE_CELLS` must equal `WIDTH_BYTES / 2`.
template <size_t SELECTOR_WIDTH, size_t NUM_VALUE_CELLS, uint32_t CASES, size_t WIDTH_BYTES>
struct StoreWidthCore {
    using Cols = StoreWidthCoreCols<uint8_t, SELECTOR_WIDTH, NUM_VALUE_CELLS>;

    BitwiseOperationLookup bitwise_lookup;

    __device__ StoreWidthCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, StoreRecord record, uint8_t shift) {
        Encoder encoder(CASES, STORE_SELECTOR_MAX_DEGREE, true, SELECTOR_WIDTH);
        encoder.write_flag_pt(row.slice_from(offsetof(Cols, selector)), shift);
        row.write_array(offsetof(Cols, read_data), BLOCK_FE_WIDTH, record.read_data);
        row.write_array(offsetof(Cols, prev_data), 2 * BLOCK_FE_WIDTH, &record.prev_data[0][0]);

        // Only the preserved boundary bytes are materialized: the low byte of the first
        // overlapped cell and the high byte of the last. The overwritten boundary-cell bytes
        // are only range checked, mirroring the AIR's derived bytes.
        uint16_t value_bytes[NUM_VALUE_CELLS][2] = {};
        uint16_t prev_bound_cells[2][2] = {};
        if (shift & 1) {
            for (size_t i = 0; i < NUM_VALUE_CELLS; i++) {
                value_bytes[i][0] = store_byte_from_cell(record.read_data[i], 0);
                value_bytes[i][1] = store_byte_from_cell(record.read_data[i], 1);
            }
            for (size_t which = 0; which < 2; which++) {
                uint16_t cell =
                    store_prev_full_cell(record, (shift >> 1) + which * NUM_VALUE_CELLS);
                prev_bound_cells[which][0] = store_byte_from_cell(cell, 0);
                prev_bound_cells[which][1] = store_byte_from_cell(cell, 1);
            }
        }
        for (size_t i = 0; i < NUM_VALUE_CELLS; i++) {
            bitwise_lookup.add_range(value_bytes[i][0], value_bytes[i][1]);
        }
        for (size_t which = 0; which < 2; which++) {
            bitwise_lookup.add_range(prev_bound_cells[which][0], prev_bound_cells[which][1]);
        }
        row.write_array(offsetof(Cols, value_bytes), NUM_VALUE_CELLS * 2, &value_bytes[0][0]);
        uint16_t prev_bound_bytes[2] = {prev_bound_cells[0][0], prev_bound_cells[1][1]};
        row.write_array(offsetof(Cols, prev_bound_bytes), 2, prev_bound_bytes);
    }
};
