#pragma once

#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/encoder.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/store.cuh"
#include "riscv/cores/shift_selector.cuh"

using namespace riscv;
using namespace program;

constexpr size_t STORE_BYTE_SELECTOR_WIDTH = 3;
constexpr size_t STORE_HALFWORD_SELECTOR_WIDTH = 3;
constexpr size_t STORE_WORD_SELECTOR_WIDTH = 3;
constexpr size_t STORE_DOUBLEWORD_SELECTOR_WIDTH = 3;

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

static_assert(sizeof(Rv64StoreAdapterRecord) == 36);
static_assert(sizeof(StoreRecord) == 24);
static_assert(sizeof(Rv64StoreRecord) == 60);
static_assert(offsetof(StoreRecord, read_data) == 0);
static_assert(offsetof(StoreRecord, prev_data) == 8);
static_assert(offsetof(Rv64StoreRecord, core) == 36);

static __device__ __forceinline__ uint16_t store_byte_from_cell(uint16_t cell, uint8_t byte_idx) {
    return (cell >> (RV64_BYTE_BITS * byte_idx)) & RV64_BYTE_MASK;
}

static __device__ __forceinline__ uint16_t
store_prev_full_cell(StoreRecord const &record, uint32_t cell) {
    return record.prev_data[cell / BLOCK_FE_WIDTH][cell % BLOCK_FE_WIDTH];
}

template <typename T, size_t SELECTOR_WIDTH, size_t NUM_VALUE_CELLS> struct StoreWidthCoreCols {
    T selector[SELECTOR_WIDTH];
    T read_data[BLOCK_FE_WIDTH];
    T prev_data[2][BLOCK_FE_WIDTH];
    T value_lo_bytes[NUM_VALUE_CELLS];
    T prev_bound_bytes[2];
};

// Shared tracegen for the halfword/word/doubleword store cores. `WIDTH_BYTES` is the access
// width; `NUM_VALUE_CELLS` must equal `WIDTH_BYTES / 2`.
template <size_t SELECTOR_WIDTH, size_t NUM_VALUE_CELLS, size_t WIDTH_BYTES>
struct StoreWidthCore {
    using Cols = StoreWidthCoreCols<uint8_t, SELECTOR_WIDTH, NUM_VALUE_CELLS>;

    BitwiseOperationLookup bitwise_lookup;

    __device__ StoreWidthCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, StoreRecord record, uint8_t shift) {
        Encoder encoder = shift_encoder(SELECTOR_WIDTH);
        encoder.write_flag_pt(row.slice_from(offsetof(Cols, selector)), shift);
        row.write_array(offsetof(Cols, read_data), BLOCK_FE_WIDTH, record.read_data);
        row.write_array(offsetof(Cols, prev_data), 2 * BLOCK_FE_WIDTH, &record.prev_data[0][0]);

        // Only the value cells' low bytes and the preserved boundary bytes (the low byte of the
        // first overlapped cell, the high byte of the last) are materialized; their counterparts
        // are derived in the AIR and only range checked. The AIR's range checks are gated on the
        // odd-shift selector sum, so even shifts request no lookups.
        uint16_t value_lo_bytes[NUM_VALUE_CELLS] = {};
        uint16_t prev_bound_bytes[2] = {};
        if (shift & 1) {
            for (size_t i = 0; i < NUM_VALUE_CELLS; i++) {
                value_lo_bytes[i] = store_byte_from_cell(record.read_data[i], 0);
                bitwise_lookup.add_range(
                    value_lo_bytes[i], store_byte_from_cell(record.read_data[i], 1)
                );
            }
            for (size_t which = 0; which < 2; which++) {
                uint16_t cell =
                    store_prev_full_cell(record, (shift >> 1) + which * NUM_VALUE_CELLS);
                bitwise_lookup.add_range(
                    store_byte_from_cell(cell, 0), store_byte_from_cell(cell, 1)
                );
                prev_bound_bytes[which] = store_byte_from_cell(cell, which);
            }
        }
        row.write_array(offsetof(Cols, value_lo_bytes), NUM_VALUE_CELLS, value_lo_bytes);
        row.write_array(offsetof(Cols, prev_bound_bytes), 2, prev_bound_bytes);
    }
};
