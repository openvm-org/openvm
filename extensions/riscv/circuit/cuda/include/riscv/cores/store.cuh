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

struct StoreRecord {
    uint16_t read_data[BLOCK_FE_WIDTH];
    // Previous contents of the first block followed by the second block, which is all-zero
    // unless the access crosses a block boundary.
    uint16_t prev_data[2][BLOCK_FE_WIDTH];
};

struct StoreByteRecord {
    uint16_t read_data[BLOCK_FE_WIDTH];
    uint16_t prev_data[BLOCK_FE_WIDTH];
};

struct Rv64StoreRecord {
    Rv64StoreMultiByteAdapterRecord adapter;
    StoreRecord core;
};

struct Rv64StoreByteRecord {
    Rv64StoreByteAdapterRecord adapter;
    StoreByteRecord core;
};

static_assert(sizeof(Rv64StoreMultiByteAdapterRecord) == 36);
static_assert(sizeof(StoreRecord) == 24);
static_assert(sizeof(Rv64StoreRecord) == 60);
static_assert(offsetof(StoreRecord, read_data) == 0);
static_assert(offsetof(StoreRecord, prev_data) == 8);
static_assert(offsetof(Rv64StoreRecord, core) == 36);
static_assert(sizeof(Rv64StoreByteAdapterRecord) == 32);
static_assert(sizeof(StoreByteRecord) == 16);
static_assert(sizeof(Rv64StoreByteRecord) == 48);
static_assert(offsetof(StoreByteRecord, prev_data) == 8);
static_assert(offsetof(Rv64StoreByteRecord, core) == 32);

static __device__ __forceinline__ uint16_t store_byte_from_cell(uint16_t cell, uint8_t byte_idx) {
    return (cell >> (RV64_BYTE_BITS * byte_idx)) & RV64_BYTE_MASK;
}

static __device__ __forceinline__ uint16_t
store_prev_full_cell(StoreRecord const &record, uint32_t cell) {
    return record.prev_data[cell / BLOCK_FE_WIDTH][cell % BLOCK_FE_WIDTH];
}

template <typename T, size_t WIDTH_BYTES> struct StoreWidthCoreCols {
    static_assert(
        WIDTH_BYTES == HALFWORD_ACCESS_WIDTH || WIDTH_BYTES == WORD_ACCESS_WIDTH ||
        WIDTH_BYTES == DOUBLEWORD_ACCESS_WIDTH
    );
    static constexpr size_t NUM_VALUE_CELLS = WIDTH_BYTES / 2;

    T selector[BYTE_SHIFT_SELECTOR_WIDTH];
    T read_data[BLOCK_FE_WIDTH];
    T prev_data[2][BLOCK_FE_WIDTH];
    T value_lo_bytes[NUM_VALUE_CELLS];
    T prev_bound_bytes[2];
};

// Shared tracegen for the halfword/word/doubleword store cores.
template <size_t WIDTH_BYTES> struct StoreWidthCore {
    using Cols = StoreWidthCoreCols<uint8_t, WIDTH_BYTES>;
    static constexpr size_t NUM_VALUE_CELLS = Cols::NUM_VALUE_CELLS;

    BitwiseOperationLookup bitwise_lookup;

    __device__ StoreWidthCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, StoreRecord record, uint8_t shift) {
        Encoder encoder = shift_encoder();
        encoder.write_flag_pt(row.slice_from(offsetof(Cols, selector)), shift);
        row.write_array(offsetof(Cols, read_data), BLOCK_FE_WIDTH, record.read_data);
        row.write_array(offsetof(Cols, prev_data), 2 * BLOCK_FE_WIDTH, &record.prev_data[0][0]);

        // Odd shifts materialize the value cells' low bytes and the two preserved boundary
        // bytes. The AIR derives each paired byte; even shifts leave these columns zero and emit
        // no byte lookups.
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
