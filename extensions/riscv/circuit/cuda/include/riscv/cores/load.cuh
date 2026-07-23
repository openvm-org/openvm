#pragma once

#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/encoder.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/load.cuh"
#include "riscv/cores/shift_selector.cuh"

using namespace riscv;
using namespace program;

struct LoadRecord {
    // The block containing the effective address followed by the second block, which is all-zero
    // unless the access crosses a block boundary.
    uint16_t read_data[2][BLOCK_FE_WIDTH];
};

struct LoadByteRecord {
    uint16_t read_data[BLOCK_FE_WIDTH];
};

struct Rv64LoadRecord {
    Rv64LoadMultiByteAdapterRecord adapter;
    LoadRecord core;
};

struct Rv64LoadByteRecord {
    Rv64LoadByteAdapterRecord adapter;
    LoadByteRecord core;
};

static_assert(sizeof(Rv64LoadMultiByteAdapterRecord) == 44);
static_assert(sizeof(LoadRecord) == 16);
static_assert(sizeof(Rv64LoadRecord) == 60);
static_assert(offsetof(LoadRecord, read_data) == 0);
static_assert(offsetof(Rv64LoadRecord, core) == 44);
static_assert(sizeof(Rv64LoadByteAdapterRecord) == 40);
static_assert(sizeof(LoadByteRecord) == 8);
static_assert(sizeof(Rv64LoadByteRecord) == 48);
static_assert(offsetof(Rv64LoadByteRecord, core) == 40);

static __device__ __forceinline__ uint16_t load_byte_from_cell(uint16_t cell, uint8_t byte_idx) {
    return (cell >> (RV64_BYTE_BITS * byte_idx)) & RV64_BYTE_MASK;
}

static __device__ __forceinline__ uint16_t load_read_full_cell(
    uint16_t const (&read_data)[2][BLOCK_FE_WIDTH],
    uint32_t cell
) {
    return read_data[cell / BLOCK_FE_WIDTH][cell % BLOCK_FE_WIDTH];
}

static __device__ __forceinline__ uint16_t
load_read_full_cell(LoadRecord const &record, uint32_t cell) {
    return load_read_full_cell(record.read_data, cell);
}

template <typename T, size_t WIDTH_BYTES> struct LoadWidthCoreCols {
    static_assert(
        WIDTH_BYTES == HALFWORD_ACCESS_WIDTH || WIDTH_BYTES == WORD_ACCESS_WIDTH ||
        WIDTH_BYTES == DOUBLEWORD_ACCESS_WIDTH
    );
    static constexpr size_t NUM_OVERLAP_CELLS = WIDTH_BYTES / 2 + 1;

    T selector[BYTE_SHIFT_SELECTOR_WIDTH];
    T read_data[2][BLOCK_FE_WIDTH];
    T overlap_lo_bytes[NUM_OVERLAP_CELLS];
};

// Shared tracegen for the halfword/word/doubleword load cores.
template <size_t WIDTH_BYTES> struct LoadWidthCore {
    using Cols = LoadWidthCoreCols<uint8_t, WIDTH_BYTES>;
    static constexpr size_t NUM_OVERLAP_CELLS = Cols::NUM_OVERLAP_CELLS;

    BitwiseOperationLookup bitwise_lookup;

    __device__ LoadWidthCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(
        RowSlice row,
        uint16_t const (&read_data)[2][BLOCK_FE_WIDTH],
        uint8_t shift
    ) {
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
                overlap_lo_bytes[j] = load_byte_from_cell(cell, 0);
                overlap_hi_bytes[j] = load_byte_from_cell(cell, 1);
            }
        }
        for (size_t j = 0; j < NUM_OVERLAP_CELLS; j++) {
            bitwise_lookup.add_range(overlap_lo_bytes[j], overlap_hi_bytes[j]);
        }
        row.write_array(offsetof(Cols, overlap_lo_bytes), NUM_OVERLAP_CELLS, overlap_lo_bytes);
    }

    __device__ void fill_trace_row(RowSlice row, LoadRecord record, uint8_t shift) {
        fill_trace_row(row, record.read_data, shift);
    }
};
