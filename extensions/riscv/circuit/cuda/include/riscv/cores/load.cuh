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

constexpr size_t LOAD_BYTE_SELECTOR_WIDTH = 3;
constexpr size_t LOAD_HALFWORD_SELECTOR_WIDTH = 3;
constexpr size_t LOAD_WORD_SELECTOR_WIDTH = 3;
constexpr size_t LOAD_DOUBLEWORD_SELECTOR_WIDTH = 3;

struct LoadRecord {
    // The block containing the effective address followed by the next block, which is all-zero
    // unless the access crosses a block boundary.
    uint16_t read_data[2][BLOCK_FE_WIDTH];
};

struct Rv64LoadRecord {
    Rv64LoadAdapterRecord adapter;
    LoadRecord core;
};

static_assert(sizeof(Rv64LoadAdapterRecord) == 44);
static_assert(sizeof(LoadRecord) == 16);
static_assert(sizeof(Rv64LoadRecord) == 60);
static_assert(offsetof(LoadRecord, read_data) == 0);
static_assert(offsetof(Rv64LoadRecord, core) == 44);

static __device__ __forceinline__ uint16_t load_byte_from_cell(uint16_t cell, uint8_t byte_idx) {
    return (cell >> (RV64_BYTE_BITS * byte_idx)) & 0xff;
}

static __device__ __forceinline__ uint16_t
load_read_full_cell(LoadRecord const &record, uint32_t cell) {
    return record.read_data[cell / BLOCK_FE_WIDTH][cell % BLOCK_FE_WIDTH];
}

template <typename T, size_t SELECTOR_WIDTH, size_t NUM_OVERLAP_CELLS> struct LoadWidthCoreCols {
    T selector[SELECTOR_WIDTH];
    T read_data[2][BLOCK_FE_WIDTH];
    T overlap_lo_bytes[NUM_OVERLAP_CELLS];
};

// Shared tracegen for the halfword/word/doubleword load cores. `WIDTH_BYTES` is the access
// width; `NUM_OVERLAP_CELLS` must equal `WIDTH_BYTES / 2 + 1`.
template <size_t SELECTOR_WIDTH, size_t NUM_OVERLAP_CELLS, size_t WIDTH_BYTES>
struct LoadWidthCore {
    using Cols = LoadWidthCoreCols<uint8_t, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>;

    BitwiseOperationLookup bitwise_lookup;

    __device__ LoadWidthCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, LoadRecord record, uint8_t shift) {
        Encoder encoder = shift_encoder(SELECTOR_WIDTH);
        encoder.write_flag_pt(row.slice_from(offsetof(Cols, selector)), shift);
        row.write_array(offsetof(Cols, read_data), 2 * BLOCK_FE_WIDTH, &record.read_data[0][0]);

        // On odd shifts, slot `j` holds the low byte of overlapped cell `j`. The high bytes are
        // derived in the AIR and only range checked here.
        uint16_t overlap_lo_bytes[NUM_OVERLAP_CELLS] = {};
        uint16_t overlap_hi_bytes[NUM_OVERLAP_CELLS] = {};
        if (shift & 1) {
            for (size_t j = 0; j < NUM_OVERLAP_CELLS; j++) {
                uint16_t cell = load_read_full_cell(record, (shift >> 1) + j);
                overlap_lo_bytes[j] = load_byte_from_cell(cell, 0);
                overlap_hi_bytes[j] = load_byte_from_cell(cell, 1);
            }
        }
        for (size_t j = 0; j < NUM_OVERLAP_CELLS; j++) {
            bitwise_lookup.add_range(overlap_lo_bytes[j], overlap_hi_bytes[j]);
        }
        row.write_array(offsetof(Cols, overlap_lo_bytes), NUM_OVERLAP_CELLS, overlap_lo_bytes);
    }
};
