#pragma once

#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/encoder.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/load.cuh"

using namespace riscv;
using namespace program;

constexpr size_t LOAD_BYTE_SELECTOR_WIDTH = 3;
constexpr uint32_t LOAD_BYTE_CASES = 8;
constexpr size_t LOAD_HALFWORD_SELECTOR_WIDTH = 3;
constexpr uint32_t LOAD_HALFWORD_CASES = 8;
constexpr size_t LOAD_WORD_SELECTOR_WIDTH = 3;
constexpr uint32_t LOAD_WORD_CASES = 8;
constexpr size_t LOAD_DOUBLEWORD_SELECTOR_WIDTH = 3;
constexpr uint32_t LOAD_DOUBLEWORD_CASES = 8;
constexpr uint32_t LOAD_SELECTOR_MAX_DEGREE = 2;

struct LoadRecord {
    // The block containing the effective address followed by the next block, which is all-zero
    // unless the access crosses a block boundary.
    uint16_t read_data[2][BLOCK_FE_WIDTH];
};

struct Rv64LoadRecord {
    Rv64LoadAdapterRecord adapter;
    LoadRecord core;
};

static_assert(sizeof(Rv64LoadAdapterRecord) == 48);
static_assert(sizeof(LoadRecord) == 16);
static_assert(sizeof(Rv64LoadRecord) == 64);
static_assert(offsetof(LoadRecord, read_data) == 0);
static_assert(offsetof(Rv64LoadRecord, core) == 48);

static __device__ __forceinline__ uint16_t load_byte_from_cell(uint16_t cell, uint8_t byte_idx) {
    return (cell >> (RV64_BYTE_BITS * byte_idx)) & 0xff;
}

static __device__ __forceinline__ uint16_t
load_read_full_cell(LoadRecord const &record, uint32_t cell) {
    return record.read_data[cell / BLOCK_FE_WIDTH][cell % BLOCK_FE_WIDTH];
}

template <typename T, size_t SELECTOR_WIDTH, size_t NUM_LOADED_CELLS> struct LoadWidthCoreCols {
    T selector[SELECTOR_WIDTH];
    T read_data[2][BLOCK_FE_WIDTH];
    T loaded_cell_bytes[NUM_LOADED_CELLS][2];
};

// Shared tracegen for the halfword/word/doubleword load cores. `WIDTH_BYTES` is the access
// width; `NUM_LOADED_CELLS` must equal `WIDTH_BYTES / 2`.
template <size_t SELECTOR_WIDTH, size_t NUM_LOADED_CELLS, uint32_t CASES, size_t WIDTH_BYTES>
struct LoadWidthCore {
    using Cols = LoadWidthCoreCols<uint8_t, SELECTOR_WIDTH, NUM_LOADED_CELLS>;

    BitwiseOperationLookup bitwise_lookup;

    __device__ LoadWidthCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, LoadRecord record, uint8_t shift) {
        Encoder encoder(CASES, LOAD_SELECTOR_MAX_DEGREE, true, SELECTOR_WIDTH);
        encoder.write_flag_pt(row.slice_from(offsetof(Cols, selector)), shift);
        row.write_array(offsetof(Cols, read_data), 2 * BLOCK_FE_WIDTH, &record.read_data[0][0]);

        // On odd shifts, slot `i` holds the byte decomposition [lo, hi] of result cell `i`: the
        // high byte of overlapped cell `i` and the low byte of overlapped cell `i + 1`. The two
        // overlapped-cell bytes outside the loaded range are only range checked, mirroring the
        // AIR's derived boundary bytes.
        uint16_t loaded_cell_bytes[NUM_LOADED_CELLS][2] = {};
        uint16_t bound_bytes[2] = {};
        if (shift & 1) {
            for (size_t i = 0; i < NUM_LOADED_CELLS; i++) {
                loaded_cell_bytes[i][0] =
                    load_byte_from_cell(load_read_full_cell(record, (shift >> 1) + i), 1);
                loaded_cell_bytes[i][1] =
                    load_byte_from_cell(load_read_full_cell(record, (shift >> 1) + i + 1), 0);
            }
            bound_bytes[0] = load_byte_from_cell(load_read_full_cell(record, shift >> 1), 0);
            bound_bytes[1] = load_byte_from_cell(
                load_read_full_cell(record, (shift >> 1) + NUM_LOADED_CELLS), 1
            );
        }
        for (size_t i = 0; i < NUM_LOADED_CELLS; i++) {
            bitwise_lookup.add_range(loaded_cell_bytes[i][0], loaded_cell_bytes[i][1]);
        }
        bitwise_lookup.add_range(bound_bytes[0], bound_bytes[1]);
        row.write_array(
            offsetof(Cols, loaded_cell_bytes),
            NUM_LOADED_CELLS * 2,
            &loaded_cell_bytes[0][0]
        );
    }
};
