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
constexpr size_t STORE_HALFWORD_SELECTOR_WIDTH = 2;
constexpr uint32_t STORE_HALFWORD_CASES = 4;
constexpr size_t STORE_WORD_SELECTOR_WIDTH = 1;
constexpr uint32_t STORE_WORD_CASES = 2;
constexpr size_t STORE_DOUBLEWORD_SELECTOR_WIDTH = 1;
constexpr uint32_t STORE_DOUBLEWORD_CASES = 1;
constexpr uint32_t STORE_SELECTOR_MAX_DEGREE = 2;

struct StoreRecord {
    uint16_t read_data[BLOCK_FE_WIDTH];
    uint16_t prev_data[BLOCK_FE_WIDTH];
};

struct Rv64StoreRecord {
    Rv64StoreAdapterRecord adapter;
    StoreRecord core;
};

static_assert(sizeof(Rv64StoreAdapterRecord) == 36);
static_assert(sizeof(StoreRecord) == 16);
static_assert(sizeof(Rv64StoreRecord) == 52);
static_assert(offsetof(StoreRecord, read_data) == 0);
static_assert(offsetof(StoreRecord, prev_data) == 8);
static_assert(offsetof(Rv64StoreRecord, core) == 36);

static __device__ __forceinline__ uint16_t store_byte_from_cell(uint16_t cell, uint8_t byte_idx) {
    return (cell >> (RV64_BYTE_BITS * byte_idx)) & 0xff;
}
