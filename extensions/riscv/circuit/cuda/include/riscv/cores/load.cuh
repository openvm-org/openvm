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
constexpr size_t LOAD_HALFWORD_SELECTOR_WIDTH = 2;
constexpr uint32_t LOAD_HALFWORD_CASES = 4;
constexpr size_t LOAD_WORD_SELECTOR_WIDTH = 1;
constexpr uint32_t LOAD_WORD_CASES = 2;
constexpr size_t LOAD_DOUBLEWORD_SELECTOR_WIDTH = 1;
constexpr uint32_t LOAD_DOUBLEWORD_CASES = 1;
constexpr uint32_t LOAD_SELECTOR_MAX_DEGREE = 2;

struct LoadRecord {
    uint16_t read_data[BLOCK_FE_WIDTH];
};

struct Rv64LoadRecord {
    Rv64LoadAdapterRecord adapter;
    LoadRecord core;
};

static_assert(sizeof(Rv64LoadAdapterRecord) == 44);
static_assert(sizeof(LoadRecord) == 8);
static_assert(sizeof(Rv64LoadRecord) == 52);
static_assert(offsetof(LoadRecord, read_data) == 0);
static_assert(offsetof(Rv64LoadRecord, core) == 44);

static __device__ __forceinline__ uint16_t load_byte_from_cell(uint16_t cell, uint8_t byte_idx) {
    return (cell >> (RV64_BYTE_BITS * byte_idx)) & 0xff;
}
