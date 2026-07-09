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
constexpr size_t LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH = 2;
constexpr uint32_t LOAD_SIGN_EXTEND_HALFWORD_CASES = 4;
constexpr size_t LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH = 1;
constexpr uint32_t LOAD_SIGN_EXTEND_WORD_CASES = 2;
constexpr uint32_t LOAD_SIGN_EXTEND_SELECTOR_MAX_DEGREE = 2;
constexpr uint16_t SIGN_BYTE = 1 << (RV64_BYTE_BITS - 1);
constexpr uint16_t SIGN_U16 = 1 << (U16_BITS - 1);

using LoadSignExtendRecord = LoadRecord;

template <typename T, size_t SELECTOR_WIDTH> struct LoadSignExtendWidthAlignedCoreCols {
    T selector[SELECTOR_WIDTH];
    T is_valid;
    T data_most_sig_bit;
    T read_data[BLOCK_FE_WIDTH];
};

struct Rv64LoadSignExtendRecord {
    Rv64LoadAdapterRecord adapter;
    LoadSignExtendRecord core;
};

static_assert(sizeof(Rv64LoadAdapterRecord) == 44);
static_assert(sizeof(LoadSignExtendRecord) == 8);
static_assert(sizeof(Rv64LoadSignExtendRecord) == 52);
static_assert(offsetof(LoadSignExtendRecord, read_data) == 0);
static_assert(offsetof(Rv64LoadSignExtendRecord, core) == 44);

static __device__ __forceinline__ uint16_t load_sign_extend_byte_from_cell(
    uint16_t cell,
    uint8_t byte_idx
) {
    return (cell >> (RV64_BYTE_BITS * byte_idx)) & 0xff;
}
