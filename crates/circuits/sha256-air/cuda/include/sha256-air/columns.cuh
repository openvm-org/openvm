#pragma once

#include "primitives/constants.h"
#include "primitives/execution.h"
#include "system/memory/offline_checker.cuh"

using namespace riscv;
using namespace sha256;

template <typename T> struct Sha256FlagsCols {
    T is_round_row;
    T is_first_4_rows;
    T is_digest_row;
    T is_last_block;
    T row_idx[SHA256_ROW_VAR_CNT];
    T global_block_idx;
    T local_block_idx;
};

template <typename T> struct Sha256MessageHelperCols {
    T w_3[SHA256_ROUNDS_PER_ROW - 1][SHA256_WORD_U16S];
    T intermed_4[SHA256_ROUNDS_PER_ROW][SHA256_WORD_U16S];
    T intermed_8[SHA256_ROUNDS_PER_ROW][SHA256_WORD_U16S];
    T intermed_12[SHA256_ROUNDS_PER_ROW][SHA256_WORD_U16S];
};

template <typename T> struct Sha256MessageScheduleCols {
    T w[SHA256_ROUNDS_PER_ROW][SHA256_WORD_BITS];
    T carry_or_buffer[SHA256_ROUNDS_PER_ROW][SHA256_WORD_U8S];
};

template <typename T> struct Sha256WorkVarsCols {
    T a[SHA256_ROUNDS_PER_ROW][SHA256_WORD_BITS];
    T e[SHA256_ROUNDS_PER_ROW][SHA256_WORD_BITS];
    T carry_a[SHA256_ROUNDS_PER_ROW][SHA256_WORD_U16S];
    T carry_e[SHA256_ROUNDS_PER_ROW][SHA256_WORD_U16S];
};

template <typename T> struct Sha256RoundCols {
    Sha256FlagsCols<T> flags;
    Sha256WorkVarsCols<T> work_vars;
    Sha256MessageHelperCols<T> schedule_helper;
    Sha256MessageScheduleCols<T> message_schedule;
};

template <typename T> struct Sha256DigestCols {
    Sha256FlagsCols<T> flags;
    Sha256WorkVarsCols<T> hash;
    Sha256MessageHelperCols<T> schedule_helper;
    T final_hash[SHA256_HASH_WORDS][SHA256_WORD_U8S];
    T prev_hash[SHA256_HASH_WORDS][SHA256_WORD_U16S];
};

template <typename T> struct Sha256VmControlCols {
    T len;
    T cur_timestamp;
    T read_ptr;
    T pad_flags[6];
    T padding_occurred;
};

template <typename T> struct Sha256VmRoundCols {
    Sha256VmControlCols<T> control;
    Sha256RoundCols<T> inner;
    MemoryReadAuxCols<T> read_aux;
};

template <typename T> struct Sha256VmDigestCols {
    Sha256VmControlCols<T> control;
    Sha256DigestCols<T> inner;
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs1_ptr;
    T rs2_ptr;
    T dst_ptr[RV32_REGISTER_NUM_LIMBS];
    T src_ptr[RV32_REGISTER_NUM_LIMBS];
    T len_data[RV32_REGISTER_NUM_LIMBS];
    MemoryReadAuxCols<T> register_reads_aux[SHA256_REGISTER_READS];
    MemoryWriteAuxCols<T, SHA256_WRITE_SIZE> writes_aux;
};

struct Sha256InnerBlockRecord {
    uint32_t global_block_idx;
    uint32_t local_block_idx;
    uint32_t is_last_block;
    uint32_t prev_hash[SHA256_HASH_WORDS];
    uint32_t input_words[SHA256_BLOCK_WORDS];
};

__global__ void sha256_fill_invalid_rows(
    Fp *d_trace,
    size_t trace_height,
    size_t trace_width,
    uint32_t rows_used
);

// ===== MACROS AND CONSTANTS =====
static constexpr size_t SHA256VM_CONTROL_WIDTH = sizeof(Sha256VmControlCols<uint8_t>);
static constexpr size_t SHA256_ROUND_WIDTH = sizeof(Sha256RoundCols<uint8_t>);
static constexpr size_t SHA256_DIGEST_WIDTH = sizeof(Sha256DigestCols<uint8_t>);
static constexpr size_t SHA256VM_ROUND_WIDTH = sizeof(Sha256VmRoundCols<uint8_t>);
static constexpr size_t SHA256VM_DIGEST_WIDTH = sizeof(Sha256VmDigestCols<uint8_t>);

static constexpr size_t SHA256_WIDTH =
    (SHA256_ROUND_WIDTH > SHA256_DIGEST_WIDTH) ? SHA256_ROUND_WIDTH : SHA256_DIGEST_WIDTH;
static constexpr size_t SHA256VM_WIDTH =
    (SHA256VM_ROUND_WIDTH > SHA256VM_DIGEST_WIDTH) ? SHA256VM_ROUND_WIDTH : SHA256VM_DIGEST_WIDTH;
static constexpr size_t SHA256_INNER_COLUMN_OFFSET = sizeof(Sha256VmControlCols<uint8_t>);

#define SHA256_WRITE_ROUND(row, FIELD, VALUE) COL_WRITE_VALUE(row, Sha256VmRoundCols, FIELD, VALUE)
#define SHA256_WRITE_DIGEST(row, FIELD, VALUE)                                                     \
    COL_WRITE_VALUE(row, Sha256VmDigestCols, FIELD, VALUE)
#define SHA256_WRITE_ARRAY_ROUND(row, FIELD, VALUES)                                               \
    COL_WRITE_ARRAY(row, Sha256VmRoundCols, FIELD, VALUES)
#define SHA256_WRITE_ARRAY_DIGEST(row, FIELD, VALUES)                                              \
    COL_WRITE_ARRAY(row, Sha256VmDigestCols, FIELD, VALUES)
#define SHA256_FILL_ZERO_ROUND(row, FIELD) COL_FILL_ZERO(row, Sha256VmRoundCols, FIELD)
#define SHA256_FILL_ZERO_DIGEST(row, FIELD) COL_FILL_ZERO(row, Sha256VmDigestCols, FIELD)
#define SHA256_SLICE_ROUND(row, FIELD) row.slice_from(COL_INDEX(Sha256VmRoundCols, FIELD))
#define SHA256_SLICE_DIGEST(row, FIELD) row.slice_from(COL_INDEX(Sha256VmDigestCols, FIELD))

#define SHA256INNER_WRITE_ROUND(row, FIELD, VALUE)                                                 \
    COL_WRITE_VALUE(row, Sha256RoundCols, FIELD, VALUE)
#define SHA256INNER_WRITE_DIGEST(row, FIELD, VALUE)                                                \
    COL_WRITE_VALUE(row, Sha256DigestCols, FIELD, VALUE)
#define SHA256INNER_WRITE_ARRAY_ROUND(row, FIELD, VALUES)                                          \
    COL_WRITE_ARRAY(row, Sha256RoundCols, FIELD, VALUES)
#define SHA256INNER_WRITE_ARRAY_DIGEST(row, FIELD, VALUES)                                         \
    COL_WRITE_ARRAY(row, Sha256DigestCols, FIELD, VALUES)
#define SHA256INNER_FILL_ZERO_ROUND(row, FIELD) COL_FILL_ZERO(row, Sha256RoundCols, FIELD)
#define SHA256INNER_FILL_ZERO_DIGEST(row, FIELD) COL_FILL_ZERO(row, Sha256DigestCols, FIELD)