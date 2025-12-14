#pragma once

#include <cstddef>
#include <cstdint>

// Column structs matching the new block-hasher AIR (request_id + inner round/digest columns).
template <typename V, typename T> struct Sha2FlagsCols {
    T is_round_row;
    T is_first_4_rows;
    T is_digest_row;
    T row_idx[V::ROW_VAR_CNT];
    T global_block_idx;
    T local_block_idx;
};

template <typename V, typename T> struct Sha2MessageHelperCols {
    T w_3[V::ROUNDS_PER_ROW_MINUS_ONE][V::WORD_U16S];
    T intermed_4[V::ROUNDS_PER_ROW][V::WORD_U16S];
    T intermed_8[V::ROUNDS_PER_ROW][V::WORD_U16S];
    T intermed_12[V::ROUNDS_PER_ROW][V::WORD_U16S];
};

template <typename V, typename T> struct Sha2MessageScheduleCols {
    T w[V::ROUNDS_PER_ROW][V::WORD_BITS];
    T carry_or_buffer[V::ROUNDS_PER_ROW][V::WORD_U8S];
};

template <typename V, typename T> struct Sha2WorkVarsCols {
    T a[V::ROUNDS_PER_ROW][V::WORD_BITS];
    T e[V::ROUNDS_PER_ROW][V::WORD_BITS];
    T carry_a[V::ROUNDS_PER_ROW][V::WORD_U16S];
    T carry_e[V::ROUNDS_PER_ROW][V::WORD_U16S];
};

template <typename V, typename T> struct Sha2RoundCols {
    Sha2FlagsCols<V, T> flags;
    Sha2WorkVarsCols<V, T> work_vars;
    Sha2MessageHelperCols<V, T> schedule_helper;
    Sha2MessageScheduleCols<V, T> message_schedule;
};

template <typename V, typename T> struct Sha2DigestCols {
    Sha2FlagsCols<V, T> flags;
    Sha2WorkVarsCols<V, T> hash;
    Sha2MessageHelperCols<V, T> schedule_helper;
    T final_hash[V::HASH_WORDS][V::WORD_U8S];
    T prev_hash[V::HASH_WORDS][V::WORD_U16S];
};

template <typename V, typename T> struct Sha2BlockHasherRoundCols {
    T request_id;
    Sha2RoundCols<V, T> inner;
};

template <typename V, typename T> struct Sha2BlockHasherDigestCols {
    T request_id;
    Sha2DigestCols<V, T> inner;
};

template <typename V> struct Sha2Layout {
    static constexpr size_t ROUND_WIDTH = sizeof(Sha2BlockHasherRoundCols<V, uint8_t>);
    static constexpr size_t DIGEST_WIDTH = sizeof(Sha2BlockHasherDigestCols<V, uint8_t>);
    static constexpr size_t WIDTH = (ROUND_WIDTH > DIGEST_WIDTH) ? ROUND_WIDTH : DIGEST_WIDTH;
    static constexpr size_t INNER_OFFSET = sizeof(uint8_t); // request_id
    static constexpr size_t INNER_COLUMN_OFFSET = sizeof(uint8_t);
};

#define SHA2_COL_INDEX(V, STRUCT, FIELD)                                                           \
    (reinterpret_cast<size_t>(&(reinterpret_cast<STRUCT<V, uint8_t> *>(0)->FIELD)))
#define SHA2_COL_ARRAY_LEN(V, STRUCT, FIELD)                                                       \
    (sizeof((reinterpret_cast<STRUCT<V, uint8_t> *>(0)->FIELD)))
#define SHA2_WRITE_VALUE(V, ROW, STRUCT, FIELD, VALUE)                                             \
    (ROW).write(SHA2_COL_INDEX(V, STRUCT, FIELD), VALUE)
#define SHA2_WRITE_ARRAY(V, ROW, STRUCT, FIELD, VALUES)                                            \
    (ROW).write_array(                                                                             \
        SHA2_COL_INDEX(V, STRUCT, FIELD), SHA2_COL_ARRAY_LEN(V, STRUCT, FIELD), VALUES             \
    )
#define SHA2_WRITE_BITS(V, ROW, STRUCT, FIELD, VALUE)                                              \
    (ROW).write_bits(SHA2_COL_INDEX(V, STRUCT, FIELD), VALUE)
#define SHA2_FILL_ZERO(V, ROW, STRUCT, FIELD)                                                      \
    (ROW).fill_zero(SHA2_COL_INDEX(V, STRUCT, FIELD), SHA2_COL_ARRAY_LEN(V, STRUCT, FIELD))
#define SHA2_SLICE_FROM(V, ROW, STRUCT, FIELD) (ROW).slice_from(SHA2_COL_INDEX(V, STRUCT, FIELD))

#define SHA2_WRITE_ROUND(V, row, FIELD, VALUE)                                                     \
    SHA2_WRITE_VALUE(V, row, Sha2BlockHasherRoundCols, FIELD, VALUE)
#define SHA2_WRITE_DIGEST(V, row, FIELD, VALUE)                                                    \
    SHA2_WRITE_VALUE(V, row, Sha2BlockHasherDigestCols, FIELD, VALUE)
#define SHA2_WRITE_ARRAY_ROUND(V, row, FIELD, VALUES)                                              \
    SHA2_WRITE_ARRAY(V, row, Sha2BlockHasherRoundCols, FIELD, VALUES)
#define SHA2_WRITE_ARRAY_DIGEST(V, row, FIELD, VALUES)                                             \
    SHA2_WRITE_ARRAY(V, row, Sha2BlockHasherDigestCols, FIELD, VALUES)
#define SHA2_FILL_ZERO_ROUND(V, row, FIELD) SHA2_FILL_ZERO(V, row, Sha2BlockHasherRoundCols, FIELD)
#define SHA2_FILL_ZERO_DIGEST(V, row, FIELD)                                                       \
    SHA2_FILL_ZERO(V, row, Sha2BlockHasherDigestCols, FIELD)
#define SHA2_SLICE_ROUND(V, row, FIELD) SHA2_SLICE_FROM(V, row, Sha2BlockHasherRoundCols, FIELD)
#define SHA2_SLICE_DIGEST(V, row, FIELD) SHA2_SLICE_FROM(V, row, Sha2BlockHasherDigestCols, FIELD)

#define SHA2INNER_WRITE_ROUND(V, row, FIELD, VALUE)                                                \
    SHA2_WRITE_VALUE(V, row, Sha2RoundCols, FIELD, VALUE)
#define SHA2INNER_WRITE_DIGEST(V, row, FIELD, VALUE)                                               \
    SHA2_WRITE_VALUE(V, row, Sha2DigestCols, FIELD, VALUE)
#define SHA2INNER_WRITE_ARRAY_ROUND(V, row, FIELD, VALUES)                                         \
    SHA2_WRITE_ARRAY(V, row, Sha2RoundCols, FIELD, VALUES)
#define SHA2INNER_WRITE_ARRAY_DIGEST(V, row, FIELD, VALUES)                                        \
    SHA2_WRITE_ARRAY(V, row, Sha2DigestCols, FIELD, VALUES)
#define SHA2INNER_FILL_ZERO_ROUND(V, row, FIELD) SHA2_FILL_ZERO(V, row, Sha2RoundCols, FIELD)
#define SHA2INNER_FILL_ZERO_DIGEST(V, row, FIELD) SHA2_FILL_ZERO(V, row, Sha2DigestCols, FIELD)
