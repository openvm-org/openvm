#pragma once

#include "primitives/constants.h"
#include "primitives/execution.h"
#include "system/memory/offline_checker.cuh"
#include "variant.cuh"

using namespace riscv;

namespace sha2 {

template <typename V, typename T> struct Sha2MainBlockCols {
    T request_id;
    T message_bytes[V::BLOCK_U8S];
    T prev_state[V::STATE_BYTES];
    T new_state[V::STATE_BYTES];
};

template <typename T> struct Sha2MainInstructionCols {
    T is_enabled;
    ExecutionState<T> from_state;
    T dst_reg_ptr;
    T state_reg_ptr;
    T input_reg_ptr;
    T dst_ptr_limbs[RV32_REGISTER_NUM_LIMBS];
    T state_ptr_limbs[RV32_REGISTER_NUM_LIMBS];
    T input_ptr_limbs[RV32_REGISTER_NUM_LIMBS];
};

template <typename V, typename T> struct Sha2MainMemoryCols {
    MemoryReadAuxCols<T> register_aux[sha2::SHA2_REGISTER_READS];
    MemoryReadAuxCols<T> input_reads[V::BLOCK_READS];
    MemoryReadAuxCols<T> state_reads[V::STATE_READS];
    MemoryWriteAuxCols<T, sha2::SHA2_WRITE_SIZE> write_aux[V::STATE_WRITES];
};

template <typename V, typename T> struct Sha2MainCols {
    Sha2MainBlockCols<V, T> block;
    Sha2MainInstructionCols<T> instruction;
    Sha2MainMemoryCols<V, T> mem;
};

template <typename V> struct Sha2MainLayout {
    static constexpr size_t WIDTH = sizeof(Sha2MainCols<V, uint8_t>);
};

#define SHA2_MAIN_COL_INDEX_V(V, STRUCT, FIELD)                                                     \
    (reinterpret_cast<size_t>(&(reinterpret_cast<STRUCT<V, uint8_t> *>(0)->FIELD)))
#define SHA2_MAIN_COL_INDEX_PLAIN(STRUCT, FIELD)                                                    \
    (reinterpret_cast<size_t>(&(reinterpret_cast<STRUCT<uint8_t> *>(0)->FIELD)))

#define SHA2_MAIN_COL_ARRAY_LEN_V(V, STRUCT, FIELD)                                                 \
    (sizeof((reinterpret_cast<STRUCT<V, uint8_t> *>(0)->FIELD)))
#define SHA2_MAIN_COL_ARRAY_LEN_PLAIN(STRUCT, FIELD)                                                \
    (sizeof((reinterpret_cast<STRUCT<uint8_t> *>(0)->FIELD)))

#define SHA2_MAIN_WRITE_VALUE_V(V, ROW, STRUCT, FIELD, VALUE)                                       \
    (ROW).write(SHA2_MAIN_COL_INDEX_V(V, STRUCT, FIELD), VALUE)
#define SHA2_MAIN_WRITE_VALUE_PLAIN(ROW, STRUCT, FIELD, VALUE)                                      \
    (ROW).write(SHA2_MAIN_COL_INDEX_PLAIN(STRUCT, FIELD), VALUE)

#define SHA2_MAIN_WRITE_ARRAY_V(V, ROW, STRUCT, FIELD, VALUES)                                      \
    (ROW).write_array(                                                                              \
        SHA2_MAIN_COL_INDEX_V(V, STRUCT, FIELD),                                                    \
        SHA2_MAIN_COL_ARRAY_LEN_V(V, STRUCT, FIELD),                                                \
        VALUES                                                                                      \
    )
#define SHA2_MAIN_WRITE_ARRAY_PLAIN(ROW, STRUCT, FIELD, VALUES)                                     \
    (ROW).write_array(                                                                              \
        SHA2_MAIN_COL_INDEX_PLAIN(STRUCT, FIELD),                                                   \
        SHA2_MAIN_COL_ARRAY_LEN_PLAIN(STRUCT, FIELD),                                               \
        VALUES                                                                                      \
    )

#define SHA2_MAIN_FILL_ZERO_V(V, ROW, STRUCT, FIELD)                                                \
    (ROW).fill_zero(SHA2_MAIN_COL_INDEX_V(V, STRUCT, FIELD), SHA2_MAIN_COL_ARRAY_LEN_V(V, STRUCT, FIELD))
#define SHA2_MAIN_SLICE_FROM_V(V, ROW, STRUCT, FIELD)                                               \
    (ROW).slice_from(SHA2_MAIN_COL_INDEX_V(V, STRUCT, FIELD))

// Compute offset of nested struct field: offsetof(Sha2MainCols, block) + offsetof(Sha2MainBlockCols, FIELD)
#define SHA2_MAIN_COL_INDEX_BLOCK_V(V, FIELD)                                                       \
    (SHA2_MAIN_COL_INDEX_V(V, Sha2MainCols, block) +                                                 \
     SHA2_MAIN_COL_INDEX_V(V, Sha2MainBlockCols, FIELD))
// Compute offset of nested struct field: offsetof(Sha2MainCols, instruction) + offsetof(Sha2MainInstructionCols, FIELD)
#define SHA2_MAIN_COL_INDEX_INSTR(V, FIELD)                                                         \
    (SHA2_MAIN_COL_INDEX_V(V, Sha2MainCols, instruction) +                                         \
     SHA2_MAIN_COL_INDEX_PLAIN(Sha2MainInstructionCols, FIELD))
// Compute offset of nested struct field: offsetof(Sha2MainCols, mem) + offsetof(Sha2MainMemoryCols, FIELD)
#define SHA2_MAIN_COL_INDEX_MEM_V(V, FIELD)                                                          \
    (SHA2_MAIN_COL_INDEX_V(V, Sha2MainCols, mem) +                                                   \
     SHA2_MAIN_COL_INDEX_V(V, Sha2MainMemoryCols, FIELD))

#define SHA2_MAIN_WRITE_BLOCK(V, ROW, FIELD, VALUE)                                                 \
    (ROW).write(SHA2_MAIN_COL_INDEX_BLOCK_V(V, FIELD), VALUE)
#define SHA2_MAIN_WRITE_ARRAY_BLOCK(V, ROW, FIELD, VALUES)                                          \
    (ROW).write_array(                                                                              \
        SHA2_MAIN_COL_INDEX_BLOCK_V(V, FIELD),                                                       \
        SHA2_MAIN_COL_ARRAY_LEN_V(V, Sha2MainBlockCols, FIELD),                                     \
        VALUES                                                                                      \
    )

#define SHA2_MAIN_WRITE_INSTR(V, ROW, FIELD, VALUE)                                                 \
    (ROW).write(SHA2_MAIN_COL_INDEX_INSTR(V, FIELD), VALUE)
#define SHA2_MAIN_WRITE_ARRAY_INSTR(V, ROW, FIELD, VALUES)                                          \
    (ROW).write_array(                                                                              \
        SHA2_MAIN_COL_INDEX_INSTR(V, FIELD),                                                         \
        SHA2_MAIN_COL_ARRAY_LEN_PLAIN(Sha2MainInstructionCols, FIELD),                             \
        VALUES                                                                                      \
    )

#define SHA2_MAIN_SLICE_MEM(V, ROW, FIELD)                                                          \
    (ROW).slice_from(SHA2_MAIN_COL_INDEX_MEM_V(V, FIELD))

} // namespace sha2
