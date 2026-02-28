#pragma once

#include <cstddef>
#include <cstdint>
#include "primitives/constants.h"
#include "system/memory/offline_checker.cuh"

namespace xorin {

inline constexpr size_t XORIN_RATE_BYTES = 136;
inline constexpr size_t XORIN_WORD_SIZE = riscv::RV32_REGISTER_NUM_LIMBS;
inline constexpr size_t XORIN_NUM_WORDS = XORIN_RATE_BYTES / XORIN_WORD_SIZE;
inline constexpr size_t XORIN_REGISTER_READS = 3;

template <typename T>
struct XorinInstructionCols {
    T pc;
    T is_enabled;
    T buffer_reg_ptr;
    T input_reg_ptr;
    T len_reg_ptr;
    T buffer_ptr;
    T buffer_ptr_limbs[XORIN_WORD_SIZE];
    T input_ptr;
    T input_ptr_limbs[XORIN_WORD_SIZE];
    T len;
    T len_limbs[XORIN_WORD_SIZE];
    T start_timestamp;
};

template <typename T>
struct XorinSpongeCols {
    T is_padding_bytes[XORIN_NUM_WORDS];
    T preimage_buffer_bytes[XORIN_RATE_BYTES];
    T input_bytes[XORIN_RATE_BYTES];
    T postimage_buffer_bytes[XORIN_RATE_BYTES];
};

template <typename T>
struct XorinMemoryCols {
    MemoryReadAuxCols<T> register_aux_cols[XORIN_REGISTER_READS];
    MemoryReadAuxCols<T> input_bytes_read_aux_cols[XORIN_NUM_WORDS];
    MemoryReadAuxCols<T> buffer_bytes_read_aux_cols[XORIN_NUM_WORDS];
    MemoryWriteAuxCols<T, XORIN_WORD_SIZE> buffer_bytes_write_aux_cols[XORIN_NUM_WORDS];
};

template <typename T>
struct XorinVmCols {
    XorinSpongeCols<T> sponge;
    XorinInstructionCols<T> instruction;
    XorinMemoryCols<T> mem_oc;
};

struct XorinVmRecord {
    uint32_t from_pc;
    uint32_t timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint32_t rs2_ptr;
    uint32_t buffer;
    uint32_t input;
    uint32_t len;
    uint8_t buffer_limbs[XORIN_RATE_BYTES];
    uint8_t input_limbs[XORIN_RATE_BYTES];
    MemoryReadAuxRecord register_aux_cols[XORIN_REGISTER_READS];
    MemoryReadAuxRecord input_read_aux_cols[XORIN_NUM_WORDS];
    MemoryReadAuxRecord buffer_read_aux_cols[XORIN_NUM_WORDS];
    MemoryWriteBytesAuxRecord<XORIN_WORD_SIZE> buffer_write_aux_cols[XORIN_NUM_WORDS];
};

} // namespace xorin