#pragma once

#include <cstddef>
#include <cstdint>
#include "primitives/constants.h"
#include "system/memory/offline_checker.cuh"

namespace xorin {

inline constexpr size_t XORIN_RATE_BYTES = keccak256::KECCAK_RATE_BYTES;
inline constexpr size_t XORIN_REGISTER_READS = 3;

template <typename T>
struct XorinInstructionCols {
    T pc;
    T is_enabled;
    T buffer_reg_ptr;
    T input_reg_ptr;
    T len_reg_ptr;
    T buffer_ptr;
    T buffer_ptr_limbs[riscv::RV64_WORD_NUM_LIMBS];
    T input_ptr;
    T input_ptr_limbs[riscv::RV64_WORD_NUM_LIMBS];
    T len;
    T len_limb;
    T start_timestamp;
};

template <typename T>
struct XorinSpongeCols {
    T is_padding_bytes[keccak256::KECCAK_RATE_MEM_OPS];
    T preimage_buffer_bytes[XORIN_RATE_BYTES];
    T input_bytes[XORIN_RATE_BYTES];
    T postimage_buffer_bytes[XORIN_RATE_BYTES];
};

template <typename T>
struct XorinMemoryCols {
    MemoryReadAuxCols<T> register_aux_cols[XORIN_REGISTER_READS];
    MemoryReadAuxCols<T> input_bytes_read_aux_cols[keccak256::KECCAK_RATE_MEM_OPS];
    MemoryReadAuxCols<T> buffer_bytes_read_aux_cols[keccak256::KECCAK_RATE_MEM_OPS];
    MemoryWriteAuxCols<T, program::DEFAULT_BLOCK_SIZE>
        buffer_bytes_write_aux_cols[keccak256::KECCAK_RATE_MEM_OPS];
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
    MemoryReadAuxRecord input_read_aux_cols[keccak256::KECCAK_RATE_MEM_OPS];
    MemoryReadAuxRecord buffer_read_aux_cols[keccak256::KECCAK_RATE_MEM_OPS];
    MemoryWriteBytesAuxRecord<program::DEFAULT_BLOCK_SIZE>
        buffer_write_aux_cols[keccak256::KECCAK_RATE_MEM_OPS];
};

} // namespace xorin
