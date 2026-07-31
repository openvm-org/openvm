#pragma once

#include <cstddef>
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
    // Low 32 bits of [buffer_reg_ptr:8]_1 as u16 cells.
    T buffer_ptr_limbs[RV64_PTR_U16_LIMBS];
    // Low 32 bits of [input_reg_ptr:8]_1 as u16 cells.
    T input_ptr_limbs[RV64_PTR_U16_LIMBS];
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
    MemoryBaseAuxCols<T> buffer_bytes_write_base_aux[keccak256::KECCAK_RATE_MEM_OPS];
};

template <typename T>
struct XorinVmCols {
    XorinSpongeCols<T> sponge;
    XorinInstructionCols<T> instruction;
    XorinMemoryCols<T> mem_oc;
};

} // namespace xorin
