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
    // Low 32 bits of the buffer register as u16 cells.
    T buffer_ptr_limbs[PTR_U16_LIMBS];
    // Low 32 bits of the input register as u16 cells.
    T input_ptr_limbs[PTR_U16_LIMBS];
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
    // Carry for converting the base `buffer`/`input` *byte* pointers to AS-native u16 *cell*
    // pointer limbs.
    T buffer_cell_carry;
    T input_cell_carry;
    // Per-block carry for adding the cell offset `i * (MEMORY_BLOCK_BYTES / U16_CELL_SIZE)` to each
    // base cell pointer (block `i`'s carry into the high cell limb). One set per heap access group
    // (buffer read, input read, buffer write).
    T buffer_read_add_carry[keccak256::KECCAK_RATE_MEM_OPS];
    T input_read_add_carry[keccak256::KECCAK_RATE_MEM_OPS];
    T buffer_write_add_carry[keccak256::KECCAK_RATE_MEM_OPS];
};

template <typename T>
struct XorinVmCols {
    XorinSpongeCols<T> sponge;
    XorinInstructionCols<T> instruction;
    XorinMemoryCols<T> mem_oc;
};

} // namespace xorin
