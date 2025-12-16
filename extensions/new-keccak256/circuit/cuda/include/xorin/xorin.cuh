#pragma once

#include <cstddef>
#include <cstdint>

namespace xorin {

template <typename T>
struct XorinInstructionCols {
    T pc;
    T is_enabled;
    T buffer_ptr;
    T input_ptr;
    T len_ptr;
    T buffer;
    T buffer_limbs[4];
    T input;
    T input_limbs[4];
    T len;
    T len_limbs[4];
    T start_timestamp;
};

template <typename T>
struct XorinSpongeCols {
    T is_padding_bytes[34];  // 136 / 4 = 34
    T preimage_buffer_bytes[136];
    T input_bytes[136];
    T postimage_buffer_bytes[136];
};

template <typename T>
struct XorinMemoryCols {
    T register_aux_cols[3];  // Simplified for now, actual structure is more complex
    T input_bytes_read_aux_cols[34];
    T buffer_bytes_read_aux_cols[34];
    T buffer_bytes_write_aux_cols[34];
};

template <typename T>
struct XorinVmCols {
    XorinSpongeCols<T> sponge;
    XorinInstructionCols<T> instruction;
    XorinMemoryCols<T> mem_oc;
};

// Record structure matching Rust's XorinVmRecordHeader
struct XorinVmRecord {
    uint32_t from_pc;
    uint32_t timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint32_t rs2_ptr;
    uint32_t buffer;
    uint32_t input;
    uint32_t len;
    uint8_t buffer_limbs[136];
    uint8_t input_limbs[136];
    // Memory auxiliary columns - simplified for now
    uint32_t register_aux_timestamps[3];
    uint32_t input_read_aux_timestamps[34];
    uint32_t buffer_read_aux_timestamps[34];
    uint32_t buffer_write_aux_timestamps[34];
    uint8_t buffer_write_prev_data[136];  // 34 * 4 bytes
};

} // namespace xorin