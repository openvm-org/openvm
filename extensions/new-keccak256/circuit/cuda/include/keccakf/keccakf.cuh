#pragma once

#include <cstddef>
#include <cstdint>
#include "primitives/constants.h"
#include "system/memory/offline_checker.cuh"

namespace keccakf {

inline constexpr size_t NUM_ROUNDS = 24;
inline constexpr size_t KECCAK_STATE_BYTES = 200;
inline constexpr size_t KECCAK_STATE_U64 = 25;
inline constexpr size_t KECCAK_STATE_U16 = 100;
inline constexpr size_t NUM_BUFFER_WORDS = KECCAK_STATE_BYTES / riscv::RV32_REGISTER_NUM_LIMBS;
inline constexpr size_t U64_LIMBS = 4;

// KeccakPermCols from p3_keccak_air - must match exactly
template <typename T>
struct KeccakPermCols {
    T step_flags[NUM_ROUNDS];
    T _export;
    T preimage[5][5][U64_LIMBS];
    T a[5][5][U64_LIMBS];
    T c[5][64];
    T c_prime[5][64];
    T a_prime[5][5][64];
    T a_prime_prime[5][5][U64_LIMBS];
    T a_prime_prime_0_0_bits[64];
    T a_prime_prime_prime_0_0_limbs[U64_LIMBS];
};

template <typename T>
struct KeccakfInstructionCols {
    T pc;
    T is_enabled;
    T rd_ptr;
    T buffer_ptr;
    T buffer_ptr_limbs[riscv::RV32_REGISTER_NUM_LIMBS];
};

template <typename T>
struct KeccakfMemoryCols {
    MemoryReadAuxCols<T> register_aux_cols[1];
    MemoryReadAuxCols<T> buffer_bytes_read_aux_cols[NUM_BUFFER_WORDS];
    MemoryWriteAuxCols<T, riscv::RV32_REGISTER_NUM_LIMBS> buffer_bytes_write_aux_cols[NUM_BUFFER_WORDS];
};

template <typename T>
struct KeccakfVmCols {
    KeccakPermCols<T> inner;
    T preimage_state_hi[KECCAK_STATE_U16];
    T postimage_state_hi[KECCAK_STATE_U16];
    KeccakfInstructionCols<T> instruction;
    KeccakfMemoryCols<T> mem_oc;
    T timestamp;
    T is_enabled_is_first_round;
    T is_enabled_is_final_round;
};

// Record structure matching Rust's KeccakfVmRecordHeader
struct KeccakfVmRecord {
    uint32_t pc;
    uint32_t timestamp;
    uint32_t buffer;
    uint8_t preimage_buffer_bytes[KECCAK_STATE_BYTES];
    uint32_t rd_ptr;
    MemoryReadAuxRecord register_aux_cols[1];
    MemoryReadAuxRecord buffer_read_aux_cols[NUM_BUFFER_WORDS];
    MemoryWriteBytesAuxRecord<riscv::RV32_REGISTER_NUM_LIMBS> buffer_write_aux_cols[NUM_BUFFER_WORDS];
};

inline constexpr size_t NUM_KECCAKF_VM_COLS = sizeof(KeccakfVmCols<uint8_t>);
inline constexpr size_t NUM_KECCAK_PERM_COLS = sizeof(KeccakPermCols<uint8_t>);

} // namespace keccakf
