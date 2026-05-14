#pragma once

#include "p3_keccakf.cuh"
#include "primitives/constants.h"
#include "system/memory/offline_checker.cuh"

#include <cstddef>
#include <cstdint>

namespace keccakf_op {
using namespace riscv;
using namespace keccak256;

inline constexpr size_t NUM_OP_ROWS_PER_INS = 1; // 1 row per instruction

// Record structure matching Rust KeccakfRecord (from trace.rs)
// Must match exact layout with #[repr(C)]
struct KeccakfOpRecord {
    uint32_t pc;
    uint32_t timestamp;
    uint32_t rd_ptr;
    uint32_t buffer_ptr;
    MemoryReadAuxRecord rd_aux;
    MemoryReadAuxRecord buffer_word_aux[KECCAK_WIDTH_MEM_OPS];
    uint8_t preimage_buffer_bytes[KECCAK_WIDTH_BYTES];
};

// Number of u16 cells used to represent the low 32 bits of `buffer_ptr`. Matches the
// Rust `BUFFER_PTR_NUM_LIMBS` constant.
inline constexpr size_t BUFFER_PTR_NUM_LIMBS = RV64_WORD_NUM_LIMBS / 2;

// Column structure matching Rust KeccakfOpCols (from columns.rs)
template <typename T> struct KeccakfOpCols {
    T pc;
    T is_valid;
    T timestamp;
    T rd_ptr;
    // Low 32 bits of `buffer_ptr` packed into 2 u16 cells; the high 32 bits of the
    // RV64 register are zero and hardcoded in the memory bus interaction.
    T buffer_ptr_limbs[BUFFER_PTR_NUM_LIMBS];
    // `preimage` / `postimage` stored as u16 cells (one per pair of state bytes),
    // matching the keccakf periphery bus and AS2 u16-celled memory.
    T preimage[KECCAK_WIDTH_U16S];
    T postimage[KECCAK_WIDTH_U16S];
    MemoryReadAuxCols<T> rd_aux;
    MemoryBaseAuxCols<T> buffer_word_aux[KECCAK_WIDTH_MEM_OPS]; // 25 words
};

inline constexpr size_t NUM_KECCAKF_OP_COLS = sizeof(KeccakfOpCols<uint8_t>);

// Compute keccak-f permutation on a 200-byte state.
// Delegates to the shared round body in keccak256::keccakf_round_body;
// __forceinline__ so the full permutation folds into the caller's frame.
__device__ __forceinline__ void keccakf_permutation(uint64_t state[25]) {
    for (int round = 0; round < 24; round++) {
        keccak256::keccakf_round_body(state, round);
    }
}

} // namespace keccakf_op
