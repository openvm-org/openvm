#pragma once

#include <cstddef>
#include <cstdint>

#include "fp.h"
#include "primitives/constants.h"
#include "system/memory/params.cuh"

namespace deferral {

using openvm::U16_BITS;
using riscv::RV64_WORD_NUM_LIMBS;

inline constexpr size_t DIGEST_SIZE = 8;
inline constexpr size_t F_NUM_BYTES = sizeof(uint32_t);
inline constexpr size_t COMMIT_NUM_BYTES = DIGEST_SIZE * F_NUM_BYTES;
inline constexpr size_t OUTPUT_LEN_NUM_BYTES = sizeof(uint64_t);
inline constexpr size_t OUTPUT_TOTAL_BYTES = COMMIT_NUM_BYTES + OUTPUT_LEN_NUM_BYTES;
inline constexpr uint32_t U16_MASK = (1u << U16_BITS) - 1;
inline constexpr size_t RV64_PTR_U16S = RV64_WORD_NUM_LIMBS / U16_CELL_SIZE;

// u16 cell-shape constants for the packed column/key layout.
inline constexpr size_t F_NUM_U16S = F_NUM_BYTES / U16_CELL_SIZE;
inline constexpr size_t COMMIT_NUM_U16S = DIGEST_SIZE * F_NUM_U16S;
// OutputKey length field width after zero-padding the low 32-bit length.
inline constexpr size_t OUTPUT_LEN_NUM_U16S = OUTPUT_LEN_NUM_BYTES / U16_CELL_SIZE;
inline constexpr size_t OUTPUT_TOTAL_NUM_U16S = COMMIT_NUM_U16S + OUTPUT_LEN_NUM_U16S;

// Memory-op counts for byte-addressed AS (RV64_MEMORY_AS): each bus message
// carries `BLOCK_FE_WIDTH` u16 cells = `MEMORY_BLOCK_BYTES` bytes.
inline constexpr size_t COMMIT_MEMORY_OPS = COMMIT_NUM_U16S / BLOCK_FE_WIDTH;
inline constexpr size_t OUTPUT_TOTAL_MEMORY_OPS = OUTPUT_TOTAL_NUM_U16S / BLOCK_FE_WIDTH;

// Guest bytes absorbed per deferral output Poseidon2 row.
inline constexpr size_t SPONGE_BYTES_PER_ROW = U16_CELL_SIZE * DIGEST_SIZE;
inline constexpr size_t SPONGE_ROW_MEMORY_OPS = SPONGE_BYTES_PER_ROW / MEMORY_BLOCK_BYTES;

// Memory-op count for F-celled DEFERRAL_AS.
inline constexpr size_t DIGEST_F_MEMORY_OPS = DIGEST_SIZE / BLOCK_FE_WIDTH;

// BabyBear modulus used by the canonicity sub-AIR.
inline constexpr uint32_t BABY_BEAR_ORDER = Fp::P;

__device__ __host__ inline uint16_t u16_from_bytes_le(uint8_t lo, uint8_t hi) {
    return static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
}

template <typename T, size_t OUT, size_t BYTES>
__device__ __host__ inline void pack_u8_pairs_le(T (&out)[OUT], uint8_t const (&bytes)[BYTES]) {
    static_assert(
        BYTES == U16_CELL_SIZE * OUT,
        "pack_u8_pairs_le expects exactly one u16 cell worth of bytes per output"
    );
    for (size_t i = 0; i < OUT; ++i) {
        const size_t offset = U16_CELL_SIZE * i;
        out[i] =
            T(static_cast<uint32_t>(u16_from_bytes_le(bytes[offset], bytes[offset + 1])));
    }
}

template <typename T>
__device__ __host__ inline void u32_bytes_to_le_u16_cells(
    T (&out)[RV64_PTR_U16S],
    uint8_t const (&bytes)[RV64_WORD_NUM_LIMBS]
) {
    pack_u8_pairs_le(out, bytes);
}

template <typename T, size_t OUT>
__device__ __host__ inline void u32_to_le_u16_cells(T (&out)[OUT], uint32_t value) {
    static_assert(OUT * U16_CELL_SIZE == sizeof(uint32_t));
    for (size_t i = 0; i < OUT; ++i) {
        out[i] = T(static_cast<uint32_t>((value >> (U16_BITS * i)) & U16_MASK));
    }
}

template <size_t NUM_U16_CELLS>
__device__ __host__ inline uint32_t scale_u16_high_cell(
    uint32_t high_u16,
    size_t address_bits
) {
    return high_u16 << static_cast<uint32_t>(U16_BITS * NUM_U16_CELLS - address_bits);
}

__device__ __host__ inline uint32_t scale_output_len(
    uint16_t const (&output_len)[F_NUM_U16S],
    size_t address_bits
) {
    return scale_u16_high_cell<F_NUM_U16S>(
        static_cast<uint32_t>(output_len[F_NUM_U16S - 1]), address_bits
    );
}

__device__ __host__ inline uint32_t scale_rv64_ptr_from_u32_bytes(
    uint8_t const (&bytes)[RV64_WORD_NUM_LIMBS],
    size_t address_bits
) {
    return scale_u16_high_cell<RV64_PTR_U16S>(
        static_cast<uint32_t>(u16_from_bytes_le(bytes[2], bytes[3])), address_bits
    );
}

} // namespace deferral
