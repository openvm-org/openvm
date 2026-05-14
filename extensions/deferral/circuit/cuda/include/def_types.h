#pragma once

#include <cstddef>
#include <cstdint>

#include "system/memory/params.cuh"

namespace deferral {

inline constexpr size_t DIGEST_SIZE = 8;
inline constexpr size_t F_NUM_BYTES = 4;
inline constexpr size_t COMMIT_NUM_BYTES = DIGEST_SIZE * F_NUM_BYTES;
inline constexpr size_t OUTPUT_LEN_NUM_BYTES = 8;
inline constexpr size_t OUTPUT_TOTAL_BYTES = COMMIT_NUM_BYTES + OUTPUT_LEN_NUM_BYTES;

// u16 cell-shape constants for the column layout (matches CPU-side
// `F_NUM_U16S`, `COMMIT_NUM_U16S`, `OUTPUT_LEN_NUM_U16S`).
inline constexpr size_t F_NUM_U16S = 2;
inline constexpr size_t COMMIT_NUM_U16S = DIGEST_SIZE * F_NUM_U16S;
inline constexpr size_t OUTPUT_LEN_NUM_U16S = F_NUM_U16S;

// Memory-op counts for byte-addressed AS (RV64_MEMORY_AS): each bus message
// covers `MEMORY_BLOCK_BYTES` bytes.
inline constexpr size_t DIGEST_BYTE_MEMORY_OPS = DIGEST_SIZE / MEMORY_BLOCK_BYTES;
inline constexpr size_t COMMIT_MEMORY_OPS = COMMIT_NUM_BYTES / MEMORY_BLOCK_BYTES;
inline constexpr size_t OUTPUT_TOTAL_MEMORY_OPS = OUTPUT_TOTAL_BYTES / MEMORY_BLOCK_BYTES;

// Memory-op count for F-celled DEFERRAL_AS: each bus message covers
// `BLOCK_FE_WIDTH` cells. Mirrors the CPU-side `DIGEST_F_MEMORY_OPS`.
inline constexpr size_t DIGEST_F_MEMORY_OPS = DIGEST_SIZE / BLOCK_FE_WIDTH;

} // namespace deferral
