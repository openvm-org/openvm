#pragma once

#include "p3_keccakf.cuh"

#include <cstddef>
#include <cstdint>

namespace keccakf_perm {

// KeccakfPermCols = KeccakCols + timestamp
// Matches Rust KeccakfPermCols (from keccakf_perm/air.rs)
template <typename T> struct KeccakfPermCols {
    p3_keccak_air::KeccakCols<T> inner;
    T timestamp;
};

inline constexpr size_t NUM_KECCAKF_PERM_COLS = sizeof(KeccakfPermCols<uint8_t>);

} // namespace keccakf_perm
