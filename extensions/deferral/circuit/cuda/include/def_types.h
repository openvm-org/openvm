#pragma once

#include <cstddef>
#include <cstdint>

#include "fp.h"

namespace deferral {

inline constexpr size_t DIGEST_SIZE = 8;
inline constexpr size_t F_NUM_BYTES = 4;
inline constexpr size_t COMMIT_NUM_BYTES = DIGEST_SIZE * F_NUM_BYTES;
inline constexpr size_t OUTPUT_LEN_NUM_BYTES = 8;
inline constexpr size_t OUTPUT_TOTAL_BYTES = COMMIT_NUM_BYTES + OUTPUT_LEN_NUM_BYTES;

inline constexpr size_t MEMORY_OP_SIZE = 8;
inline constexpr size_t DIGEST_MEMORY_OPS = DIGEST_SIZE / MEMORY_OP_SIZE;
inline constexpr size_t COMMIT_MEMORY_OPS = COMMIT_NUM_BYTES / MEMORY_OP_SIZE;
inline constexpr size_t OUTPUT_TOTAL_MEMORY_OPS = OUTPUT_TOTAL_BYTES / MEMORY_OP_SIZE;

inline constexpr uint32_t BABY_BEAR_ORDER = Fp::P;
inline constexpr uint8_t BABY_BEAR_ORDER_BE[F_NUM_BYTES] = {
    static_cast<uint8_t>((BABY_BEAR_ORDER >> 24) & 0xff),
    static_cast<uint8_t>((BABY_BEAR_ORDER >> 16) & 0xff),
    static_cast<uint8_t>((BABY_BEAR_ORDER >> 8) & 0xff),
    static_cast<uint8_t>(BABY_BEAR_ORDER & 0xff),
};

} // namespace deferral
