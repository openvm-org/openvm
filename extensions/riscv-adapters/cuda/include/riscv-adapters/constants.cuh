#pragma once

#include <cstddef>
#include <cstdint>

constexpr size_t RV64_LOW32_U16_LIMBS = sizeof(uint32_t) / sizeof(uint16_t);
constexpr size_t RV64_LOW32_HIGH_U16_LIMBS = RV64_LOW32_U16_LIMBS - 1;
constexpr size_t RV64_U16_LIMB_BITS = 8 * sizeof(uint16_t);
constexpr size_t RV64_LOW32_BITS = RV64_LOW32_U16_LIMBS * RV64_U16_LIMB_BITS;
constexpr uint32_t RV64_U16_LIMB_MASK = (1u << RV64_U16_LIMB_BITS) - 1;
constexpr size_t RV64_U16_SIGN_BIT = RV64_U16_LIMB_BITS - 1;
