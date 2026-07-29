#ifndef OPENVM_RVR_RV64B_H
#define OPENVM_RVR_RV64B_H

#include <stdint.h>

static __attribute__((always_inline)) inline uint64_t rv64b_sext8(uint8_t value) {
  return (uint64_t)(int64_t)(int8_t)value;
}

static __attribute__((always_inline)) inline uint64_t rv64b_sext16(uint16_t value) {
  return (uint64_t)(int64_t)(int16_t)value;
}

static __attribute__((always_inline)) inline uint64_t rv64b_sext32(uint32_t value) {
  return (uint64_t)(int64_t)(int32_t)value;
}

static __attribute__((always_inline)) inline uint64_t rv64b_rol64(uint64_t value,
                                                                  uint64_t shift) {
  uint32_t amount = (uint32_t)shift & 0x3fu;
  return (value << amount) | (value >> ((64u - amount) & 0x3fu));
}

static __attribute__((always_inline)) inline uint64_t rv64b_ror64(uint64_t value,
                                                                  uint64_t shift) {
  uint32_t amount = (uint32_t)shift & 0x3fu;
  return (value >> amount) | (value << ((64u - amount) & 0x3fu));
}

static __attribute__((always_inline)) inline uint32_t rv64b_rol32(uint32_t value,
                                                                  uint64_t shift) {
  uint32_t amount = (uint32_t)shift & 0x1fu;
  return (uint32_t)((value << amount) | (value >> ((32u - amount) & 0x1fu)));
}

static __attribute__((always_inline)) inline uint32_t rv64b_ror32(uint32_t value,
                                                                  uint64_t shift) {
  uint32_t amount = (uint32_t)shift & 0x1fu;
  return (uint32_t)((value >> amount) | (value << ((32u - amount) & 0x1fu)));
}

static __attribute__((always_inline)) inline uint64_t rv64b_clz64(uint64_t value) {
  return value == 0 ? 64ull : (uint64_t)__builtin_clzll(value);
}

static __attribute__((always_inline)) inline uint64_t rv64b_ctz64(uint64_t value) {
  return value == 0 ? 64ull : (uint64_t)__builtin_ctzll(value);
}

static __attribute__((always_inline)) inline uint64_t rv64b_clz32(uint32_t value) {
  return value == 0 ? 32ull : (uint64_t)__builtin_clz(value);
}

static __attribute__((always_inline)) inline uint64_t rv64b_ctz32(uint32_t value) {
  return value == 0 ? 32ull : (uint64_t)__builtin_ctz(value);
}

static __attribute__((always_inline)) inline uint64_t rv64b_cpop64(uint64_t value) {
  return (uint64_t)__builtin_popcountll(value);
}

static __attribute__((always_inline)) inline uint64_t rv64b_cpop32(uint32_t value) {
  return (uint64_t)__builtin_popcount(value);
}

static __attribute__((always_inline)) inline uint64_t rv64b_orc_b(uint64_t value) {
  uint64_t out = 0;
  for (uint32_t i = 0; i < 8; ++i) {
    uint64_t byte = (value >> (8u * i)) & 0xffu;
    if (byte != 0) {
      out |= 0xffull << (8u * i);
    }
  }
  return out;
}

static __attribute__((always_inline)) inline uint64_t rv64b_rev8(uint64_t value) {
  return __builtin_bswap64(value);
}

#endif /* OPENVM_RVR_RV64B_H */
