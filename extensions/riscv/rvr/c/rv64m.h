#ifndef OPENVM_RVR_RV64M_H
#define OPENVM_RVR_RV64M_H

#include <stdint.h>

#include "openvm_state.h"

static __attribute__((always_inline)) inline uint64_t rv_div(int64_t a,
                                                             int64_t b) {
  if (unlikely(b == 0)) {
    return 0xffffffffffffffffull;
  }
  if (unlikely(a == (int64_t)0x8000000000000000ull && b == -1)) {
    return (uint64_t)a;
  }
  return (uint64_t)(a / b);
}

static __attribute__((always_inline)) inline uint64_t rv_divu(uint64_t a,
                                                              uint64_t b) {
  if (unlikely(b == 0)) {
    return 0xffffffffffffffffull;
  }
  return a / b;
}

static __attribute__((always_inline)) inline uint64_t rv_rem(int64_t a,
                                                             int64_t b) {
  if (unlikely(b == 0)) {
    return (uint64_t)a;
  }
  if (unlikely(a == (int64_t)0x8000000000000000ull && b == -1)) {
    return 0;
  }
  return (uint64_t)(a % b);
}

static __attribute__((always_inline)) inline uint64_t rv_remu(uint64_t a,
                                                              uint64_t b) {
  if (unlikely(b == 0)) {
    return a;
  }
  return a % b;
}

static __attribute__((always_inline)) inline uint64_t rv_mulh(int64_t a,
                                                              int64_t b) {
  return (uint64_t)(((__int128)a * b) >> 64);
}

static __attribute__((always_inline)) inline uint64_t rv_mulhu(uint64_t a,
                                                               uint64_t b) {
  return (uint64_t)(((unsigned __int128)a * b) >> 64);
}

static __attribute__((always_inline)) inline uint64_t rv_mulhsu(int64_t a,
                                                                uint64_t b) {
  return (uint64_t)(((__int128)a * (__int128)b) >> 64);
}

static __attribute__((always_inline)) inline uint64_t rv_divw(int32_t a,
                                                              int32_t b) {
  if (unlikely(b == 0)) {
    return (uint64_t)(int64_t)(int32_t)0xffffffffu;
  }
  if (unlikely(a == (int32_t)0x80000000 && b == -1)) {
    return (uint64_t)(int64_t)a;
  }
  return (uint64_t)(int64_t)(a / b);
}

static __attribute__((always_inline)) inline uint64_t rv_divuw(uint32_t a,
                                                               uint32_t b) {
  if (unlikely(b == 0)) {
    return (uint64_t)(int64_t)(int32_t)0xffffffffu;
  }
  return (uint64_t)(int64_t)(int32_t)(a / b);
}

static __attribute__((always_inline)) inline uint64_t rv_remw(int32_t a,
                                                              int32_t b) {
  if (unlikely(b == 0)) {
    return (uint64_t)(int64_t)a;
  }
  if (unlikely(a == (int32_t)0x80000000 && b == -1)) {
    return 0;
  }
  return (uint64_t)(int64_t)(a % b);
}

static __attribute__((always_inline)) inline uint64_t rv_remuw(uint32_t a,
                                                               uint32_t b) {
  if (unlikely(b == 0)) {
    return (uint64_t)(int64_t)(int32_t)a;
  }
  return (uint64_t)(int64_t)(int32_t)(a % b);
}

#endif
