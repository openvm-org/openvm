#ifndef RV_MULDIV_H
#define RV_MULDIV_H

#include <stdint.h>

#include "openvm_state.h"

/* RISC-V M-extension helpers with spec-compliant division semantics.
   Division by zero returns all-ones (unsigned) or the dividend (remainder).
   Signed overflow (INT32_MIN / -1) returns INT32_MIN (div) or 0 (rem). */

static __attribute__((always_inline)) inline uint32_t rv_div(int32_t a, int32_t b) {
  if (unlikely(b == 0)) {
    return 0xffffffffu;
  }
  if (unlikely(a == (int32_t)0x80000000 && b == -1)) {
    return 0x80000000u;
  }
  return (uint32_t)(a / b);
}

static __attribute__((always_inline)) inline uint32_t rv_divu(uint32_t a, uint32_t b) {
  if (unlikely(b == 0)) {
    return 0xffffffffu;
  }
  return a / b;
}

static __attribute__((always_inline)) inline uint32_t rv_rem(int32_t a, int32_t b) {
  if (unlikely(b == 0)) {
    return (uint32_t)a;
  }
  if (unlikely(a == (int32_t)0x80000000 && b == -1)) {
    return 0;
  }
  return (uint32_t)(a % b);
}

static __attribute__((always_inline)) inline uint32_t rv_remu(uint32_t a, uint32_t b) {
  if (unlikely(b == 0)) {
    return a;
  }
  return a % b;
}

#endif /* RV_MULDIV_H */
