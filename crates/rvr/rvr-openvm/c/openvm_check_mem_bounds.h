#ifndef OPENVM_CHECK_MEM_BOUNDS_H
#define OPENVM_CHECK_MEM_BOUNDS_H
#include <stddef.h>
#include <stdint.h>

#include "openvm_constants.h"
#include "openvm_util.h"

__attribute__((noreturn, cold))
void abort_oob(uint32_t start, size_t size, size_t mem_size);

static __attribute__((always_inline)) inline void check_mem_bounds_u8(uint32_t start) {
  if (unlikely(start > (size_t)MEMORY_MASK + 1u - sizeof(uint8_t))) {
    abort_oob(start, sizeof(uint8_t), (size_t)MEMORY_MASK + 1u);
  }
}

static __attribute__((always_inline)) inline void check_mem_bounds_i8(uint32_t start) {
  if (unlikely(start > (size_t)MEMORY_MASK + 1u - sizeof(int8_t))) {
    abort_oob(start, sizeof(int8_t), (size_t)MEMORY_MASK + 1u);
  }
}

static __attribute__((always_inline)) inline void check_mem_bounds_u16(uint32_t start) {
  if (unlikely(start > (size_t)MEMORY_MASK + 1u - sizeof(uint16_t))) {
    abort_oob(start, sizeof(uint16_t), (size_t)MEMORY_MASK + 1u);
  }
}

static __attribute__((always_inline)) inline void check_mem_bounds_i16(uint32_t start) {
  if (unlikely(start > (size_t)MEMORY_MASK + 1u - sizeof(int16_t))) {
    abort_oob(start, sizeof(int16_t), (size_t)MEMORY_MASK + 1u);
  }
}

static __attribute__((always_inline)) inline void check_mem_bounds_u32(uint32_t start) {
  if (unlikely(start > (size_t)MEMORY_MASK + 1u - sizeof(uint32_t))) {
    abort_oob(start, sizeof(uint32_t), (size_t)MEMORY_MASK + 1u);
  }
}

static __attribute__((always_inline)) inline void check_mem_bounds_range(uint32_t start, size_t size) {
  const size_t mem_size = (size_t)MEMORY_MASK + 1u;
  if (unlikely(start > mem_size || size > mem_size - start)) {
    abort_oob(start, size, mem_size);
  }
}

static __attribute__((always_inline)) inline void check_mem_bounds_u32_range(uint32_t base_addr, uint32_t num_words) {
  check_mem_bounds_range(base_addr, (size_t)num_words * sizeof(uint32_t));
}

#endif
