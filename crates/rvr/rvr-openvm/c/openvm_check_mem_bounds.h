#ifndef OPENVM_CHECK_MEM_BOUNDS_H
#define OPENVM_CHECK_MEM_BOUNDS_H
#include <stddef.h>
#include <stdint.h>

#include "openvm_constants.h"
#include "openvm_util.h"

__attribute__((noreturn, cold)) void abort_oob(uint64_t start, size_t size,
                                               size_t mem_size);

static constexpr size_t OPENVM_MEM_SIZE = (size_t)MEMORY_MASK + 1u;

static __attribute__((always_inline)) inline void check_mem_bounds_u8(
    uint64_t start) {
  if (unlikely(start > OPENVM_MEM_SIZE - sizeof(uint8_t))) {
    abort_oob(start, sizeof(uint8_t), OPENVM_MEM_SIZE);
  }
  assume(start <= OPENVM_MEM_SIZE - sizeof(uint8_t));
}

static __attribute__((always_inline)) inline void check_mem_bounds_i8(
    uint64_t start) {
  if (unlikely(start > OPENVM_MEM_SIZE - sizeof(int8_t))) {
    abort_oob(start, sizeof(int8_t), OPENVM_MEM_SIZE);
  }
  assume(start <= OPENVM_MEM_SIZE - sizeof(int8_t));
}

static __attribute__((always_inline)) inline void check_mem_bounds_u16(
    uint64_t start) {
  if (unlikely(start > OPENVM_MEM_SIZE - sizeof(uint16_t))) {
    abort_oob(start, sizeof(uint16_t), OPENVM_MEM_SIZE);
  }
  assume(start <= OPENVM_MEM_SIZE - sizeof(uint16_t));
}

static __attribute__((always_inline)) inline void check_mem_bounds_i16(
    uint64_t start) {
  if (unlikely(start > OPENVM_MEM_SIZE - sizeof(int16_t))) {
    abort_oob(start, sizeof(int16_t), OPENVM_MEM_SIZE);
  }
  assume(start <= OPENVM_MEM_SIZE - sizeof(int16_t));
}

static __attribute__((always_inline)) inline void check_mem_bounds_u32(
    uint64_t start) {
  if (unlikely(start > OPENVM_MEM_SIZE - sizeof(uint32_t))) {
    abort_oob(start, sizeof(uint32_t), OPENVM_MEM_SIZE);
  }
  assume(start <= OPENVM_MEM_SIZE - sizeof(uint32_t));
}

static __attribute__((always_inline)) inline void check_mem_bounds_u64(
    uint64_t start) {
  if (unlikely(start > OPENVM_MEM_SIZE - sizeof(uint64_t))) {
    abort_oob(start, sizeof(uint64_t), OPENVM_MEM_SIZE);
  }
  assume(start <= OPENVM_MEM_SIZE - sizeof(uint64_t));
}

static __attribute__((always_inline)) inline void check_mem_bounds_range(
    uint64_t start, size_t size) {
  if (unlikely(start > OPENVM_MEM_SIZE || size > OPENVM_MEM_SIZE - start)) {
    abort_oob(start, size, OPENVM_MEM_SIZE);
  }
  assume(start <= OPENVM_MEM_SIZE);
  assume(size <= OPENVM_MEM_SIZE - start);
}

static __attribute__((always_inline)) inline void check_mem_bounds_u64_range(
    uint64_t base_addr, uint32_t num_words) {
  check_mem_bounds_range(base_addr, (size_t)num_words * sizeof(uint64_t));
}

#endif
