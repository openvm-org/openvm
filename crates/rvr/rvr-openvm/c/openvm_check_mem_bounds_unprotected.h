#ifndef OPENVM_CHECK_MEM_BOUNDS_H
#define OPENVM_CHECK_MEM_BOUNDS_H
#include <stddef.h>
#include <stdint.h>

static __attribute__((always_inline)) inline void check_mem_bounds_u8(
    uint32_t start) {}
static __attribute__((always_inline)) inline void check_mem_bounds_i8(
    uint32_t start) {}
static __attribute__((always_inline)) inline void check_mem_bounds_u16(
    uint32_t start) {}
static __attribute__((always_inline)) inline void check_mem_bounds_i16(
    uint32_t start) {}
static __attribute__((always_inline)) inline void check_mem_bounds_u32(
    uint32_t start) {}
static __attribute__((always_inline)) inline void check_mem_bounds_range(
    uint32_t start, size_t size) {}
static __attribute__((always_inline)) inline void check_mem_bounds_u32_range(
    uint32_t base_addr, uint32_t num_words) {}

#endif
