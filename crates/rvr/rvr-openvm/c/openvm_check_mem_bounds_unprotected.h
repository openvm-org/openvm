#ifndef OPENVM_CHECK_MEM_BOUNDS_H
#define OPENVM_CHECK_MEM_BOUNDS_H
#include <stddef.h>
#include <stdint.h>

#include "openvm_constants.h"

/* Extension callbacks still need the configured memory extent for validating
 * host-written ranges even when instruction memory accesses are unchecked. */
static constexpr size_t OPENVM_MEM_SIZE = (size_t)MEMORY_MASK + 1u;

/* Unprotected builds keep the checked-mode interface but skip bounds checks. */
static __attribute__((always_inline)) inline void check_mem_bounds_u8(
    uint64_t start [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void check_mem_bounds_i8(
    uint64_t start [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void check_mem_bounds_u16(
    uint64_t start [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void check_mem_bounds_i16(
    uint64_t start [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void check_mem_bounds_u32(
    uint64_t start [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void check_mem_bounds_u64(
    uint64_t start [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void check_mem_bounds_range(
    uint64_t start [[maybe_unused]], size_t size [[maybe_unused]]) {}
static __attribute__((always_inline)) inline void check_mem_bounds_u64_range(
    uint64_t base_addr [[maybe_unused]], uint32_t num_words [[maybe_unused]]) {}

#endif
