#include <stdio.h>
#include <stdlib.h>
#include "openvm_check_mem_bounds.h"
#include "openvm_constants.h"

static __attribute__((always_inline)) inline int unlikely(int x) {
    return __builtin_expect(!!(x), 0);
}

/* Abort on out-of-bound. */
static __attribute__((noinline, cold, noreturn))
void abort_oob(uint32_t start, size_t size, size_t mem_size) {
    fprintf(stderr, "Memory access out of bounds: start=%u size=%zu memory_size=%zu\n",
            start, size, mem_size);
    fflush(stderr);
    abort();
}

/* Memory bound checkers per width. */
void check_mem_bounds_u8(uint32_t start) {
    const size_t mem_size = (size_t)MEMORY_MASK + 1u;
    if (unlikely(start > mem_size - sizeof(uint8_t)))
        abort_oob(start, sizeof(uint8_t), mem_size);
}

void check_mem_bounds_i8(uint32_t start) {
    const size_t mem_size = (size_t)MEMORY_MASK + 1u;
    if (unlikely(start > mem_size - sizeof(int8_t)))
        abort_oob(start, sizeof(int8_t), mem_size);
}

void check_mem_bounds_u16(uint32_t start) {
    const size_t mem_size = (size_t)MEMORY_MASK + 1u;
    if (unlikely(start > mem_size - sizeof(uint16_t)))
        abort_oob(start, sizeof(uint16_t), mem_size);
}

void check_mem_bounds_i16(uint32_t start) {
    const size_t mem_size = (size_t)MEMORY_MASK + 1u;
    if (unlikely(start > mem_size - sizeof(int16_t)))
        abort_oob(start, sizeof(int16_t), mem_size);
}

void check_mem_bounds_u32(uint32_t start) {
    const size_t mem_size = (size_t)MEMORY_MASK + 1u;
    if (unlikely(start > mem_size - sizeof(uint32_t)))
        abort_oob(start, sizeof(uint32_t), mem_size);
}

void check_mem_bounds_u32_range(uint32_t base_addr, uint32_t num_words) {
    const size_t mem_size = (size_t)MEMORY_MASK + 1u;
    const size_t size = (size_t)num_words * sizeof(uint32_t);
    if (unlikely(base_addr > mem_size || size > mem_size - base_addr))
        abort_oob(base_addr, size, mem_size);
}
