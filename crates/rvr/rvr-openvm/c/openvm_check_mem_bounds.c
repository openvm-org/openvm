#include "openvm_check_mem_bounds.h"

#include <stdio.h>
#include <stdlib.h>

__attribute__((noreturn, cold)) void abort_oob(uint32_t start, size_t size,
                                               size_t mem_size) {
  fprintf(stderr,
          "Memory access out of bounds: start=%u size=%zu memory_size=%zu\n",
          start, size, mem_size);
  fflush(stderr);
  abort();
}
