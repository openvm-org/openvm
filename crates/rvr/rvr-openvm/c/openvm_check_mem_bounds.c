#include "openvm_check_mem_bounds.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((noreturn, cold)) void abort_oob(uint64_t start, size_t size,
                                               size_t mem_size) {
  fprintf(stderr,
          "Memory access out of bounds: start=%" PRIu64
          " size=%zu memory_size=%zu\n",
          start, size, mem_size);
  fflush(stderr);
  abort();
}
