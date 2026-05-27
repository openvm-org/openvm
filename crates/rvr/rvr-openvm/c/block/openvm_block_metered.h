/* OpenVM block-begin hook for metered tracing without block events. */

#ifndef OPENVM_BLOCK_H
#define OPENVM_BLOCK_H

#include <stdint.h>

#include "openvm_state.h"

static __attribute__((always_inline)) inline uint8_t begin_block(
    RvState* restrict state, uint32_t pc, uint32_t block_insn_count) {
  trace_block(state, pc, block_insn_count);
  return 0;
}

static __attribute__((preserve_most, cold, noinline)) uint32_t
metered_checkpoint(RvState* restrict state, uint32_t check_counter) {
  Tracer* t = state->tracer;
  t->check_counter = check_counter;
  t->on_check(t);
  return t->check_counter;
}

#endif /* OPENVM_BLOCK_H */
