/* OpenVM block-begin hook for metered tracing with segment-boundary events. */

#ifndef OPENVM_BLOCK_H
#define OPENVM_BLOCK_H

#include <stdint.h>

#include "openvm_state.h"

static __attribute__((always_inline)) inline uint8_t begin_block(RvState* restrict state, uint32_t pc,
                                                                 uint32_t block_insn_count) {
  return trace_block_with_segment_check(state, pc, block_insn_count);
}

#endif /* OPENVM_BLOCK_H */
