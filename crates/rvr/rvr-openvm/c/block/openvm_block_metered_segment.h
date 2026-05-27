/* OpenVM block-begin hook for metered tracing with segment-boundary events. */

#ifndef OPENVM_BLOCK_H
#define OPENVM_BLOCK_H

#include <stdint.h>

#include "openvm_state.h"

typedef struct MeteredSegmentCheckpointResult {
  uint32_t check_counter;
  uint8_t suspend_signal;
} MeteredSegmentCheckpointResult;

static __attribute__((always_inline)) inline uint8_t begin_block(
    RvState* restrict state, uint32_t pc, uint32_t block_insn_count) {
  return trace_block_with_segment_check(state, pc, block_insn_count);
}

static __attribute__((preserve_most, cold,
                      noinline)) MeteredSegmentCheckpointResult
metered_segment_checkpoint(RvState* restrict state, uint32_t check_counter) {
  Tracer* t = state->tracer;
  t->check_counter = check_counter;
  uint8_t suspend_signal = t->on_check(t);
  return (MeteredSegmentCheckpointResult){
      .check_counter = t->check_counter,
      .suspend_signal = suspend_signal,
  };
}

#endif /* OPENVM_BLOCK_H */
