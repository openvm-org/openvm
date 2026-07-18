/* OpenVM cold checkpoint helper for segment-boundary metered tracing. */

#ifndef OPENVM_BLOCK_H
#define OPENVM_BLOCK_H

#include <stdint.h>

#include "openvm_state.h"

typedef struct MeteredSegmentCheckpointResult {
  uint32_t check_counter;
  uint8_t suspend_signal;
  /* Keep the returned aggregate fully initialized. */
  uint8_t reserved[3];
} MeteredSegmentCheckpointResult;

static __attribute__((preserve_most, cold,
                     noinline)) MeteredSegmentCheckpointResult
metered_segment_checkpoint(RvState* restrict state, uint32_t check_counter) {
  MeteringState* metering = &state->mode_state;
  metering->check_counter = check_counter;
  uint8_t suspend_signal = metering->on_check(metering);
  return (MeteredSegmentCheckpointResult){
      .check_counter = metering->check_counter,
      .suspend_signal = suspend_signal,
      .reserved = {0},
  };
}

#endif /* OPENVM_BLOCK_H */
