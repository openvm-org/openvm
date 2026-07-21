/* OpenVM cold checkpoint helper for metered tracing. */

#ifndef OPENVM_BLOCK_H
#define OPENVM_BLOCK_H

#include <stdint.h>

#include "openvm_state.h"

[[maybe_unused]] static __attribute__((preserve_most, cold, noinline)) uint32_t
metered_checkpoint(RvState* restrict state, uint32_t check_counter) {
  MeteringState* metering = &state->mode_state;
  metering->check_counter = check_counter;
  metering->on_check(metering);
  return metering->check_counter;
}

#endif /* OPENVM_BLOCK_H */
