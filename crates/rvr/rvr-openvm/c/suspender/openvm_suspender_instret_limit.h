/* OpenVM instruction-retirement-limit suspension policy helpers. */

#ifndef OPENVM_SUSPENDER_H
#define OPENVM_SUSPENDER_H

#include <stdint.h>

#include "openvm_state.h"

static __attribute__((always_inline)) inline uint8_t should_suspend(RvState* restrict state, uint32_t pc,
                                                                    uint32_t block_insn_count,
                                                                    uint8_t suspend_signal) {
  if (unlikely(state->instret > state->target_instret)) {
    state->instret -= block_insn_count;
    return 1;
  }
  return 0;
}

#endif /* OPENVM_SUSPENDER_H */
