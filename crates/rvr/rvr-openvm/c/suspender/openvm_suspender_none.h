/* OpenVM no-op suspension policy helpers. */

#ifndef OPENVM_SUSPENDER_H
#define OPENVM_SUSPENDER_H

#include <stdint.h>

#include "openvm_state.h"

static __attribute__((always_inline)) inline uint8_t should_suspend(RvState* restrict state, uint32_t pc,
                                                                    uint32_t block_insn_count,
                                                                    uint8_t suspend_signal) {
  return 0;
}

#endif /* OPENVM_SUSPENDER_H */
