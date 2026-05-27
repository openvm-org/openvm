/* OpenVM segment-boundary suspension policy helpers.
 *
 * Used by metered RVR execution variants that return after the metered
 * segmentation callback creates a segment.
 */

#ifndef OPENVM_SUSPENDER_H
#define OPENVM_SUSPENDER_H

#include <stdint.h>

#include "openvm_state.h"

static __attribute__((always_inline)) inline uint8_t should_suspend(
    RvState* restrict state, uint32_t pc, uint32_t block_insn_count,
    uint8_t suspend_signal) {
  return suspend_signal;
}

#endif /* OPENVM_SUSPENDER_H */
