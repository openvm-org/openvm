/* OpenVM block-begin hook for non-metered tracers.
 *
 * Maintains final instruction-retirement accounting. The selected suspender
 * decides whether this accounting should stop execution.
 */

#ifndef OPENVM_BLOCK_H
#define OPENVM_BLOCK_H

#include <stdint.h>

#include "openvm_state.h"

static __attribute__((always_inline)) inline uint8_t begin_block(
    RvState* restrict state, uint32_t pc, uint32_t block_insn_count) {
  state->instret += block_insn_count;
  trace_block(state, pc, block_insn_count);
  return 0;
}

#endif /* OPENVM_BLOCK_H */
