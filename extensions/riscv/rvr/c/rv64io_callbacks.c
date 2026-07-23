/*
 * Dispatch table and forwarding stubs for the Rv64Io operations: the
 * hint-store consumers (HINT_STOREW, HINT_BUFFER) and public-values stores
 * routed through openvm_reveal.
 */

/* openvm.h exposes the mode-specific static trace helpers. Clang cannot prove
 * the fixed-capacity bounds maintained by the Rust-owned preflight buffers. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#include "openvm.h"
#pragma clang diagnostic pop
#include "rv64io_callbacks.h"

static thread_local Rv64IoHostCallbacks g_rv64io_host_callbacks;

void register_rv64io_host_callbacks(const Rv64IoHostCallbacks* cb) {
  g_rv64io_host_callbacks = *cb;
}

bool openvm_hint_prepare(uint64_t dest_addr, uint32_t num_words) {
  return g_rv64io_host_callbacks.hint_prepare(openvm_get_io_ctx(), dest_addr,
                                               num_words);
}

void openvm_hint_read_words(uint64_t* words, uint32_t num_words) {
  g_rv64io_host_callbacks.hint_read_words(openvm_get_io_ctx(), words, num_words);
}

bool openvm_reveal(RvState* state, uint64_t src_val, uint64_t base_addr,
                   uint64_t effective_addr, uint8_t width) {
  void* ctx = openvm_get_io_ctx();
  Rv64RevealPlan plan;
  if (unlikely(!g_rv64io_host_callbacks.reveal_prepare(
          ctx, src_val, base_addr, effective_addr, width, &plan))) {
    return false;
  }

  uint32_t writes = plan.crosses != 0u ? 2u : 1u;
  uint32_t slots = width == 1u ? 1u : 2u;
  if (unlikely(!trace_reserve_memory_writes(state, writes, slots))) {
    return false;
  }
  if (unlikely(!trace_write_other_block_u64(
          state, AS_PUBLIC_VALUES, (uint32_t)(plan.block_addr >> 1),
          plan.post[0], plan.previous[0]))) {
    return false;
  }
  if (width != 1u) {
    if (plan.crosses != 0u) {
      if (unlikely(!trace_write_other_block_u64(
              state, AS_PUBLIC_VALUES,
              (uint32_t)((plan.block_addr + WORD_SIZE) >> 1), plan.post[1],
              plan.previous[1]))) {
        return false;
      }
    } else {
      trace_timestamp(state);
    }
  }

  g_rv64io_host_callbacks.reveal_commit(ctx, &plan);
  return true;
}
