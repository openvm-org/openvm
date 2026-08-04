/* Compact checkpoint-and-replay-value OpenVM preflight transcript. */

#ifndef OPENVM_TRACER_PREFLIGHT_H
#define OPENVM_TRACER_PREFLIGHT_H

#include <assert.h>

#include "openvm_state.h"

static constexpr uint32_t PREFLIGHT_ERROR_NONE = 0u;
static constexpr uint32_t PREFLIGHT_ERROR_CHECKPOINT_CAPACITY = 1u;
static constexpr uint32_t PREFLIGHT_ERROR_REPLAY_VALUE_CAPACITY = 2u;
static constexpr uint32_t PREFLIGHT_ERROR_TIMESTAMP_OVERFLOW = 3u;

/* Block-local cursors avoid repeated loads while keeping the generated block
 * ABI identical to pure execution. */
typedef struct PreflightLocal {
  RvrCheckpoint* checkpoint_log;
  uint64_t* replay_value_log;
  uint64_t checkpoint_log_len;
  uint64_t checkpoint_log_cap;
  uint64_t replay_value_log_len;
  uint64_t replay_value_log_cap;
  /* Capacity promised to fixed and dynamic appends in this block. */
  uint64_t replay_value_log_reserved;
  uint32_t timestamp;
  uint32_t retired;
  uint32_t checkpoint_interval;
  uint32_t last_checkpoint_retired;
  uint32_t error;
  uint32_t instruction_limit;
  uint32_t last_memory_dirty_page;
  uint32_t padding;
} PreflightLocal;

static_assert(sizeof(PreflightLocal) == 88);
static_assert(alignof(PreflightLocal) == 8);

static __attribute__((always_inline)) inline PreflightLocal
preflight_local_load(RvState* restrict state) {
  PreflightTranscriptState* restrict p = &state->mode_state;
  return (PreflightLocal){
      .checkpoint_log = p->checkpoint_log,
      .replay_value_log = p->replay_value_log,
      .checkpoint_log_len = p->checkpoint_log_len,
      .checkpoint_log_cap = p->checkpoint_log_cap,
      .replay_value_log_len = p->replay_value_log_len,
      .replay_value_log_cap = p->replay_value_log_cap,
      .replay_value_log_reserved = p->replay_value_log_len,
      .timestamp = p->timestamp,
      .retired = p->retired,
      .checkpoint_interval = p->checkpoint_interval,
      .last_checkpoint_retired = p->last_checkpoint_retired,
      .error = p->error,
      .instruction_limit = p->instruction_limit,
      .last_memory_dirty_page = p->last_memory_dirty_page,
      .padding = 0u,
  };
}

static __attribute__((always_inline)) inline void
preflight_local_flush(
    RvState* restrict state,
    const PreflightLocal* restrict local) {
  PreflightTranscriptState* restrict p = &state->mode_state;
  p->checkpoint_log_len = local->checkpoint_log_len;
  p->replay_value_log_len = local->replay_value_log_len;
  p->timestamp = local->timestamp;
  p->retired = local->retired;
  p->last_checkpoint_retired = local->last_checkpoint_retired;
  p->error = local->error;
  p->last_memory_dirty_page = local->last_memory_dirty_page;
}

/* Dirty pages are executor bookkeeping for sparse state transfer. They are
 * deliberately not part of the replay transcript. Rust owns exact-size
 * bitsets, and the normal memory bounds checks imply these indices fit. */
#pragma clang unsafe_buffer_usage begin
static __attribute__((always_inline)) inline void
preflight_mark_dirty_page(uint64_t* restrict dirty_pages,
                          uint64_t dirty_page_words, uint64_t page) {
  uint64_t word = page >> 6;
  debug_assume(dirty_pages != NULL && word < dirty_page_words);
  dirty_pages[word] |= 1ull << (page & 63ull);
}

static __attribute__((always_inline)) inline void
preflight_local_mark_memory_write(
    RvState* restrict state, PreflightLocal* restrict local,
    uint64_t address, uint64_t size) {
  assume(size != 0u);
  PreflightTranscriptState* restrict p = &state->mode_state;
  uint32_t first = (uint32_t)(address >> PREFLIGHT_DIRTY_PAGE_BITS);
  uint32_t last =
      (uint32_t)((address + size - 1u) >> PREFLIGHT_DIRTY_PAGE_BITS);
  if (first != local->last_memory_dirty_page) {
    preflight_mark_dirty_page(
        p->memory_dirty_pages, p->memory_dirty_page_words, first);
  }
  if (last != first && last != local->last_memory_dirty_page) {
    preflight_mark_dirty_page(
        p->memory_dirty_pages, p->memory_dirty_page_words, last);
  }
  local->last_memory_dirty_page = last;
}

static __attribute__((always_inline)) inline void
preflight_mark_memory_range(RvState* restrict state,
                            uint64_t address, uint64_t size) {
  if (size == 0u) return;
  PreflightTranscriptState* restrict p = &state->mode_state;
  uint32_t first = (uint32_t)(address >> PREFLIGHT_DIRTY_PAGE_BITS);
  uint32_t last =
      (uint32_t)((address + size - 1u) >> PREFLIGHT_DIRTY_PAGE_BITS);
  for (uint32_t page = first;; ++page) {
    if (page != p->last_memory_dirty_page) {
      preflight_mark_dirty_page(
          p->memory_dirty_pages, p->memory_dirty_page_words, page);
    }
    if (page == last) break;
  }
  p->last_memory_dirty_page = last;
}

static __attribute__((always_inline)) inline void
preflight_local_set_error(
    PreflightLocal* restrict p, uint32_t error) {
  if (p->error == PREFLIGHT_ERROR_NONE) p->error = error;
}

/* Reserve before an instruction's first authoritative side effect. */
static __attribute__((always_inline)) inline bool
preflight_local_reserve(
    PreflightLocal* restrict p, uint32_t replay_values,
    uint32_t timestamp_slots) {
  if (unlikely(p->error != PREFLIGHT_ERROR_NONE)) return false;
  if (unlikely(p->replay_value_log_len > p->replay_value_log_reserved ||
               p->replay_value_log_reserved > p->replay_value_log_cap ||
               (uint64_t)replay_values >
                   p->replay_value_log_cap - p->replay_value_log_reserved ||
               (replay_values != 0u && p->replay_value_log == NULL))) {
    preflight_local_set_error(
        p, PREFLIGHT_ERROR_REPLAY_VALUE_CAPACITY);
    return false;
  }
  p->replay_value_log_reserved += replay_values;
  if (unlikely(timestamp_slots > UINT32_MAX - p->timestamp)) {
    preflight_local_set_error(
        p, PREFLIGHT_ERROR_TIMESTAMP_OVERFLOW);
    return false;
  }
  return true;
}

static __attribute__((always_inline)) inline bool
preflight_local_reserve_replay_values(
    PreflightLocal* restrict p, uint32_t replay_values) {
  return preflight_local_reserve(p, replay_values, 0u);
}

static __attribute__((always_inline)) inline void
preflight_local_add_timestamp_unchecked(
    PreflightLocal* restrict p, uint32_t slots) {
  p->timestamp += slots;
}

static __attribute__((always_inline)) inline void
preflight_local_append_replay_value_unchecked(
    PreflightLocal* restrict p, uint64_t value) {
  p->replay_value_log[p->replay_value_log_len++] = value;
}

static __attribute__((always_inline)) inline bool
preflight_local_checkpoint_due(
    const PreflightLocal* restrict p) {
  return p->checkpoint_interval != 0u &&
         p->retired > p->last_checkpoint_retired &&
         p->retired - p->last_checkpoint_retired >= p->checkpoint_interval;
}

static __attribute__((always_inline)) inline bool
preflight_local_can_execute_block(
    const PreflightLocal* restrict p, uint32_t instructions) {
  return p->retired <= p->instruction_limit &&
         instructions <= p->instruction_limit - p->retired;
}

/* The generated block has already reserved timestamp headroom and checked its
 * instruction budget, so these additions cannot overflow. */
static __attribute__((always_inline)) inline void
preflight_local_finish_block(
    PreflightLocal* restrict p, uint32_t instructions) {
  p->retired += instructions;
}

/* Hot registers must be saved to state before this helper is called. */
static __attribute__((always_inline)) inline bool
preflight_append_checkpoint(
    RvState* restrict state, uint64_t pc) {
  PreflightTranscriptState* restrict p = &state->mode_state;
  uint64_t index = p->checkpoint_log_len;
  if (unlikely(p->error != PREFLIGHT_ERROR_NONE)) return false;
  if (unlikely(index >= p->checkpoint_log_cap || p->checkpoint_log == NULL)) {
    p->error = PREFLIGHT_ERROR_CHECKPOINT_CAPACITY;
    return false;
  }
  if (unlikely(p->replay_value_log_len > UINT32_MAX)) {
    p->error = PREFLIGHT_ERROR_REPLAY_VALUE_CAPACITY;
    return false;
  }

  RvrCheckpoint* restrict checkpoint = &p->checkpoint_log[index];
  checkpoint->pc = (uint32_t)pc;
  checkpoint->timestamp = p->timestamp;
  checkpoint->retired = p->retired;
  checkpoint->replay_value_cursor = (uint32_t)p->replay_value_log_len;
  for (uint32_t i = 1u; i < 32u; ++i) {
    checkpoint->regs[i - 1u] = state->regs[i];
  }
  p->checkpoint_log_len = index + 1u;
  p->last_checkpoint_retired = p->retired;
  return true;
}
#pragma clang unsafe_buffer_usage end

/* Extension callbacks use one tracing ABI in every execution mode. Timestamp
 * and replay-value accounting stays block-local during preflight, so the
 * callback-facing hooks only preserve execution behavior. */
static __attribute__((always_inline)) inline void
trace_write_other_block_u64(
    RvState* restrict state,
    uint32_t address_space, uint32_t pointer,
    uint64_t value [[maybe_unused]], uint64_t previous_value [[maybe_unused]]) {
  if (address_space == AS_PUBLIC_VALUES) {
    PreflightTranscriptState* restrict p = &state->mode_state;
    uint64_t byte_address = (uint64_t)pointer * sizeof(uint16_t);
    uint64_t page = byte_address >> PREFLIGHT_DIRTY_PAGE_BITS;
    preflight_mark_dirty_page(
        p->public_values_dirty_pages, p->public_values_dirty_page_words, page);
  }
}

static __attribute__((always_inline)) inline void read_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, uint64_t* restrict out,
    uint32_t num_words) {
  read_mem_u64_range_raw(state, base_addr, out, num_words);
}

static __attribute__((always_inline)) inline void write_mem_u64_range(
    RvState* restrict state, uint64_t base_addr,
    const uint64_t* restrict values, uint32_t num_words) {
  write_mem_u64_range_raw(state, base_addr, values, num_words);
  preflight_mark_memory_range(
      state, base_addr, (uint64_t)num_words * sizeof(uint64_t));
}

static __attribute__((always_inline)) inline uint64_t peek_mem_u64(
    RvState* restrict state, uint64_t address) {
  return read_mem_u64(state->memory, address);
}

static __attribute__((always_inline)) inline void peek_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, uint64_t* restrict out,
    uint32_t num_words) {
  read_mem_u64_range_raw(state, base_addr, out, num_words);
}

static __attribute__((always_inline)) inline void
trace_page_access_u64_range(
    RvState* restrict state [[maybe_unused]],
    uint64_t base_addr [[maybe_unused]], uint64_t num_dwords [[maybe_unused]],
    uint32_t addr_space [[maybe_unused]], bool is_write [[maybe_unused]]) {}

static __attribute__((always_inline)) inline void
flush_main_memory_page_buffer(RvState* restrict state [[maybe_unused]]) {}

#endif /* OPENVM_TRACER_PREFLIGHT_H */
