/* Minimal append-only OpenVM preflight logging. */

#ifndef OPENVM_TRACER_PREFLIGHT_H
#define OPENVM_TRACER_PREFLIGHT_H

#include "openvm_state.h"

static constexpr uint32_t PREFLIGHT_ERROR_NONE = 0u;
static constexpr uint32_t PREFLIGHT_ERROR_PROGRAM_CAPACITY = 1u;
static constexpr uint32_t PREFLIGHT_ERROR_MEMORY_CAPACITY = 2u;
static constexpr uint32_t PREFLIGHT_ERROR_INITIAL_WRITE_CAPACITY = 3u;
static constexpr uint32_t PREFLIGHT_ERROR_TIMESTAMP_OVERFLOW = 4u;
static constexpr uint32_t PREFLIGHT_WRITE_BIT = 1u << 31;
static constexpr uint32_t PREFLIGHT_BLOCK_CELLS = 4u;
static constexpr uint32_t PREFLIGHT_BLOCK_BYTES = 8u;

static_assert(WORD_SIZE == PREFLIGHT_BLOCK_BYTES,
              "RVR preflight requires four u16 cells per memory event");

/* Block-local cursors keep the dependent append state in registers. They are
 * loaded after the whole-block suspension guard and flushed before control
 * leaves the generated block. */
typedef struct PreflightLocal {
  PreflightProgramEvent* program_log;
  PreflightMemoryEvent* memory_log;
  PreflightInitialWrite* initial_write_log;
  uint64_t program_log_len;
  uint64_t program_log_cap;
  uint64_t memory_log_len;
  uint64_t memory_log_cap;
  uint64_t initial_write_log_len;
  uint64_t initial_write_log_cap;
  uint32_t timestamp;
  uint32_t error;
  uint32_t seen_register_blocks;
  uint32_t wrote_register;
  uint32_t has_non_register_events;
  uint32_t padding;
} PreflightLocal;

static __attribute__((always_inline)) inline PreflightLocal
preflight_local_load(RvState* restrict state) {
  PreflightState* restrict p = &state->mode_state;
  return (PreflightLocal){
      .program_log = p->program_log,
      .memory_log = p->memory_log,
      .initial_write_log = p->initial_write_log,
      .program_log_len = p->program_log_len,
      .program_log_cap = p->program_log_cap,
      .memory_log_len = p->memory_log_len,
      .memory_log_cap = p->memory_log_cap,
      .initial_write_log_len = p->initial_write_log_len,
      .initial_write_log_cap = p->initial_write_log_cap,
      .timestamp = p->timestamp,
      .error = p->error,
      .seen_register_blocks = p->seen_register_blocks,
      .wrote_register = p->wrote_register,
      .has_non_register_events = p->has_non_register_events,
  };
}

static __attribute__((always_inline)) inline void preflight_local_flush(
    RvState* restrict state, const PreflightLocal* restrict local) {
  PreflightState* restrict p = &state->mode_state;
  p->program_log_len = local->program_log_len;
  p->memory_log_len = local->memory_log_len;
  p->initial_write_log_len = local->initial_write_log_len;
  p->timestamp = local->timestamp;
  p->error = local->error;
  p->seen_register_blocks = local->seen_register_blocks;
  p->wrote_register = local->wrote_register;
  p->has_non_register_events = local->has_non_register_events;
}

static __attribute__((always_inline)) inline void preflight_local_reload(
    PreflightLocal* restrict local, RvState* restrict state) {
  PreflightState* restrict p = &state->mode_state;
  local->program_log_len = p->program_log_len;
  local->memory_log_len = p->memory_log_len;
  local->initial_write_log_len = p->initial_write_log_len;
  local->timestamp = p->timestamp;
  local->error = p->error;
  local->seen_register_blocks = p->seen_register_blocks;
  local->wrote_register = p->wrote_register;
  local->has_non_register_events = p->has_non_register_events;
}

static __attribute__((always_inline)) inline void preflight_set_error(
    PreflightState* restrict p, uint32_t error) {
  if (p->error == PREFLIGHT_ERROR_NONE) p->error = error;
}

static __attribute__((always_inline)) inline void preflight_pack_u64(
    uint16_t out[static PREFLIGHT_BLOCK_CELLS], uint64_t value) {
  out[0] = (uint16_t)value;
  out[1] = (uint16_t)(value >> 16);
  out[2] = (uint16_t)(value >> 32);
  out[3] = (uint16_t)(value >> 48);
}

static __attribute__((always_inline)) inline void preflight_local_set_error(
    PreflightLocal* restrict p, uint32_t error) {
  if (p->error == PREFLIGHT_ERROR_NONE) p->error = error;
}

static __attribute__((always_inline)) inline bool preflight_local_can_tick(
    PreflightLocal* restrict p) {
  if (unlikely(p->timestamp == UINT32_MAX)) {
    preflight_local_set_error(p, PREFLIGHT_ERROR_TIMESTAMP_OVERFLOW);
    return false;
  }
  return true;
}

static __attribute__((always_inline)) inline void preflight_local_timestamp(
    PreflightLocal* restrict p) {
  if (unlikely(p->error != PREFLIGHT_ERROR_NONE)) return;
  if (preflight_local_can_tick(p)) p->timestamp++;
}

static __attribute__((always_inline)) inline void
preflight_local_advance_timestamp(PreflightLocal* restrict p,
                                  uint32_t slots) {
  if (unlikely(p->error != PREFLIGHT_ERROR_NONE)) return;
  if (unlikely(slots > UINT32_MAX - p->timestamp)) {
    preflight_local_set_error(p, PREFLIGHT_ERROR_TIMESTAMP_OVERFLOW);
    return;
  }
  p->timestamp += slots;
}

static __attribute__((always_inline)) inline void preflight_local_trace_pc(
    PreflightLocal* restrict p, uint64_t pc) {
  uint64_t index = p->program_log_len;
  p->program_log[index] =
      (PreflightProgramEvent){.pc = (uint32_t)pc, .timestamp = p->timestamp};
  p->program_log_len = index + 1u;
}

static __attribute__((always_inline)) inline void
preflight_local_append_read_u64(PreflightLocal* restrict p,
                                uint32_t address_space, uint32_t pointer,
                                uint64_t value) {
  if (unlikely(p->error != PREFLIGHT_ERROR_NONE)) return;
  uint64_t index = p->memory_log_len;
  if (unlikely(index >= p->memory_log_cap)) {
    preflight_local_set_error(p, PREFLIGHT_ERROR_MEMORY_CAPACITY);
    return;
  }
  if (unlikely(!preflight_local_can_tick(p))) return;

  PreflightMemoryEvent* restrict event = &p->memory_log[index];
  event->timestamp = p->timestamp++;
  event->address_space_and_kind = address_space;
  event->pointer = pointer;
  preflight_pack_u64(event->value, value);
  p->memory_log_len = index + 1u;
  p->has_non_register_events |= address_space != AS_REGISTER;
}

static __attribute__((always_inline)) inline void
preflight_local_append_write_event_u64(PreflightLocal* restrict p,
                                       uint32_t address_space,
                                       uint32_t pointer, uint64_t value) {
  if (unlikely(p->error != PREFLIGHT_ERROR_NONE)) return;
  uint64_t memory_index = p->memory_log_len;
  if (unlikely(memory_index >= p->memory_log_cap)) {
    preflight_local_set_error(p, PREFLIGHT_ERROR_MEMORY_CAPACITY);
    return;
  }
  if (unlikely(!preflight_local_can_tick(p))) return;

  PreflightMemoryEvent* restrict event = &p->memory_log[memory_index];
  event->timestamp = p->timestamp++;
  event->address_space_and_kind = address_space | PREFLIGHT_WRITE_BIT;
  event->pointer = pointer;
  preflight_pack_u64(event->value, value);
  p->memory_log_len = memory_index + 1u;
  p->has_non_register_events |= address_space != AS_REGISTER;
}

static __attribute__((always_inline)) inline void
preflight_local_append_write_u64(PreflightLocal* restrict p,
                                 uint32_t address_space, uint32_t pointer,
                                 uint64_t value, uint64_t previous_value) {
  if (unlikely(p->error != PREFLIGHT_ERROR_NONE)) return;
  uint64_t memory_index = p->memory_log_len;
  uint64_t initial_index = p->initial_write_log_len;
  if (unlikely(memory_index >= p->memory_log_cap)) {
    preflight_local_set_error(p, PREFLIGHT_ERROR_MEMORY_CAPACITY);
    return;
  }
  if (unlikely(initial_index >= p->initial_write_log_cap)) {
    preflight_local_set_error(p, PREFLIGHT_ERROR_INITIAL_WRITE_CAPACITY);
    return;
  }
  if (unlikely(!preflight_local_can_tick(p))) return;

  PreflightMemoryEvent* restrict event = &p->memory_log[memory_index];
  event->timestamp = p->timestamp++;
  event->address_space_and_kind = address_space | PREFLIGHT_WRITE_BIT;
  event->pointer = pointer;
  preflight_pack_u64(event->value, value);
  p->memory_log_len = memory_index + 1u;

  PreflightInitialWrite* restrict initial =
      &p->initial_write_log[initial_index];
  initial->address_space = address_space;
  initial->pointer = pointer;
  preflight_pack_u64(initial->initial_value, previous_value);
  p->initial_write_log_len = initial_index + 1u;
  p->has_non_register_events |= address_space != AS_REGISTER;
}

static __attribute__((always_inline)) inline void preflight_local_reg_read(
    PreflightLocal* restrict p, uint8_t index, uint64_t value) {
  preflight_local_append_read_u64(
      p, AS_REGISTER, (uint32_t)index * PREFLIGHT_BLOCK_CELLS, value);
  if (likely(p->error == PREFLIGHT_ERROR_NONE)) {
    p->seen_register_blocks |= 1u << index;
  }
}

static __attribute__((always_inline)) inline void preflight_local_reg_write(
    PreflightLocal* restrict p, uint8_t index, uint64_t value,
    uint64_t previous_value) {
  uint32_t bit = 1u << index;
  uint32_t pointer = (uint32_t)index * PREFLIGHT_BLOCK_CELLS;
  if ((p->seen_register_blocks & bit) == 0u) {
    preflight_local_append_write_u64(p, AS_REGISTER, pointer, value,
                                     previous_value);
  } else {
    preflight_local_append_write_event_u64(p, AS_REGISTER, pointer, value);
  }
  if (likely(p->error == PREFLIGHT_ERROR_NONE)) {
    p->seen_register_blocks |= bit;
    p->wrote_register = 1u;
  }
}

static __attribute__((always_inline)) inline bool preflight_local_reserve(
    PreflightLocal* restrict p, uint32_t events, uint32_t writes,
    uint32_t slots) {
  if (unlikely(p->error != PREFLIGHT_ERROR_NONE)) return false;
  if (unlikely(p->memory_log_len > p->memory_log_cap ||
               events > p->memory_log_cap - p->memory_log_len)) {
    preflight_local_set_error(p, PREFLIGHT_ERROR_MEMORY_CAPACITY);
    return false;
  }
  if (unlikely(p->initial_write_log_len > p->initial_write_log_cap ||
               writes >
                   p->initial_write_log_cap - p->initial_write_log_len)) {
    preflight_local_set_error(p, PREFLIGHT_ERROR_INITIAL_WRITE_CAPACITY);
    return false;
  }
  if (unlikely(slots > UINT32_MAX - p->timestamp)) {
    preflight_local_set_error(p, PREFLIGHT_ERROR_TIMESTAMP_OVERFLOW);
    return false;
  }
  return true;
}

static __attribute__((always_inline)) inline void
preflight_local_timestamp_unchecked(PreflightLocal* restrict p) {
  p->timestamp++;
}

static __attribute__((always_inline)) inline void
preflight_local_advance_timestamp_unchecked(PreflightLocal* restrict p,
                                            uint32_t slots) {
  p->timestamp += slots;
}

static __attribute__((always_inline)) inline void
preflight_local_append_read_u64_unchecked(PreflightLocal* restrict p,
                                          uint32_t address_space,
                                          uint32_t pointer, uint64_t value) {
  uint64_t index = p->memory_log_len++;
  PreflightMemoryEvent* restrict event = &p->memory_log[index];
  event->timestamp = p->timestamp++;
  event->address_space_and_kind = address_space;
  event->pointer = pointer;
  preflight_pack_u64(event->value, value);
  p->has_non_register_events |= address_space != AS_REGISTER;
}

static __attribute__((always_inline)) inline void
preflight_local_append_write_event_u64_unchecked(
    PreflightLocal* restrict p, uint32_t address_space, uint32_t pointer,
    uint64_t value) {
  uint64_t memory_index = p->memory_log_len++;
  PreflightMemoryEvent* restrict event = &p->memory_log[memory_index];
  event->timestamp = p->timestamp++;
  event->address_space_and_kind = address_space | PREFLIGHT_WRITE_BIT;
  event->pointer = pointer;
  preflight_pack_u64(event->value, value);
  p->has_non_register_events |= address_space != AS_REGISTER;
}

static __attribute__((always_inline)) inline void
preflight_local_append_write_u64_unchecked(
    PreflightLocal* restrict p, uint32_t address_space, uint32_t pointer,
    uint64_t value, uint64_t previous_value) {
  uint64_t memory_index = p->memory_log_len++;
  PreflightMemoryEvent* restrict event = &p->memory_log[memory_index];
  event->timestamp = p->timestamp++;
  event->address_space_and_kind = address_space | PREFLIGHT_WRITE_BIT;
  event->pointer = pointer;
  preflight_pack_u64(event->value, value);

  uint64_t initial_index = p->initial_write_log_len++;
  PreflightInitialWrite* restrict initial =
      &p->initial_write_log[initial_index];
  initial->address_space = address_space;
  initial->pointer = pointer;
  preflight_pack_u64(initial->initial_value, previous_value);
  p->has_non_register_events |= address_space != AS_REGISTER;
}

static __attribute__((always_inline)) inline void
preflight_local_reg_read_unchecked(PreflightLocal* restrict p, uint8_t index,
                                   uint64_t value) {
  preflight_local_append_read_u64_unchecked(
      p, AS_REGISTER, (uint32_t)index * PREFLIGHT_BLOCK_CELLS, value);
  p->seen_register_blocks |= 1u << index;
}

static __attribute__((always_inline)) inline void
preflight_local_reg_write_unchecked(PreflightLocal* restrict p, uint8_t index,
                                    uint64_t value,
                                    uint64_t previous_value) {
  uint32_t bit = 1u << index;
  uint32_t pointer = (uint32_t)index * PREFLIGHT_BLOCK_CELLS;
  if ((p->seen_register_blocks & bit) == 0u) {
    preflight_local_append_write_u64_unchecked(p, AS_REGISTER, pointer, value,
                                               previous_value);
  } else {
    preflight_local_append_write_event_u64_unchecked(p, AS_REGISTER, pointer,
                                                     value);
  }
  p->seen_register_blocks |= bit;
  p->wrote_register = 1u;
}

static __attribute__((always_inline)) inline bool preflight_can_tick(
    PreflightState* restrict p) {
  if (unlikely(p->timestamp == UINT32_MAX)) {
    preflight_set_error(p, PREFLIGHT_ERROR_TIMESTAMP_OVERFLOW);
    return false;
  }
  return true;
}

static __attribute__((always_inline)) inline void trace_timestamp(
    RvState* restrict state) {
  PreflightState* restrict p = &state->mode_state;
  if (unlikely(p->error != PREFLIGHT_ERROR_NONE)) return;
  if (preflight_can_tick(p)) p->timestamp++;
}

static __attribute__((always_inline)) inline void trace_advance_timestamp(
    RvState* restrict state, uint32_t slots) {
  PreflightState* restrict p = &state->mode_state;
  if (unlikely(p->error != PREFLIGHT_ERROR_NONE)) return;
  if (unlikely(slots > UINT32_MAX - p->timestamp)) {
    preflight_set_error(p, PREFLIGHT_ERROR_TIMESTAMP_OVERFLOW);
    return;
  }
  p->timestamp += slots;
}

/* The generated block boundary reserves every PC entry in the block before
 * executing its first instruction. PCs are compile-time validated u32 values,
 * so the individual append is deliberately unchecked. */
static __attribute__((always_inline)) inline void trace_pc(
    RvState* restrict state, uint64_t pc) {
  PreflightState* restrict p = &state->mode_state;
  uint64_t index = p->program_log_len;
  p->program_log[index] =
      (PreflightProgramEvent){.pc = (uint32_t)pc, .timestamp = p->timestamp};
  p->program_log_len = index + 1u;
}

static __attribute__((always_inline)) inline void preflight_append_read_u64(
    RvState* restrict state, uint32_t address_space, uint32_t pointer,
    uint64_t value) {
  PreflightState* restrict p = &state->mode_state;
  if (unlikely(p->error != PREFLIGHT_ERROR_NONE)) return;
  uint64_t index = p->memory_log_len;
  if (unlikely(p->memory_log == NULL || index >= p->memory_log_cap)) {
    preflight_set_error(p, PREFLIGHT_ERROR_MEMORY_CAPACITY);
    return;
  }
  if (unlikely(!preflight_can_tick(p))) return;

  PreflightMemoryEvent* restrict event = &p->memory_log[index];
  event->timestamp = p->timestamp++;
  event->address_space_and_kind = address_space;
  event->pointer = pointer;
  preflight_pack_u64(event->value, value);
  p->memory_log_len = index + 1u;
  p->has_non_register_events |= address_space != AS_REGISTER;
}

static __attribute__((always_inline)) inline void
preflight_append_write_event_u64(RvState* restrict state,
                                 uint32_t address_space, uint32_t pointer,
                                 uint64_t value) {
  PreflightState* restrict p = &state->mode_state;
  if (unlikely(p->error != PREFLIGHT_ERROR_NONE)) return;
  uint64_t memory_index = p->memory_log_len;
  if (unlikely(p->memory_log == NULL || memory_index >= p->memory_log_cap)) {
    preflight_set_error(p, PREFLIGHT_ERROR_MEMORY_CAPACITY);
    return;
  }
  if (unlikely(!preflight_can_tick(p))) return;

  PreflightMemoryEvent* restrict event = &p->memory_log[memory_index];
  event->timestamp = p->timestamp++;
  event->address_space_and_kind = address_space | PREFLIGHT_WRITE_BIT;
  event->pointer = pointer;
  preflight_pack_u64(event->value, value);
  p->memory_log_len = memory_index + 1u;
  p->has_non_register_events |= address_space != AS_REGISTER;
}

static __attribute__((always_inline)) inline void preflight_append_write_u64(
    RvState* restrict state, uint32_t address_space, uint32_t pointer,
    uint64_t value, uint64_t previous_value) {
  PreflightState* restrict p = &state->mode_state;
  if (unlikely(p->error != PREFLIGHT_ERROR_NONE)) return;
  uint64_t memory_index = p->memory_log_len;
  uint64_t initial_index = p->initial_write_log_len;
  if (unlikely(p->memory_log == NULL || memory_index >= p->memory_log_cap)) {
    preflight_set_error(p, PREFLIGHT_ERROR_MEMORY_CAPACITY);
    return;
  }
  if (unlikely(p->initial_write_log == NULL ||
               initial_index >= p->initial_write_log_cap)) {
    preflight_set_error(p, PREFLIGHT_ERROR_INITIAL_WRITE_CAPACITY);
    return;
  }
  if (unlikely(!preflight_can_tick(p))) return;

  PreflightMemoryEvent* restrict event = &p->memory_log[memory_index];
  event->timestamp = p->timestamp++;
  event->address_space_and_kind = address_space | PREFLIGHT_WRITE_BIT;
  event->pointer = pointer;
  preflight_pack_u64(event->value, value);
  p->memory_log_len = memory_index + 1u;

  PreflightInitialWrite* restrict initial =
      &p->initial_write_log[initial_index];
  initial->address_space = address_space;
  initial->pointer = pointer;
  preflight_pack_u64(initial->initial_value, previous_value);
  p->initial_write_log_len = initial_index + 1u;
  p->has_non_register_events |= address_space != AS_REGISTER;
}

static __attribute__((always_inline)) inline bool
trace_reserve_memory_writes(RvState* restrict state, uint32_t writes,
                            uint32_t slots) {
  PreflightState* restrict p = &state->mode_state;
  if (unlikely(p->error != PREFLIGHT_ERROR_NONE)) return false;
  if (unlikely(p->memory_log_len > p->memory_log_cap ||
               writes > p->memory_log_cap - p->memory_log_len ||
               (writes != 0u && p->memory_log == NULL))) {
    preflight_set_error(p, PREFLIGHT_ERROR_MEMORY_CAPACITY);
    return false;
  }
  if (unlikely(p->initial_write_log_len > p->initial_write_log_cap ||
               writes >
                   p->initial_write_log_cap - p->initial_write_log_len ||
               (writes != 0u && p->initial_write_log == NULL))) {
    preflight_set_error(p, PREFLIGHT_ERROR_INITIAL_WRITE_CAPACITY);
    return false;
  }
  if (unlikely(slots > UINT32_MAX - p->timestamp)) {
    preflight_set_error(p, PREFLIGHT_ERROR_TIMESTAMP_OVERFLOW);
    return false;
  }
  return true;
}

static __attribute__((always_inline)) inline bool
trace_write_other_block_u64(RvState* restrict state, uint32_t address_space,
                            uint32_t pointer, uint64_t value,
                            uint64_t previous_value) {
  preflight_append_write_u64(state, address_space, pointer, value,
                             previous_value);
  return state->mode_state.error == PREFLIGHT_ERROR_NONE;
}

static __attribute__((always_inline)) inline void trace_reg_read(
    RvState* restrict state, uint8_t index, uint64_t value) {
  preflight_append_read_u64(state, AS_REGISTER,
                            (uint32_t)index * PREFLIGHT_BLOCK_CELLS, value);
  if (likely(state->mode_state.error == PREFLIGHT_ERROR_NONE)) {
    state->mode_state.seen_register_blocks |= 1u << index;
  }
}

static __attribute__((always_inline)) inline void trace_reg_peek(
    RvState* restrict state [[maybe_unused]],
    uint8_t index [[maybe_unused]], uint64_t value [[maybe_unused]]) {}

static __attribute__((always_inline)) inline void trace_reg_write(
    RvState* restrict state, uint8_t index, uint64_t value,
    uint64_t previous_value) {
  PreflightState* restrict p = &state->mode_state;
  uint32_t bit = 1u << index;
  uint32_t pointer = (uint32_t)index * PREFLIGHT_BLOCK_CELLS;
  if ((p->seen_register_blocks & bit) == 0u) {
    preflight_append_write_u64(state, AS_REGISTER, pointer, value,
                               previous_value);
  } else {
    preflight_append_write_event_u64(state, AS_REGISTER, pointer, value);
  }
  if (likely(p->error == PREFLIGHT_ERROR_NONE)) {
    p->seen_register_blocks |= bit;
    p->wrote_register = 1u;
  }
}

static __attribute__((always_inline)) inline uint64_t preflight_read_block(
    RvState* restrict state, uint64_t block_address) {
  return read_mem_u64(state->memory, block_address);
}

static __attribute__((always_inline)) inline void preflight_trace_load(
    RvState* restrict state, uint64_t address, uint32_t width) {
  uint64_t block_address = address & ~(uint64_t)(PREFLIGHT_BLOCK_BYTES - 1u);
  uint64_t value = preflight_read_block(state, block_address);
  preflight_append_read_u64(state, AS_MEMORY, (uint32_t)(block_address >> 1),
                            value);

  if (width != 1u) {
    uint32_t shift = (uint32_t)(address - block_address);
    if (shift + width > PREFLIGHT_BLOCK_BYTES) {
      block_address += PREFLIGHT_BLOCK_BYTES;
      value = preflight_read_block(state, block_address);
      preflight_append_read_u64(state, AS_MEMORY,
                                (uint32_t)(block_address >> 1), value);
    } else {
      trace_timestamp(state);
    }
  }
}

static __attribute__((always_inline)) inline void preflight_trace_store(
    RvState* restrict state, uint64_t address, uint64_t scalar,
    uint32_t width) {
  uint64_t block_address = address & ~(uint64_t)(PREFLIGHT_BLOCK_BYTES - 1u);
  uint32_t shift = (uint32_t)(address - block_address);
  uint64_t previous[2] = {preflight_read_block(state, block_address), 0u};
  uint8_t bytes[2 * PREFLIGHT_BLOCK_BYTES];
  memcpy(bytes, &previous[0], PREFLIGHT_BLOCK_BYTES);

  bool crosses = shift + width > PREFLIGHT_BLOCK_BYTES;
  if (crosses) {
    previous[1] =
        preflight_read_block(state, block_address + PREFLIGHT_BLOCK_BYTES);
    memcpy(bytes + PREFLIGHT_BLOCK_BYTES, &previous[1], PREFLIGHT_BLOCK_BYTES);
  } else {
    memset(bytes + PREFLIGHT_BLOCK_BYTES, 0, PREFLIGHT_BLOCK_BYTES);
  }
  memcpy(bytes + shift, &scalar, width);

  uint64_t post;
  memcpy(&post, bytes, PREFLIGHT_BLOCK_BYTES);
  preflight_append_write_u64(state, AS_MEMORY, (uint32_t)(block_address >> 1),
                             post, previous[0]);

  if (width != 1u) {
    if (crosses) {
      memcpy(&post, bytes + PREFLIGHT_BLOCK_BYTES, PREFLIGHT_BLOCK_BYTES);
      preflight_append_write_u64(
          state, AS_MEMORY,
          (uint32_t)((block_address + PREFLIGHT_BLOCK_BYTES) >> 1), post,
          previous[1]);
    } else {
      trace_timestamp(state);
    }
  }
}

static __attribute__((always_inline)) inline void trace_read_mem_u8(
    RvState* restrict state, uint64_t address,
    uint64_t value [[maybe_unused]]) {
  preflight_trace_load(state, address, 1u);
}

static __attribute__((always_inline)) inline void trace_read_mem_i8(
    RvState* restrict state, uint64_t address,
    int32_t value [[maybe_unused]]) {
  preflight_trace_load(state, address, 1u);
}

static __attribute__((always_inline)) inline void trace_read_mem_u16(
    RvState* restrict state, uint64_t address,
    uint64_t value [[maybe_unused]]) {
  preflight_trace_load(state, address, 2u);
}

static __attribute__((always_inline)) inline void trace_read_mem_i16(
    RvState* restrict state, uint64_t address,
    int32_t value [[maybe_unused]]) {
  preflight_trace_load(state, address, 2u);
}

static __attribute__((always_inline)) inline void trace_read_mem_u32(
    RvState* restrict state, uint64_t address,
    uint64_t value [[maybe_unused]]) {
  preflight_trace_load(state, address, 4u);
}

static __attribute__((always_inline)) inline void trace_read_mem_i32(
    RvState* restrict state, uint64_t address,
    int32_t value [[maybe_unused]]) {
  preflight_trace_load(state, address, 4u);
}

static __attribute__((always_inline)) inline void trace_read_mem_u64(
    RvState* restrict state, uint64_t address,
    uint64_t value [[maybe_unused]]) {
  preflight_trace_load(state, address, 8u);
}

static __attribute__((always_inline)) inline void trace_write_mem_u8(
    RvState* restrict state, uint64_t address, uint64_t value) {
  preflight_trace_store(state, address, value, 1u);
}

static __attribute__((always_inline)) inline void trace_write_mem_u16(
    RvState* restrict state, uint64_t address, uint64_t value) {
  preflight_trace_store(state, address, value, 2u);
}

static __attribute__((always_inline)) inline void trace_write_mem_u32(
    RvState* restrict state, uint64_t address, uint64_t value) {
  preflight_trace_store(state, address, value, 4u);
}

static __attribute__((always_inline)) inline void trace_write_mem_u64(
    RvState* restrict state, uint64_t address, uint64_t value) {
  preflight_trace_store(state, address, value, 8u);
}

static __attribute__((always_inline)) inline void trace_write_mem_block_u64(
    RvState* restrict state, uint64_t address, uint64_t value) {
  uint64_t previous_value = preflight_read_block(state, address);
  preflight_append_write_u64(state, AS_MEMORY, (uint32_t)(address >> 1),
                             value, previous_value);
}

/* Extension range access is not preflight-safe yet. Artifact construction
 * rejects extension registries that request these wrappers. */
static __attribute__((always_inline)) inline void read_mem_u64_range(
    RvState* restrict state, uint64_t base_addr, uint64_t* restrict out,
    uint32_t num_words) {
  read_mem_u64_range_raw(state, base_addr, out, num_words);
}

static __attribute__((always_inline)) inline void write_mem_u64_range(
    RvState* restrict state, uint64_t base_addr,
    const uint64_t* restrict values, uint32_t num_words) {
  write_mem_u64_range_raw(state, base_addr, values, num_words);
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
flush_main_memory_page_buffer(RvState* restrict state [[maybe_unused]]) {}

#endif /* OPENVM_TRACER_PREFLIGHT_H */
