#pragma once

#include "arch/rvr/preflight.cuh"
#include "primitives/buffer_view.cuh"
#include "system/memory/params.cuh"

struct ReplayPreviousValue {
    uint32_t timestamp;
    uint16_t value[BLOCK_FE_WIDTH];
};

static __device__ bool replay_u16_block(
    uint32_t const (&source)[BLOCK_FE_WIDTH],
    uint16_t (&out)[BLOCK_FE_WIDTH]
) {
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        if (source[i] > 0xffffu) {
            return false;
        }
        out[i] = static_cast<uint16_t>(source[i]);
    }
    return true;
}

static __device__ bool replay_previous_value(
    size_t event_index,
    PreflightMemoryEvent const &event,
    uint32_t predecessor,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    ReplayPreviousValue &out
) {
    if (predecessor == 0) {
        out.timestamp = 0;
        if (preflight_is_write(event)) return false;
        // A first read defines the initial touched value for this block. The
        // system-memory trace later binds it to the segment's initial state.
        return replay_u16_block(event.value, out.value);
    }
    if ((predecessor & MEMORY_PREDECESSOR_SEED_BIT) != 0) {
        uint32_t seed_index = predecessor & MEMORY_PREDECESSOR_INDEX_MASK;
        if (!preflight_is_write(event) || seed_index >= seeds.len()) {
            return false;
        }
        auto const &seed = seeds[seed_index];
        if (seed.address_space != preflight_address_space(event) || seed.pointer != event.pointer ||
            !replay_u16_block(seed.initial_value, out.value)) {
            return false;
        }
        out.timestamp = 0;
        return true;
    }

    size_t previous_index = predecessor - 1;
    if (previous_index >= event_index || previous_index >= memory.len()) {
        return false;
    }
    auto const &previous = memory[previous_index];
    if (preflight_address_space(previous) != preflight_address_space(event) ||
        previous.pointer != event.pointer || previous.timestamp >= event.timestamp ||
        !replay_u16_block(previous.value, out.value)) {
        return false;
    }
    out.timestamp = previous.timestamp;
    if (!preflight_is_write(event)) {
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            if (event.value[i] != out.value[i]) return false;
        }
    }
    return true;
}
