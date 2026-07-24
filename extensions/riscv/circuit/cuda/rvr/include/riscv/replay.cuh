#pragma once

#include "arch/rvr/preflight.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "system/memory/params.cuh"

struct ReplayPreviousValue {
    uint32_t timestamp;
    uint16_t value[BLOCK_FE_WIDTH];
};

enum class ReplayPcEffect : uint8_t { Sequential, Dynamic };

struct ReplayProgramTransition {
    PreflightProgramEvent const *from;
    PreflightProgramEvent const *to;
    RvrReplayInstruction const *instruction;
};

/// Resolves one replay step against the immutable program and validates its
/// timestamp transition. Dynamic-PC callers must validate `to.pc` before
/// emitting a row.
static __device__ __forceinline__ bool replay_program_transition(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    size_t program_index,
    uint32_t timestamp_delta,
    ReplayPcEffect pc_effect,
    ReplayProgramTransition &out,
    uint32_t *error,
    uint32_t error_base
) {
    if (program_index >= program.len() || program.len() - program_index <= 1) {
        preflight_set_error(error, error_base);
        return false;
    }
    auto const &from = program[program_index];
    auto const &to = program[program_index + 1];
    bool invalid_pc =
        from.pc < pc_base || (from.pc - pc_base) % ::program::DEFAULT_PC_STEP != 0;
    bool invalid_timestamp = from.timestamp > UINT32_MAX - timestamp_delta ||
                             to.timestamp != from.timestamp + timestamp_delta;
    bool invalid_sequential = pc_effect == ReplayPcEffect::Sequential &&
                              (from.pc > UINT32_MAX - ::program::DEFAULT_PC_STEP ||
                               to.pc != from.pc + ::program::DEFAULT_PC_STEP);
    if (invalid_pc || invalid_timestamp || invalid_sequential) {
        preflight_set_error(error, error_base + 1);
        return false;
    }
    size_t instruction_index = (from.pc - pc_base) / ::program::DEFAULT_PC_STEP;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, error_base + 2);
        return false;
    }
    out = ReplayProgramTransition{
        .from = &from,
        .to = &to,
        .instruction = &instructions[instruction_index],
    };
    return true;
}

static __device__ bool replay_u16_block(
    uint16_t const (&source)[BLOCK_FE_WIDTH],
    uint16_t (&out)[BLOCK_FE_WIDTH]
) {
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        out[i] = source[i];
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
