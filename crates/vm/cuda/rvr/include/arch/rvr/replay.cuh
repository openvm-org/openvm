#pragma once

#include "arch/rvr/preflight.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "system/memory/params.cuh"

static constexpr uint32_t REPLAY_THREADS = 256;

struct ReplayPreviousValue {
    uint32_t timestamp;
    uint16_t value[BLOCK_FE_WIDTH];
};

enum class ReplayPcEffect : uint8_t { Sequential, Dynamic };

enum class ReplayProgramTransitionError : uint8_t {
    None = 0,
    MissingProgramEvent = 1,
    InvalidTransition = 2,
    MissingInstruction = 3,
};

struct ReplayProgramTransition {
    PreflightProgramEvent const *from;
    PreflightProgramEvent const *to;
    RvrReplayInstruction const *instruction;
};

static constexpr __host__ __device__ bool replay_canonical_register_pointer(uint32_t pointer) {
    return pointer < 32u * 8u && (pointer & 7u) == 0;
}

static_assert(replay_canonical_register_pointer(0));
static_assert(replay_canonical_register_pointer(31u * 8u));
static_assert(!replay_canonical_register_pointer(2));
static_assert(!replay_canonical_register_pointer(32u * 8u));

static __device__ __forceinline__ RvrReplayInstruction const *resolve_replay_instruction(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    uint32_t pc,
    size_t *slot_out = nullptr
) {
    if (pc < pc_base || (pc - pc_base) % ::program::DEFAULT_PC_STEP != 0) {
        return nullptr;
    }
    size_t slot = (pc - pc_base) / ::program::DEFAULT_PC_STEP;
    if (slot >= instructions.len() || instructions[slot].words[0] == UINT32_MAX) {
        return nullptr;
    }
    if (slot_out != nullptr) {
        *slot_out = slot;
    }
    return &instructions[slot];
}

/// Resolves one replay step against the immutable program and validates its
/// timestamp transition. Dynamic-PC callers must validate `to.pc` before
/// emitting a row.
static __device__ __forceinline__ ReplayProgramTransitionError resolve_replay_program_transition(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    size_t program_index,
    uint32_t timestamp_delta,
    ReplayPcEffect pc_effect,
    ReplayProgramTransition &out
) {
    if (program_index >= program.len() || program.len() - program_index <= 1) {
        return ReplayProgramTransitionError::MissingProgramEvent;
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
        return ReplayProgramTransitionError::InvalidTransition;
    }
    auto const *instruction = resolve_replay_instruction(instructions, pc_base, from.pc);
    if (instruction == nullptr) {
        return ReplayProgramTransitionError::MissingInstruction;
    }
    out = ReplayProgramTransition{
        .from = &from,
        .to = &to,
        .instruction = instruction,
    };
    return ReplayProgramTransitionError::None;
}

/// Reports the three transition failures at `error_base`, `error_base + 1`,
/// and `error_base + 2`, respectively.
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
    ReplayProgramTransitionError transition_error = resolve_replay_program_transition(
        instructions,
        pc_base,
        program,
        program_index,
        timestamp_delta,
        pc_effect,
        out
    );
    if (transition_error != ReplayProgramTransitionError::None) {
        preflight_set_error(
            error,
            error_base + static_cast<uint32_t>(transition_error) - 1
        );
        return false;
    }
    return true;
}

static __device__ void replay_u16_block(
    uint16_t const (&source)[BLOCK_FE_WIDTH],
    uint16_t (&out)[BLOCK_FE_WIDTH]
) {
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        out[i] = source[i];
    }
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
        replay_u16_block(event.value, out.value);
        return true;
    }
    if ((predecessor & MEMORY_PREDECESSOR_SEED_BIT) != 0) {
        uint32_t seed_index = predecessor & MEMORY_PREDECESSOR_INDEX_MASK;
        if (!preflight_is_write(event) || seed_index >= seeds.len()) {
            return false;
        }
        auto const &seed = seeds[seed_index];
        if (seed.address_space != preflight_address_space(event) || seed.pointer != event.pointer) {
            return false;
        }
        replay_u16_block(seed.initial_value, out.value);
        out.timestamp = 0;
        return true;
    }

    size_t previous_index = predecessor - 1;
    if (previous_index >= event_index || previous_index >= memory.len()) {
        return false;
    }
    auto const &previous = memory[previous_index];
    if (preflight_address_space(previous) != preflight_address_space(event) ||
        previous.pointer != event.pointer || previous.timestamp >= event.timestamp) {
        return false;
    }
    replay_u16_block(previous.value, out.value);
    out.timestamp = previous.timestamp;
    if (!preflight_is_write(event)) {
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            if (event.value[i] != out.value[i]) return false;
        }
    }
    return true;
}
