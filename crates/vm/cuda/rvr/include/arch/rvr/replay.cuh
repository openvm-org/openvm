#pragma once

#include "arch/rvr/preflight.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "system/memory/params.cuh"

static constexpr uint32_t REPLAY_THREADS = 256;
static constexpr uint32_t INSTRUCTION_OPERAND_MAX = (1u << 29) - 1;

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

// Decodes the canonical field encoding of a signed byte pc offset (negatives are encoded as
// p - |offset|).
static constexpr __host__ __device__ int64_t replay_decode_signed_pc_offset(uint32_t encoded) {
    constexpr uint32_t FIELD_ORDER = 2013265921u; // BabyBear
    return encoded > FIELD_ORDER / 2 ? int64_t(encoded) - int64_t(FIELD_ORDER)
                                     : int64_t(encoded);
}

// Byte target of a taken branch/jump, wrapping like the host interpreter.
static constexpr __host__ __device__ uint32_t replay_taken_branch_pc(
    uint32_t pc, uint32_t encoded
) {
    return uint32_t(int64_t(pc) + replay_decode_signed_pc_offset(encoded));
}

// True if a taken branch/jump from byte pc `pc` lands inside the implemented PC address space
// on a DEFAULT_PC_STEP-aligned slot.
static constexpr __host__ __device__ bool replay_branch_target_in_bounds(
    uint32_t pc, uint32_t encoded
) {
    int64_t target = int64_t(pc) + replay_decode_signed_pc_offset(encoded);
    return target >= 0 && target <= int64_t(::program::MAX_ALLOWED_PC) &&
           target % int64_t(::program::DEFAULT_PC_STEP) == 0;
}

static_assert(replay_canonical_register_pointer(0));
static_assert(replay_canonical_register_pointer(31u * 8u));
static_assert(!replay_canonical_register_pointer(2));
static_assert(!replay_canonical_register_pointer(32u * 8u));

static constexpr __host__ __device__ bool replay_valid_phantom_instruction(
    RvrReplayInstruction const &instruction,
    uint32_t phantom_opcode
) {
    return instruction.words[0] == phantom_opcode &&
           instruction.words[1] <= INSTRUCTION_OPERAND_MAX &&
           instruction.words[2] <= INSTRUCTION_OPERAND_MAX && instruction.words[3] <= UINT16_MAX &&
           instruction.words[4] <= UINT16_MAX && instruction.words[5] == 0 &&
           instruction.words[6] == 0 && instruction.words[7] == 0;
}

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
