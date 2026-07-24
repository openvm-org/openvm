#pragma once

#include "arch/rvr/preflight.cuh"
#include "block_hasher/variant.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "riscv/replay.cuh"
#include <cstddef>
#include <cstdint>

namespace sha2 {

using namespace program;
using namespace riscv;

inline constexpr uint32_t SHA2_REPLAY_ERROR = 901;

// Scalar locations resolved from one replay step. Values and memory auxiliaries stay in the
// shared immutable logs; this is thread-local indexing state, not a record-shaped staging buffer.
struct Sha2ReplayInput {
    uint32_t from_pc;
    uint32_t timestamp;
    uint32_t dst_reg_ptr;
    uint32_t state_reg_ptr;
    uint32_t input_reg_ptr;
    uint32_t dst_ptr;
    uint32_t state_ptr;
    uint32_t input_ptr;
    size_t register_start;
    size_t input_start;
    size_t state_start;
    size_t write_start;
};

static __device__ __forceinline__ bool sha2_canonical_register_pointer(uint32_t pointer) {
    return pointer < 32u * RV64_REGISTER_NUM_LIMBS &&
           pointer % RV64_REGISTER_NUM_LIMBS == 0;
}

static __device__ __forceinline__ uint32_t sha2_replay_pointer(
    PreflightMemoryEvent const &event,
    bool &valid
) {
    valid = event.value[2] == 0 && event.value[3] == 0;
    return uint32_t(event.value[0]) | (uint32_t(event.value[1]) << 16);
}

static __device__ __forceinline__ bool sha2_pointer_range_fits(
    uint32_t pointer,
    size_t bytes,
    uint32_t pointer_max_bits
) {
    if (pointer_max_bits > 32 || bytes == 0) return false;
    uint64_t limit = uint64_t(1) << pointer_max_bits;
    return uint64_t(pointer) + bytes <= limit;
}

static __device__ __forceinline__ uint8_t sha2_replay_byte(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    size_t first_event,
    size_t byte_offset
) {
    auto const &event = memory[first_event + byte_offset / MEMORY_BLOCK_BYTES];
    uint16_t cell = event.value[(byte_offset % MEMORY_BLOCK_BYTES) / U16_CELL_SIZE];
    return uint8_t(cell >> (8 * (byte_offset % U16_CELL_SIZE)));
}

template <typename V>
static __device__ bool replay_sha2_instruction(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    RvrReplayStep const &step,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    Sha2ReplayInput &out
) {
    size_t program_index = step.program_index;
    if (program_index + 1 >= program.len()) return false;
    auto const &from = program[program_index];
    auto const &to = program[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % DEFAULT_PC_STEP != 0 ||
        from.pc > UINT32_MAX - DEFAULT_PC_STEP || from.timestamp > UINT32_MAX - V::TIMESTAMP_DELTA ||
        to.pc != from.pc + DEFAULT_PC_STEP ||
        to.timestamp != from.timestamp + V::TIMESTAMP_DELTA) {
        return false;
    }
    size_t instruction_index = (from.pc - pc_base) / DEFAULT_PC_STEP;
    if (instruction_index >= instructions.len()) return false;
    auto const &instruction = instructions[instruction_index];
    if (instruction.words[0] != expected_opcode || instruction.words[4] != register_as ||
        instruction.words[5] != memory_as || instruction.words[6] != 0 ||
        instruction.words[7] != 0 ||
        !sha2_canonical_register_pointer(instruction.words[1]) ||
        !sha2_canonical_register_pointer(instruction.words[2]) ||
        !sha2_canonical_register_pointer(instruction.words[3])) {
        return false;
    }

    constexpr size_t NUM_EVENTS = V::TIMESTAMP_DELTA;
    size_t memory_start = step.memory_start;
    if (memory_start > memory.len() || NUM_EVENTS > memory.len() - memory_start ||
        memory.len() != predecessors.len()) {
        return false;
    }

    uint32_t pointers[SHA2_REGISTER_READS];
#pragma unroll
    for (size_t i = 0; i < SHA2_REGISTER_READS; i++) {
        size_t event_index = memory_start + i;
        auto const &event = memory[event_index];
        uint32_t register_pointer = instruction.words[1 + i];
        if (event.timestamp != from.timestamp + i || preflight_is_write(event) ||
            preflight_address_space(event) != register_as ||
            event.pointer != register_pointer / U16_CELL_SIZE) {
            return false;
        }
        ReplayPreviousValue previous;
        bool valid_pointer;
        pointers[i] = sha2_replay_pointer(event, valid_pointer);
        if (!valid_pointer ||
            !replay_previous_value(
                event_index, event, predecessors[event_index], memory, seeds, previous
            )) {
            return false;
        }
    }
    if (pointers[0] % MEMORY_BLOCK_BYTES != 0 || pointers[1] % MEMORY_BLOCK_BYTES != 0 ||
        pointers[2] % MEMORY_BLOCK_BYTES != 0 ||
        !sha2_pointer_range_fits(pointers[0], V::STATE_BYTES, pointer_max_bits) ||
        !sha2_pointer_range_fits(pointers[1], V::STATE_BYTES, pointer_max_bits) ||
        !sha2_pointer_range_fits(pointers[2], V::BLOCK_BYTES, pointer_max_bits)) {
        return false;
    }

    size_t input_start = memory_start + SHA2_REGISTER_READS;
    size_t state_start = input_start + V::BLOCK_READS;
    size_t write_start = state_start + V::STATE_READS;
    for (size_t i = 0; i < V::BLOCK_READS; i++) {
        size_t event_index = input_start + i;
        auto const &event = memory[event_index];
        ReplayPreviousValue previous;
        if (event.timestamp != from.timestamp + SHA2_REGISTER_READS + i ||
            preflight_is_write(event) || preflight_address_space(event) != memory_as ||
            event.pointer != (pointers[2] + i * SHA2_READ_SIZE) / U16_CELL_SIZE ||
            !replay_previous_value(
                event_index, event, predecessors[event_index], memory, seeds, previous
            )) {
            return false;
        }
    }
    for (size_t i = 0; i < V::STATE_READS; i++) {
        size_t event_index = state_start + i;
        auto const &event = memory[event_index];
        ReplayPreviousValue previous;
        if (event.timestamp !=
                from.timestamp + SHA2_REGISTER_READS + V::BLOCK_READS + i ||
            preflight_is_write(event) || preflight_address_space(event) != memory_as ||
            event.pointer != (pointers[1] + i * SHA2_READ_SIZE) / U16_CELL_SIZE ||
            !replay_previous_value(
                event_index, event, predecessors[event_index], memory, seeds, previous
            )) {
            return false;
        }
    }
    for (size_t i = 0; i < V::STATE_WRITES; i++) {
        size_t event_index = write_start + i;
        auto const &event = memory[event_index];
        ReplayPreviousValue previous;
        if (event.timestamp != from.timestamp + SHA2_REGISTER_READS + V::BLOCK_READS +
                                   V::STATE_READS + i ||
            !preflight_is_write(event) || preflight_address_space(event) != memory_as ||
            event.pointer != (pointers[0] + i * SHA2_WRITE_SIZE) / U16_CELL_SIZE ||
            !replay_previous_value(
                event_index, event, predecessors[event_index], memory, seeds, previous
            )) {
            return false;
        }
    }
    // Slice exhaustion is implied: NUM_EVENTS == TIMESTAMP_DELTA pins one event to every
    // clock slot in [from.timestamp, to.timestamp), and chronology preparation enforces
    // globally strictly increasing event timestamps, so no extra event can fit the interval.

    out = {
        from.pc,
        from.timestamp,
        instruction.words[1],
        instruction.words[2],
        instruction.words[3],
        pointers[0],
        pointers[1],
        pointers[2],
        memory_start,
        input_start,
        state_start,
        write_start,
    };
    return true;
}

} // namespace sha2
