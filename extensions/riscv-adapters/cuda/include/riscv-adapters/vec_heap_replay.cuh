#pragma once

#include "primitives/constants.h"
#include "arch/rvr/replay.cuh"

template <size_t NUM_READS, size_t BLOCKS>
struct VecHeapTraceInput {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t local_opcode;
    uint32_t rs_ptrs[NUM_READS];
    uint32_t rd_ptr;
    uint32_t rs_vals[NUM_READS];
    uint32_t rd_val;
    uint32_t rs_prev_timestamps[NUM_READS];
    uint32_t rd_prev_timestamp;
    uint32_t heap_prev_timestamps[NUM_READS][BLOCKS];
    uint32_t write_prev_timestamps[BLOCKS];
    uint16_t heap_reads[NUM_READS][BLOCKS][BLOCK_FE_WIDTH];
    uint16_t writes[BLOCKS][BLOCK_FE_WIDTH];
    uint16_t write_predecessors[BLOCKS][BLOCK_FE_WIDTH];
};

template <size_t NUM_READS, size_t BLOCKS>
constexpr size_t VEC_HEAP_TRACE_INPUT_BYTES =
    24 + 12 * NUM_READS + 12 * NUM_READS * BLOCKS + 20 * BLOCKS;

static constexpr uint32_t VEC_HEAP_REPLAY_ERROR = 0x56010001;

static __device__ bool vec_heap_replay_event(
    size_t event_index,
    uint32_t timestamp,
    uint32_t address_space,
    uint32_t pointer,
    bool is_write,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    ReplayPreviousValue &previous,
    uint32_t *error
) {
    if (event_index >= memory.len() || event_index >= predecessors.len()) {
        preflight_set_error(error, VEC_HEAP_REPLAY_ERROR);
        return false;
    }
    auto const &event = memory[event_index];
    if (event.timestamp != timestamp || preflight_address_space(event) != address_space ||
        event.pointer != pointer || preflight_is_write(event) != is_write ||
        !replay_previous_value(
            event_index, event, predecessors[event_index], memory, seeds, previous
        ) ||
        previous.timestamp >= timestamp) {
        preflight_set_error(error, VEC_HEAP_REPLAY_ERROR);
        return false;
    }
    return true;
}

template <size_t NUM_READS, size_t BLOCKS>
__global__ void vec_heap_replay_gather(
    VecHeapTraceInput<NUM_READS, BLOCKS> *output,
    size_t output_start,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t expected_opcode,
    uint32_t local_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    uint32_t *error
) {
    size_t index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (index >= num_steps) return;

    if (step_start > steps.len() || index >= steps.len() - step_start ||
        predecessors.len() != memory.len()) {
        preflight_set_error(error, VEC_HEAP_REPLAY_ERROR);
        return;
    }
    auto const &step = steps[step_start + index];
    constexpr uint32_t EVENT_COUNT = NUM_READS + 1 + NUM_READS * BLOCKS + BLOCKS;
    ReplayProgramTransition transition;
    if (resolve_replay_program_transition(
            instructions,
            pc_base,
            program,
            step.program_index,
            EVENT_COUNT,
            ReplayPcEffect::Sequential,
            transition
        ) != ReplayProgramTransitionError::None) {
        preflight_set_error(error, VEC_HEAP_REPLAY_ERROR);
        return;
    }
    auto const &from = *transition.from;
    auto const &to = *transition.to;
    auto const &instruction = *transition.instruction;
    if (instruction.words[0] != expected_opcode || instruction.words[4] != register_as ||
        instruction.words[5] != memory_as || instruction.words[6] != 0 ||
        instruction.words[7] != 0 ||
        !replay_canonical_register_pointer(instruction.words[1]) ||
        !replay_canonical_register_pointer(instruction.words[2]) ||
        (NUM_READS == 1 && instruction.words[3] != 0) ||
        (NUM_READS == 2 && !replay_canonical_register_pointer(instruction.words[3]))) {
        preflight_set_error(error, VEC_HEAP_REPLAY_ERROR);
        return;
    }

    // A rejected row must not mutate the projection buffer. Accumulate every
    // field locally and publish it only after all replay checks have passed.
    VecHeapTraceInput<NUM_READS, BLOCKS> input = {};
    input.from_pc = from.pc;
    input.from_timestamp = from.timestamp;
    input.local_opcode = local_opcode;
    input.rd_ptr = instruction.words[1];
    for (size_t read = 0; read < NUM_READS; read++) {
        input.rs_ptrs[read] = instruction.words[2 + read];
    }

    ReplayPreviousValue previous;
    size_t cursor = step.memory_start;
    for (size_t read = 0; read < NUM_READS; read++, cursor++) {
        if (!vec_heap_replay_event(
                cursor,
                from.timestamp + static_cast<uint32_t>(read),
                register_as,
                input.rs_ptrs[read] / 2,
                false,
                memory,
                seeds,
                predecessors,
                previous,
                error
            )) {
            return;
        }
        auto const &event = memory[cursor];
        if (event.value[2] != 0 || event.value[3] != 0) {
            preflight_set_error(error, VEC_HEAP_REPLAY_ERROR);
            return;
        }
        input.rs_vals[read] = static_cast<uint32_t>(event.value[0]) |
                              (static_cast<uint32_t>(event.value[1]) << openvm::U16_BITS);
        input.rs_prev_timestamps[read] = previous.timestamp;
    }
    if (!vec_heap_replay_event(
            cursor,
            from.timestamp + NUM_READS,
            register_as,
            input.rd_ptr / 2,
            false,
            memory,
            seeds,
            predecessors,
            previous,
            error
        )) {
        return;
    }
    auto const &rd_event = memory[cursor++];
    if (rd_event.value[2] != 0 || rd_event.value[3] != 0) {
        preflight_set_error(error, VEC_HEAP_REPLAY_ERROR);
        return;
    }
    input.rd_val = static_cast<uint32_t>(rd_event.value[0]) |
                   (static_cast<uint32_t>(rd_event.value[1]) << openvm::U16_BITS);
    input.rd_prev_timestamp = previous.timestamp;

    uint64_t pointer_limit = pointer_max_bits < 32 ? uint64_t(1) << pointer_max_bits
                                                   : uint64_t(1) << 32;
    if (pointer_max_bits > 32) {
        preflight_set_error(error, VEC_HEAP_REPLAY_ERROR);
        return;
    }
    for (size_t read = 0; read < NUM_READS; read++) {
        uint64_t end = static_cast<uint64_t>(input.rs_vals[read]) + BLOCKS * MEMORY_BLOCK_BYTES;
        if ((input.rs_vals[read] & 1) != 0 || end > pointer_limit) {
            preflight_set_error(error, VEC_HEAP_REPLAY_ERROR);
            return;
        }
        for (size_t block = 0; block < BLOCKS; block++, cursor++) {
            uint32_t timestamp =
                from.timestamp + NUM_READS + 1 + read * BLOCKS + block;
            uint32_t pointer =
                (input.rs_vals[read] + block * MEMORY_BLOCK_BYTES) / U16_CELL_SIZE;
            if (!vec_heap_replay_event(
                    cursor,
                    timestamp,
                    memory_as,
                    pointer,
                    false,
                    memory,
                    seeds,
                    predecessors,
                    previous,
                    error
                )) {
                return;
            }
            input.heap_prev_timestamps[read][block] = previous.timestamp;
            for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
                input.heap_reads[read][block][limb] = memory[cursor].value[limb];
            }
        }
    }

    uint64_t write_end = static_cast<uint64_t>(input.rd_val) + BLOCKS * MEMORY_BLOCK_BYTES;
    if ((input.rd_val & 1) != 0 || write_end > pointer_limit) {
        preflight_set_error(error, VEC_HEAP_REPLAY_ERROR);
        return;
    }
    for (size_t block = 0; block < BLOCKS; block++, cursor++) {
        uint32_t timestamp = from.timestamp + NUM_READS + 1 + NUM_READS * BLOCKS + block;
        uint32_t pointer = (input.rd_val + block * MEMORY_BLOCK_BYTES) / U16_CELL_SIZE;
        if (!vec_heap_replay_event(
                cursor,
                timestamp,
                memory_as,
                pointer,
                true,
                memory,
                seeds,
                predecessors,
                previous,
                error
            )) {
            return;
        }
        input.write_prev_timestamps[block] = previous.timestamp;
        for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
            input.writes[block][limb] = memory[cursor].value[limb];
            input.write_predecessors[block][limb] = previous.value[limb];
        }
    }
    if (cursor != step.memory_start + EVENT_COUNT ||
        (cursor < memory.len() && memory[cursor].timestamp < to.timestamp)) {
        preflight_set_error(error, VEC_HEAP_REPLAY_ERROR);
        return;
    }
    output[output_start + index] = input;
}
