#pragma once

#include "primitives/constants.h"
#include "riscv-adapters/vec_heap_replay.cuh"

/// Registers read per instruction: `rs1`, `rs2`, `rd`.
static constexpr size_t EC_MUL_REGISTER_READS = 3;
/// Memory blocks spanned by the 256-bit scalar operand.
static constexpr size_t EC_MUL_SCALAR_BLOCKS = 4;

/// Projection of one `EC_MUL` or `SETUP_EC_MUL` instruction. Field order matches the Rust side:
/// every `uint32_t` ahead of every `uint16_t` array, so there is no interior padding.
template <size_t BLOCKS> struct EcMulTraceInput {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t is_setup;
    uint32_t reg_ptrs[EC_MUL_REGISTER_READS];
    uint32_t reg_vals[EC_MUL_REGISTER_READS];
    uint32_t reg_prev_timestamps[EC_MUL_REGISTER_READS];
    uint32_t point_prev_timestamps[BLOCKS];
    uint32_t scalar_prev_timestamps[EC_MUL_SCALAR_BLOCKS];
    uint32_t write_prev_timestamps[BLOCKS];
    uint16_t point_blocks[BLOCKS][BLOCK_FE_WIDTH];
    uint16_t scalar_blocks[EC_MUL_SCALAR_BLOCKS][BLOCK_FE_WIDTH];
    uint16_t write_blocks[BLOCKS][BLOCK_FE_WIDTH];
    uint16_t write_predecessors[BLOCKS][BLOCK_FE_WIDTH];
};

template <size_t BLOCKS>
constexpr size_t EC_MUL_TRACE_INPUT_BYTES =
    4 * (3 + 3 * EC_MUL_REGISTER_READS + BLOCKS + EC_MUL_SCALAR_BLOCKS + BLOCKS) +
    2 * ((BLOCKS + EC_MUL_SCALAR_BLOCKS + 2 * BLOCKS) * BLOCK_FE_WIDTH);

static constexpr uint32_t EC_MUL_REPLAY_ERROR = 0x56020001;

/// Reads one contiguous run of blocks starting at `byte_base`, recording values and predecessor
/// timestamps. `timestamp_base` is the logical clock of the run's first block.
template <size_t COUNT>
static __device__ bool ec_mul_replay_blocks(
    size_t &cursor,
    uint32_t byte_base,
    uint32_t timestamp_base,
    uint32_t memory_as,
    bool is_write,
    uint64_t pointer_limit,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    uint32_t (&prev_timestamps)[COUNT],
    uint16_t (&values)[COUNT][BLOCK_FE_WIDTH],
    uint16_t (*predecessor_values)[BLOCK_FE_WIDTH],
    uint32_t *error
) {
    uint64_t end = static_cast<uint64_t>(byte_base) + COUNT * MEMORY_BLOCK_BYTES;
    if ((byte_base & 1) != 0 || end > pointer_limit) {
        preflight_set_error(error, EC_MUL_REPLAY_ERROR);
        return false;
    }
    ReplayPreviousValue previous;
    for (size_t block = 0; block < COUNT; block++, cursor++) {
        uint32_t pointer = (byte_base + block * MEMORY_BLOCK_BYTES) / U16_CELL_SIZE;
        if (!vec_heap_replay_event(
                cursor,
                timestamp_base + static_cast<uint32_t>(block),
                memory_as,
                pointer,
                is_write,
                memory,
                seeds,
                predecessors,
                previous,
                error
            )) {
            return false;
        }
        prev_timestamps[block] = previous.timestamp;
        for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
            values[block][limb] = memory[cursor].value[limb];
            if (predecessor_values != nullptr) {
                predecessor_values[block][limb] = previous.value[limb];
            }
        }
    }
    return true;
}

/// Gathers `EC_MUL` projections from the replayed memory history, in the order the AIR assigns
/// timestamps: rs1, rs2, rd, the point blocks, the scalar blocks, then the result writes.
template <size_t BLOCKS>
__global__ void ec_mul_replay_gather(
    EcMulTraceInput<BLOCKS> *output,
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
    uint32_t is_setup,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    uint32_t *error
) {
    size_t index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (index >= num_steps) return;

    if (step_start > steps.len() || index >= steps.len() - step_start ||
        predecessors.len() != memory.len() || pointer_max_bits > 32) {
        preflight_set_error(error, EC_MUL_REPLAY_ERROR);
        return;
    }
    auto const &step = steps[step_start + index];
    constexpr uint32_t EVENT_COUNT =
        EC_MUL_REGISTER_READS + BLOCKS + EC_MUL_SCALAR_BLOCKS + BLOCKS;
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
        preflight_set_error(error, EC_MUL_REPLAY_ERROR);
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
        !replay_canonical_register_pointer(instruction.words[3])) {
        preflight_set_error(error, EC_MUL_REPLAY_ERROR);
        return;
    }

    // A rejected row must not mutate the projection buffer. Accumulate locally and publish only
    // after every replay check has passed, as the vec-heap gather does.
    EcMulTraceInput<BLOCKS> input = {};
    input.from_pc = from.pc;
    input.from_timestamp = from.timestamp;
    input.is_setup = is_setup;
    // Timestamp order is rs1, rs2, rd; the instruction encodes rd first.
    input.reg_ptrs[0] = instruction.words[2];
    input.reg_ptrs[1] = instruction.words[3];
    input.reg_ptrs[2] = instruction.words[1];

    ReplayPreviousValue previous;
    size_t cursor = step.memory_start;
    for (size_t reg = 0; reg < EC_MUL_REGISTER_READS; reg++, cursor++) {
        if (!vec_heap_replay_event(
                cursor,
                from.timestamp + static_cast<uint32_t>(reg),
                register_as,
                input.reg_ptrs[reg] / U16_CELL_SIZE,
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
            preflight_set_error(error, EC_MUL_REPLAY_ERROR);
            return;
        }
        input.reg_vals[reg] = static_cast<uint32_t>(event.value[0]) |
                              (static_cast<uint32_t>(event.value[1]) << openvm::U16_BITS);
        input.reg_prev_timestamps[reg] = previous.timestamp;
    }

    uint64_t pointer_limit =
        pointer_max_bits < 32 ? uint64_t(1) << pointer_max_bits : uint64_t(1) << 32;
    uint32_t point_base = from.timestamp + EC_MUL_REGISTER_READS;
    if (!ec_mul_replay_blocks<BLOCKS>(
            cursor, input.reg_vals[0], point_base, memory_as, false, pointer_limit, memory, seeds,
            predecessors, input.point_prev_timestamps, input.point_blocks, nullptr, error
        )) {
        return;
    }
    if (!ec_mul_replay_blocks<EC_MUL_SCALAR_BLOCKS>(
            cursor, input.reg_vals[1], point_base + BLOCKS, memory_as, false, pointer_limit, memory,
            seeds, predecessors, input.scalar_prev_timestamps, input.scalar_blocks, nullptr, error
        )) {
        return;
    }
    if (!ec_mul_replay_blocks<BLOCKS>(
            cursor,
            input.reg_vals[2],
            point_base + BLOCKS + EC_MUL_SCALAR_BLOCKS,
            memory_as,
            true,
            pointer_limit,
            memory,
            seeds,
            predecessors,
            input.write_prev_timestamps,
            input.write_blocks,
            input.write_predecessors,
            error
        )) {
        return;
    }

    if (cursor != step.memory_start + EVENT_COUNT ||
        (cursor < memory.len() && memory[cursor].timestamp < to.timestamp)) {
        preflight_set_error(error, EC_MUL_REPLAY_ERROR);
        return;
    }
    output[output_start + index] = input;
}
