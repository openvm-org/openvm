#include "arch/rvr/replay.cuh"

namespace {

static constexpr uint32_t REVEAL_REPLAY_ERROR = 721;

struct ReplayRevealInput {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t src_ptr;
    ReplayPreviousValue src_previous;
    uint16_t src_data[BLOCK_FE_WIDTH];
};

static __device__ bool replay_reveal_instruction(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    RvrReplayStep const &step,
    uint32_t opcode,
    uint32_t register_as,
    ReplayRevealInput &out,
    uint32_t *error
) {
    ReplayProgramTransition transition;
    if (!replay_program_transition(
            instructions,
            pc_base,
            program,
            step.program_index,
            1,
            ReplayPcEffect::Sequential,
            transition,
            error,
            REVEAL_REPLAY_ERROR
        )) {
        return false;
    }
    auto const &instruction = *transition.instruction;
    if (instruction.words[0] != opcode ||
        !replay_canonical_register_pointer(instruction.words[1]) ||
        instruction.words[2] != 0 || instruction.words[3] != 0 ||
        instruction.words[4] != 0 || instruction.words[5] != 0 ||
        instruction.words[6] != 0 || instruction.words[7] != 0 ||
        step.memory_start >= memory.len() || step.memory_start >= predecessors.len()) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 3);
        return false;
    }

    size_t src_index = step.memory_start;
    auto const &src_read = memory[src_index];
    if (src_read.timestamp != transition.from->timestamp || preflight_is_write(src_read) ||
        preflight_address_space(src_read) != register_as ||
        src_read.pointer != instruction.words[1] / U16_CELL_SIZE) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 4);
        return false;
    }

    ReplayPreviousValue src_previous;
    if (!replay_previous_value(
            src_index, src_read, predecessors[src_index], memory, seeds, src_previous
        )) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 5);
        return false;
    }
    size_t next_index = src_index + 1;
    if (next_index < memory.len() && memory[next_index].timestamp < transition.to->timestamp) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 6);
        return false;
    }

    out.from_pc = transition.from->pc;
    out.from_timestamp = transition.from->timestamp;
    out.src_ptr = instruction.words[1];
    out.src_previous = src_previous;
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) out.src_data[i] = src_read.value[i];
    return true;
}

__global__ void reveal_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t *error,
    uint32_t opcode,
    uint32_t register_as,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(RevealCols<uint8_t>));
    if (idx >= num_steps || *error != 0) return;

    ReplayRevealInput input{};
    if (!replay_reveal_instruction(
            instructions,
            pc_base,
            program,
            memory,
            seeds,
            predecessors,
            steps[step_start + idx],
            opcode,
            register_as,
            input,
            error
        )) {
        return;
    }

    COL_WRITE_VALUE(row, RevealCols, is_valid, 1);
    COL_WRITE_VALUE(row, RevealCols, from_state.pc, input.from_pc);
    COL_WRITE_VALUE(row, RevealCols, from_state.timestamp, input.from_timestamp);
    COL_WRITE_VALUE(row, RevealCols, src_ptr, input.src_ptr);
    COL_WRITE_ARRAY(row, RevealCols, src_data, input.src_data);
    COL_WRITE_VALUE(row, RevealCols, ordinal, idx);

    VariableRangeChecker range(range_checker, range_checker_num_bins);
    MemoryAuxColsFactory mem_helper(range, timestamp_max_bits);
    mem_helper.fill(
        row.slice_from(COL_INDEX(RevealCols, src_aux)),
        input.src_previous.timestamp,
        input.from_timestamp
    );

    if (idx + 1 < num_steps) {
        auto const &next_step = steps[step_start + idx + 1];
        if (next_step.program_index >= program.len()) {
            preflight_set_error(error, REVEAL_REPLAY_ERROR + 7);
            return;
        }
        uint32_t next_timestamp = program[next_step.program_index].timestamp;
        if (next_timestamp <= input.from_timestamp) {
            preflight_set_error(error, REVEAL_REPLAY_ERROR + 8);
            return;
        }
        uint32_t timestamp_delta = next_timestamp - input.from_timestamp - 1;
        uint32_t low_bits = min(timestamp_max_bits, range.max_bits());
        uint32_t low_mask = (1u << low_bits) - 1;
        uint32_t low = timestamp_delta & low_mask;
        uint32_t high = timestamp_delta >> low_bits;
        COL_WRITE_VALUE(row, RevealCols, has_next, 1);
        COL_WRITE_VALUE(row, RevealCols, timestamp_delta_low, low);
        range.add_count(low, low_bits);
        range.add_count(high, timestamp_max_bits - low_bits);
    }
}

} // namespace

extern "C" int _reveal_replay_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrReplayInstruction> d_instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> d_program,
    DeviceBufferConstView<PreflightMemoryEvent> d_memory,
    DeviceBufferConstView<PreflightInitialWrite> d_seeds,
    DeviceBufferConstView<uint32_t> d_predecessors,
    DeviceBufferConstView<RvrReplayStep> d_steps,
    size_t step_start,
    size_t num_steps,
    uint32_t *d_error,
    uint32_t opcode,
    uint32_t register_as,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(RevealCols<uint8_t>));
    assert(d_memory.len() == d_predecessors.len());
    assert(step_start <= d_steps.len());
    assert(num_steps <= d_steps.len() - step_start);
    assert(height >= num_steps);
    auto [grid, block] = kernel_launch_params(height, REPLAY_THREADS);
    reveal_replay_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_instructions,
        pc_base,
        d_program,
        d_memory,
        d_seeds,
        d_predecessors,
        d_steps,
        step_start,
        num_steps,
        d_error,
        opcode,
        register_as,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
