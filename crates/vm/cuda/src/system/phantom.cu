#include "launcher.cuh"
#include "primitives/trace_access.h"

static constexpr uint32_t NUM_PHANTOM_OPERANDS = 4;
static constexpr uint32_t INSTRUCTION_OPERAND_MAX = (1u << 29) - 1;

template <typename T> struct PhantomCols {
    T pc;
    T operands[NUM_PHANTOM_OPERANDS];
    T timestamp;
    T is_valid;
};

#include "arch/rvr/replay.cuh"

__global__ void phantom_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t *error,
    uint32_t phantom_opcode
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(PhantomCols<uint8_t>));
    if (idx >= num_steps) return;

    auto const &step = steps[step_start + idx];
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
            851
        )) {
        return;
    }
    auto const &from = *transition.from;
    auto const &to = *transition.to;
    auto const &instruction = *transition.instruction;
    if (instruction.words[0] != phantom_opcode ||
        instruction.words[1] > INSTRUCTION_OPERAND_MAX ||
        instruction.words[2] > INSTRUCTION_OPERAND_MAX || instruction.words[3] > UINT16_MAX ||
        instruction.words[4] > UINT16_MAX || instruction.words[5] != 0 ||
        instruction.words[6] != 0 || instruction.words[7] != 0) {
        preflight_set_error(error, 854);
        return;
    }

    size_t memory_start = step.memory_start;
    if (memory_start > memory.len() ||
        (memory_start < memory.len() && memory[memory_start].timestamp < to.timestamp)) {
        preflight_set_error(error, 855);
        return;
    }

    uint32_t operands[NUM_PHANTOM_OPERANDS] = {
        instruction.words[1], instruction.words[2], instruction.words[3], instruction.words[4]
    };
    COL_WRITE_VALUE(row, PhantomCols, pc, from.pc);
    COL_WRITE_ARRAY(row, PhantomCols, operands, operands);
    COL_WRITE_VALUE(row, PhantomCols, timestamp, from.timestamp);
    COL_WRITE_VALUE(row, PhantomCols, is_valid, Fp::one());
}

extern "C" int _phantom_replay_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrReplayInstruction> d_instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> d_program,
    DeviceBufferConstView<PreflightMemoryEvent> d_memory,
    DeviceBufferConstView<RvrReplayStep> d_steps,
    size_t step_start,
    size_t num_steps,
    uint32_t *d_error,
    uint32_t phantom_opcode,
    cudaStream_t stream
) {
    assert(width == sizeof(PhantomCols<uint8_t>));
    assert(step_start <= d_steps.len());
    assert(num_steps <= d_steps.len() - step_start);
    assert(height >= num_steps);

    auto [grid, block] = kernel_launch_params(height, REPLAY_THREADS);
    phantom_replay_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_instructions,
        pc_base,
        d_program,
        d_memory,
        d_steps,
        step_start,
        num_steps,
        d_error,
        phantom_opcode
    );
    return CHECK_KERNEL();
}
