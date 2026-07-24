#include "launcher.cuh"
#include "arch/rvr/preflight.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"

static constexpr uint32_t NUM_PHANTOM_OPERANDS = 3;

struct PhantomRecord {
    uint32_t pc;
    uint32_t operands[NUM_PHANTOM_OPERANDS];
    uint32_t timestamp;
};

template <typename T> struct PhantomCols {
    T pc;
    T operands[NUM_PHANTOM_OPERANDS];
    T timestamp;
    T is_valid;
};

__global__ void phantom_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<PhantomRecord> records
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        COL_WRITE_VALUE(row, PhantomCols, pc, rec.pc);
        COL_WRITE_ARRAY(row, PhantomCols, operands, rec.operands);
        COL_WRITE_VALUE(row, PhantomCols, timestamp, rec.timestamp);
        COL_WRITE_VALUE(row, PhantomCols, is_valid, Fp::one());
    } else {
        row.fill_zero(0, width);
    }
}

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
    size_t program_index = step.program_index;
    if (program_index + 1 >= program.len()) {
        preflight_set_error(error, 231);
        return;
    }

    auto const &from = program[program_index];
    auto const &to = program[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % 4 != 0 ||
        from.pc > UINT32_MAX - 4 || from.timestamp == UINT32_MAX ||
        to.pc != from.pc + 4 || to.timestamp != from.timestamp + 1) {
        preflight_set_error(error, 232);
        return;
    }

    size_t instruction_index = (from.pc - pc_base) / 4;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 233);
        return;
    }
    auto const &instruction = instructions[instruction_index];
    if (instruction.words[0] != phantom_opcode || instruction.words[4] != 0 ||
        instruction.words[5] != 0 || instruction.words[6] != 0 ||
        instruction.words[7] != 0) {
        preflight_set_error(error, 234);
        return;
    }

    size_t memory_start = step.memory_start;
    if (memory_start > memory.len() ||
        (memory_start < memory.len() && memory[memory_start].timestamp < to.timestamp)) {
        preflight_set_error(error, 235);
        return;
    }

    uint32_t operands[NUM_PHANTOM_OPERANDS] = {
        instruction.words[1], instruction.words[2], instruction.words[3]
    };
    COL_WRITE_VALUE(row, PhantomCols, pc, from.pc);
    COL_WRITE_ARRAY(row, PhantomCols, operands, operands);
    COL_WRITE_VALUE(row, PhantomCols, timestamp, from.timestamp);
    COL_WRITE_VALUE(row, PhantomCols, is_valid, Fp::one());
}

extern "C" int _phantom_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<PhantomRecord> d_records,
    cudaStream_t stream
) {
    assert(width == sizeof(PhantomCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    phantom_tracegen<<<grid, block, 0, stream>>>(d_trace, height, width, d_records);
    return CHECK_KERNEL();
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

    auto [grid, block] = kernel_launch_params(height);
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
