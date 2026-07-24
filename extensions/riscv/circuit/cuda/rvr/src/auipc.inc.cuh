#include "riscv/replay.cuh"


__global__ void auipc_replay_tracegen(
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
    uint32_t auipc_opcode,
    uint32_t register_as,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(Rv64AuipcCols<uint8_t>));
    if (idx >= num_steps) return;

    auto const &step = steps[step_start + idx];
    ReplayProgramTransition transition;
    if (!replay_program_transition(
            instructions,
            pc_base,
            program,
            step.program_index,
            1u,
            ReplayPcEffect::Sequential,
            transition,
            error,
            191
        )) {
        return;
    }
    auto const &from = *transition.from;
    auto const &to = *transition.to;
    constexpr uint32_t MAX_PC = (1u << PC_BITS) - 1;
    if (from.pc > MAX_PC - DEFAULT_PC_STEP) {
        preflight_set_error(error, 192);
        return;
    }
    auto const &instruction = *transition.instruction;
    uint32_t rd_ptr = instruction.words[1];
    uint32_t imm = instruction.words[3];
    constexpr uint32_t REGISTER_FILE_BYTES = 32 * RV64_REGISTER_NUM_LIMBS;
    bool rd_is_canonical =
        rd_ptr != 0 && rd_ptr < REGISTER_FILE_BYTES && rd_ptr % RV64_REGISTER_NUM_LIMBS == 0;
    if (instruction.words[0] != auipc_opcode || !rd_is_canonical ||
        instruction.words[2] != 0 || imm >= (1u << 24) ||
        instruction.words[4] != register_as || instruction.words[5] != 0 ||
        instruction.words[6] != 0 || instruction.words[7] != 0) {
        preflight_set_error(error, 194);
        return;
    }

    size_t write_index = step.memory_start;
    if (write_index >= memory.len() || write_index >= predecessors.len()) {
        preflight_set_error(error, 195);
        return;
    }
    auto const &write = memory[write_index];
    if (write.timestamp != from.timestamp || !preflight_is_write(write) ||
        preflight_address_space(write) != register_as || write.pointer != rd_ptr / 2 ||
        (write_index + 1 < memory.len() &&
         memory[write_index + 1].timestamp < to.timestamp)) {
        preflight_set_error(error, 195);
        return;
    }

    uint16_t logged_data[BLOCK_FE_WIDTH];
    if (!replay_u16_block(write.value, logged_data)) {
        preflight_set_error(error, 196);
        return;
    }
    uint64_t expected_result = run_auipc(from.pc, imm);
    uint64_t expected_high = expected_result >> 32;
    if (expected_high != 0 && expected_high != UINT32_MAX) {
        preflight_set_error(error, 199);
        return;
    }
    uint16_t expected_data[BLOCK_FE_WIDTH] = {
        static_cast<uint16_t>(expected_result),
        static_cast<uint16_t>(expected_result >> U16_BITS),
        static_cast<uint16_t>(expected_result >> (2 * U16_BITS)),
        static_cast<uint16_t>(expected_result >> (3 * U16_BITS)),
    };
    bool matches = true;
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        matches &= logged_data[i] == expected_data[i];
    }
    if (!matches) {
        preflight_set_error(error, 197);
        return;
    }

    ReplayPreviousValue previous = {};
    if (!replay_previous_value(
            write_index, write, predecessors[write_index], memory, seeds, previous
        )) {
        preflight_set_error(error, 198);
        return;
    }

    Rv64RdWriteAdapter adapter(
        VariableRangeChecker(range_checker, range_checker_num_bins), timestamp_max_bits
    );
    adapter.fill_trace_row(
        row, from.pc, from.timestamp, rd_ptr, previous.timestamp, previous.value
    );
    Rv64AuipcCore core(VariableRangeChecker(range_checker, range_checker_num_bins));
    core.fill_trace_row(row.slice_from(COL_INDEX(Rv64AuipcCols, core)), from.pc, imm);
}



extern "C" int _auipc_replay_tracegen(
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
    uint32_t auipc_opcode,
    uint32_t register_as,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AuipcCols<uint8_t>));
    assert(d_memory.len() == d_predecessors.len());
    assert(step_start <= d_steps.len());
    assert(num_steps <= d_steps.len() - step_start);
    assert(height >= num_steps);

    auto [grid, block] = kernel_launch_params(height, 512);
    auipc_replay_tracegen<<<grid, block, 0, stream>>>(
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
        auipc_opcode,
        register_as,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
