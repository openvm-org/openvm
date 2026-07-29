#include "arch/rvr/replay.cuh"

static constexpr uint32_t ADD_SUB_REPLAY_ERROR_BASE = 1001;

__global__ void add_sub_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t add_step_start,
    size_t num_add_steps,
    size_t sub_step_start,
    size_t num_sub_steps,
    uint32_t *error,
    uint32_t add_opcode,
    uint32_t sub_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(Rv64AddSubCols<uint8_t>));
    size_t total_steps = num_add_steps + num_sub_steps;
    if (idx >= total_steps) return;

    bool is_add = idx < num_add_steps;
    size_t group_index = is_add ? idx : idx - num_add_steps;
    size_t step_index = (is_add ? add_step_start : sub_step_start) + group_index;
    uint32_t expected_opcode = is_add ? add_opcode : sub_opcode;
    uint8_t local_opcode = is_add ? 0 : 1;
    auto const &step = steps[step_index];
    ReplayProgramTransition transition;
    if (!replay_program_transition(
            instructions,
            pc_base,
            program_log,
            step.program_index,
            3,
            ReplayPcEffect::Sequential,
            transition,
            error,
            ADD_SUB_REPLAY_ERROR_BASE
        )) {
        return;
    }
    auto const &from = *transition.from;
    auto const &to = *transition.to;
    auto const &instruction = *transition.instruction;
    uint32_t rd_ptr = instruction.words[1];
    uint32_t rs1_ptr = instruction.words[2];
    uint32_t rs2_ptr = instruction.words[3];
    if (instruction.words[0] != expected_opcode ||
        instruction.words[4] != register_address_space ||
        instruction.words[5] != register_address_space || rd_ptr == 0 || !replay_canonical_register_pointer(rd_ptr) ||
        !replay_canonical_register_pointer(rs1_ptr) || !replay_canonical_register_pointer(rs2_ptr)) {
        preflight_set_error(error, ADD_SUB_REPLAY_ERROR_BASE + 3);
        return;
    }

    size_t rs1_index = step.memory_start;
    size_t rs2_index = rs1_index + 1;
    size_t write_index = rs1_index + 2;
    if (write_index >= memory.len() || write_index >= predecessors.len()) {
        preflight_set_error(error, ADD_SUB_REPLAY_ERROR_BASE + 4);
        return;
    }
    auto const &rs1 = memory[rs1_index];
    auto const &rs2 = memory[rs2_index];
    auto const &write = memory[write_index];
    if (rs1.timestamp != from.timestamp || preflight_is_write(rs1) ||
        preflight_address_space(rs1) != register_address_space || rs1.pointer != rs1_ptr / 2 ||
        rs2.timestamp != from.timestamp + 1 || preflight_is_write(rs2) ||
        preflight_address_space(rs2) != register_address_space || rs2.pointer != rs2_ptr / 2 ||
        write.timestamp != from.timestamp + 2 || !preflight_is_write(write) ||
        preflight_address_space(write) != register_address_space || write.pointer != rd_ptr / 2 ||
        (write_index + 1 < memory.len() && memory[write_index + 1].timestamp < to.timestamp)) {
        preflight_set_error(error, ADD_SUB_REPLAY_ERROR_BASE + 5);
        return;
    }

    uint16_t b[BLOCK_FE_WIDTH];
    uint16_t c[BLOCK_FE_WIDTH];
    uint16_t logged_result[BLOCK_FE_WIDTH];
    replay_u16_block(rs1.value, b);
    replay_u16_block(rs2.value, c);
    replay_u16_block(write.value, logged_result);
    uint16_t expected_result[BLOCK_FE_WIDTH];
    uint32_t carry[BLOCK_FE_WIDTH];
    if (is_add) {
        run_add<BLOCK_FE_WIDTH, U16_BITS>(b, c, expected_result, carry);
    } else {
        run_sub<BLOCK_FE_WIDTH, U16_BITS>(b, c, expected_result, carry);
    }
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        if (logged_result[i] != expected_result[i]) {
            preflight_set_error(error, ADD_SUB_REPLAY_ERROR_BASE + 7);
            return;
        }
    }

    ReplayPreviousValue rs1_previous;
    ReplayPreviousValue rs2_previous;
    ReplayPreviousValue write_previous;
    if (!replay_previous_value(
            rs1_index, rs1, predecessors[rs1_index], memory, seeds, rs1_previous
        ) ||
        !replay_previous_value(
            rs2_index, rs2, predecessors[rs2_index], memory, seeds, rs2_previous
        ) ||
        !replay_previous_value(
            write_index, write, predecessors[write_index], memory, seeds, write_previous
        )) {
        preflight_set_error(error, ADD_SUB_REPLAY_ERROR_BASE + 8);
        return;
    }

    auto checker = VariableRangeChecker(range_checker, range_checker_num_bins);
    auto adapter = Rv64BaseAluRegU16Adapter(checker, timestamp_max_bits);
    adapter.fill_trace_row(
        row,
        from.pc,
        from.timestamp,
        rd_ptr,
        rs1_ptr,
        rs2_ptr,
        rs1_previous.timestamp,
        rs2_previous.timestamp,
        write_previous.timestamp,
        write_previous.value
    );
    auto core = Rv64AddSubCore(checker);
    core.fill_trace_row(
        row.slice_from(COL_INDEX(Rv64AddSubCols, core)), b, c, local_opcode
    );
}



extern "C" int _add_sub_replay_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t add_step_start,
    size_t num_add_steps,
    size_t sub_step_start,
    size_t num_sub_steps,
    uint32_t *error,
    uint32_t add_opcode,
    uint32_t sub_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddSubCols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(add_step_start <= steps.len());
    assert(num_add_steps <= steps.len() - add_step_start);
    assert(sub_step_start <= steps.len());
    assert(num_sub_steps <= steps.len() - sub_step_start);
    assert(num_add_steps <= SIZE_MAX - num_sub_steps);
    assert(height >= num_add_steps + num_sub_steps);
    auto [grid, block] = kernel_launch_params(height, RV64_REPLAY_THREADS);
    add_sub_replay_tracegen<<<grid, block, 0, stream>>>(
        trace,
        height,
        instructions,
        pc_base,
        program_log,
        memory,
        seeds,
        predecessors,
        steps,
        add_step_start,
        num_add_steps,
        sub_step_start,
        num_sub_steps,
        error,
        add_opcode,
        sub_opcode,
        register_address_space,
        range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
