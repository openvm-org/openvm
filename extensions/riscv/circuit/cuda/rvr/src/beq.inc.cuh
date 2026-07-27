#include "arch/rvr/replay.cuh"


__global__ void beq_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t beq_step_start,
    size_t num_beq_steps,
    size_t bne_step_start,
    size_t num_bne_steps,
    uint32_t *error,
    uint32_t beq_opcode,
    uint32_t bne_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(BranchEqualCols<uint8_t>));

    size_t total_steps = num_beq_steps + num_bne_steps;
    if (idx >= total_steps) return;
    bool is_beq = idx < num_beq_steps;
    size_t group_index = is_beq ? idx : idx - num_beq_steps;
    size_t step_index = (is_beq ? beq_step_start : bne_step_start) + group_index;
    uint32_t expected_opcode = is_beq ? beq_opcode : bne_opcode;
    uint8_t local_opcode = is_beq ? 0 : 1;

    auto const &step = steps[step_index];
    ReplayProgramTransition transition;
    if (!replay_program_transition(
            instructions,
            pc_base,
            program,
            step.program_index,
            2u,
            ReplayPcEffect::Dynamic,
            transition,
            error,
            21
        )) {
        return;
    }
    auto const &from = *transition.from;
    auto const &to = *transition.to;
    auto const &instruction = *transition.instruction;
    uint32_t rs1_ptr = instruction.words[1];
    uint32_t rs2_ptr = instruction.words[2];
    uint32_t encoded_imm = instruction.words[3];
    if (instruction.words[0] != expected_opcode ||
        instruction.words[4] != register_address_space ||
        instruction.words[5] != register_address_space || (rs1_ptr & 1) != 0 ||
        (rs2_ptr & 1) != 0) {
        preflight_set_error(error, 24);
        return;
    }

    size_t first_read_index = step.memory_start;
    size_t second_read_index = first_read_index + 1;
    if (second_read_index >= memory.len() || second_read_index >= predecessors.len()) {
        preflight_set_error(error, 25);
        return;
    }
    auto const &first_read = memory[first_read_index];
    auto const &second_read = memory[second_read_index];
    if (first_read.timestamp != from.timestamp || preflight_is_write(first_read) ||
        preflight_address_space(first_read) != register_address_space ||
        first_read.pointer != rs1_ptr / 2 || second_read.timestamp != from.timestamp + 1 ||
        preflight_is_write(second_read) ||
        preflight_address_space(second_read) != register_address_space ||
        second_read.pointer != rs2_ptr / 2 ||
        (second_read_index + 1 < memory.len() &&
         memory[second_read_index + 1].timestamp < to.timestamp)) {
        preflight_set_error(error, 26);
        return;
    }

    uint16_t rs1[BLOCK_FE_WIDTH];
    uint16_t rs2[BLOCK_FE_WIDTH];
    if (!replay_u16_block(first_read.value, rs1) || !replay_u16_block(second_read.value, rs2)) {
        preflight_set_error(error, 27);
        return;
    }
    bool equal = true;
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        equal &= rs1[i] == rs2[i];
    }
    bool should_branch = is_beq ? equal : !equal;
    Fp expected_next_pc_field(from.pc);
    expected_next_pc_field += Fp(should_branch ? encoded_imm : 4);
    uint32_t expected_next_pc = expected_next_pc_field.asUInt32();
    if (to.pc != expected_next_pc) {
        preflight_set_error(error, 28);
        return;
    }

    ReplayPreviousValue first_previous;
    ReplayPreviousValue second_previous;
    if (!replay_previous_value(
            first_read_index,
            first_read,
            predecessors[first_read_index],
            memory,
            seeds,
            first_previous
        ) ||
        !replay_previous_value(
            second_read_index,
            second_read,
            predecessors[second_read_index],
            memory,
            seeds,
            second_previous
        )) {
        preflight_set_error(error, 29);
        return;
    }

    Rv64BranchAdapter adapter(
        VariableRangeChecker(range_checker, range_checker_num_bins), timestamp_max_bits
    );
    adapter.fill_trace_row(
        row,
        from.pc,
        from.timestamp,
        rs1_ptr,
        rs2_ptr,
        first_previous.timestamp,
        second_previous.timestamp
    );
    Rv64BranchEqualCoreRecord core_record{};
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        core_record.a[i] = rs1[i];
        core_record.b[i] = rs2[i];
    }
    core_record.imm = encoded_imm;
    core_record.local_opcode = local_opcode;
    Rv64BranchEqualCore core;
    core.fill_trace_row(row.slice_from(COL_INDEX(BranchEqualCols, core)), core_record);
}



extern "C" int _beq_replay_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrReplayInstruction> d_instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> d_program_log,
    DeviceBufferConstView<PreflightMemoryEvent> d_memory_log,
    DeviceBufferConstView<PreflightInitialWrite> d_initial_write_log,
    DeviceBufferConstView<uint32_t> d_memory_predecessors,
    DeviceBufferConstView<RvrReplayStep> d_steps,
    size_t beq_step_start,
    size_t num_beq_steps,
    size_t bne_step_start,
    size_t num_bne_steps,
    uint32_t *d_error,
    uint32_t beq_opcode,
    uint32_t bne_opcode,
    uint32_t register_address_space,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(BranchEqualCols<uint8_t>));
    assert(d_memory_log.len() == d_memory_predecessors.len());
    assert(beq_step_start <= d_steps.len());
    assert(num_beq_steps <= d_steps.len() - beq_step_start);
    assert(bne_step_start <= d_steps.len());
    assert(num_bne_steps <= d_steps.len() - bne_step_start);
    assert(num_beq_steps <= SIZE_MAX - num_bne_steps);
    assert(height >= num_beq_steps + num_bne_steps);
    auto [grid, block] = kernel_launch_params(height, RV64_REPLAY_THREADS);
    beq_replay_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_instructions,
        pc_base,
        d_program_log,
        d_memory_log,
        d_initial_write_log,
        d_memory_predecessors,
        d_steps,
        beq_step_start,
        num_beq_steps,
        bne_step_start,
        num_bne_steps,
        d_error,
        beq_opcode,
        bne_opcode,
        register_address_space,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
