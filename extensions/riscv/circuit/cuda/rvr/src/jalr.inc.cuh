#include "arch/rvr/replay.cuh"


__global__ void jalr_replay_tracegen(
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
    uint32_t jalr_opcode,
    uint32_t register_as,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(JalrCols<uint8_t>));
    if (idx >= num_steps) return;

    auto const &step = steps[step_start + idx];
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
            201
        )) {
        return;
    }
    auto const &from = *transition.from;
    auto const &to = *transition.to;
    auto const &instruction = *transition.instruction;
    uint32_t rd_ptr = instruction.words[1];
    uint32_t rs1_ptr = instruction.words[2];
    uint32_t imm = instruction.words[3];
    uint32_t needs_write = instruction.words[6];
    uint32_t imm_sign = instruction.words[7];
    if (instruction.words[0] != jalr_opcode || instruction.words[4] != register_as ||
        instruction.words[5] != 0 || imm > UINT16_MAX || needs_write > 1 || imm_sign > 1 ||
        needs_write != (rd_ptr != 0) || !replay_canonical_register_pointer(rd_ptr) ||
        !replay_canonical_register_pointer(rs1_ptr)) {
        preflight_set_error(error, 204);
        return;
    }

    size_t read_index = step.memory_start;
    if (read_index >= memory.len() || read_index >= predecessors.len()) {
        preflight_set_error(error, 205);
        return;
    }
    auto const &read = memory[read_index];
    size_t write_index = read_index + 1;
    if (read.timestamp != from.timestamp || preflight_is_write(read) ||
        preflight_address_space(read) != register_as || read.pointer != rs1_ptr / 2) {
        preflight_set_error(error, 205);
        return;
    }
    if (needs_write) {
        if (write_index >= memory.len() || write_index >= predecessors.len()) {
            preflight_set_error(error, 205);
            return;
        }
        auto const &write = memory[write_index];
        if (write.timestamp != from.timestamp + 1 || !preflight_is_write(write) ||
            preflight_address_space(write) != register_as || write.pointer != rd_ptr / 2 ||
            (write_index + 1 < memory.len() &&
             memory[write_index + 1].timestamp < to.timestamp)) {
            preflight_set_error(error, 205);
            return;
        }
    } else if (write_index < memory.len() && memory[write_index].timestamp < to.timestamp) {
        preflight_set_error(error, 205);
        return;
    }

    uint16_t rs1[BLOCK_FE_WIDTH];
    replay_u16_block(read.value, rs1);
    if (rs1[2] != 0 || rs1[3] != 0) {
        preflight_set_error(error, 206);
        return;
    }
    uint32_t rs1_val = static_cast<uint32_t>(rs1[0]) |
                       (static_cast<uint32_t>(rs1[1]) << U16_BITS);
    uint32_t imm_extended = imm + imm_sign * 0xffff0000u;
    int64_t unaligned_signed =
        static_cast<int64_t>(rs1_val) + static_cast<int64_t>(static_cast<int32_t>(imm_extended));
    // The raw sum must fit in the implemented u32 PC domain. RISC-V then clears bit 0 before
    // checking instruction alignment (mirrors `try_run_jalr`).
    if (unaligned_signed < 0 || unaligned_signed > int64_t(UINT32_MAX)) {
        preflight_set_error(error, 209);
        return;
    }
    uint32_t raw_target_pc = static_cast<uint32_t>(unaligned_signed);
    uint32_t to_pc = raw_target_pc & ~1u;
    if (to_pc % DEFAULT_PC_STEP != 0) {
        preflight_set_error(error, 209);
        return;
    }
    if (to.pc != to_pc) {
        preflight_set_error(error, 207);
        return;
    }

    uint64_t rd = uint64_t(from.pc) + DEFAULT_PC_STEP;
    uint16_t expected_rd[BLOCK_FE_WIDTH] = {
        static_cast<uint16_t>(rd),
        static_cast<uint16_t>(rd >> U16_BITS),
        static_cast<uint16_t>(rd >> (2 * U16_BITS)),
        0,
    };
    ReplayPreviousValue write_previous = {};
    if (needs_write) {
        auto const &write = memory[write_index];
        uint16_t logged_rd[BLOCK_FE_WIDTH];
        replay_u16_block(write.value, logged_rd);
        bool matches = true;
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            matches &= logged_rd[i] == expected_rd[i];
        }
        if (!matches) {
            preflight_set_error(error, 208);
            return;
        }
    }

    ReplayPreviousValue read_previous;
    if (!replay_previous_value(
            read_index, read, predecessors[read_index], memory, seeds, read_previous
        ) ||
        (needs_write &&
         !replay_previous_value(
             write_index,
             memory[write_index],
             predecessors[write_index],
             memory,
             seeds,
             write_previous
         ))) {
        preflight_set_error(error, 210);
        return;
    }

    auto checker = VariableRangeChecker(range_checker, range_checker_num_bins);
    JalrAdapter adapter(checker, timestamp_max_bits);
    adapter.fill_trace_row(
        row,
        from.pc,
        from.timestamp,
        rs1_ptr,
        rd_ptr,
        needs_write,
        read_previous.timestamp,
        write_previous.timestamp,
        write_previous.value
    );
    JalrCore core(checker);
    core.fill_trace_row(
        row.slice_from(COL_INDEX(JalrCols, core)),
        from.pc,
        rs1_val,
        static_cast<uint16_t>(imm),
        imm_sign
    );
}



extern "C" int _jalr_replay_tracegen(
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
    uint32_t jalr_opcode,
    uint32_t register_as,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(JalrCols<uint8_t>));
    assert(d_memory.len() == d_predecessors.len());
    assert(step_start <= d_steps.len());
    assert(num_steps <= d_steps.len() - step_start);
    assert(height >= num_steps);

    auto [grid, block] = kernel_launch_params(height, REPLAY_THREADS);
    jalr_replay_tracegen<<<grid, block, 0, stream>>>(
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
        jalr_opcode,
        register_as,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
