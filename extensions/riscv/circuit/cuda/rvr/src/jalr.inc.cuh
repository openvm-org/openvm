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
    row.fill_zero(0, sizeof(Rv64JalrCols<uint8_t>));
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
    constexpr uint32_t REGISTER_FILE_BYTES = 32 * RV64_REGISTER_NUM_LIMBS;
    bool rd_is_canonical =
        rd_ptr < REGISTER_FILE_BYTES && rd_ptr % RV64_REGISTER_NUM_LIMBS == 0;
    bool rs1_is_canonical =
        rs1_ptr < REGISTER_FILE_BYTES && rs1_ptr % RV64_REGISTER_NUM_LIMBS == 0;
    if (instruction.words[0] != jalr_opcode || instruction.words[4] != register_as ||
        instruction.words[5] != 0 || imm > UINT16_MAX || needs_write > 1 || imm_sign > 1 ||
        needs_write != (rd_ptr != 0) || !rd_is_canonical || !rs1_is_canonical) {
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
    if (!replay_u16_block(read.value, rs1) || rs1[2] != 0 || rs1[3] != 0) {
        preflight_set_error(error, 206);
        return;
    }
    uint32_t rs1_val = static_cast<uint32_t>(rs1[0]) |
                       (static_cast<uint32_t>(rs1[1]) << U16_BITS);
    uint32_t imm_extended = imm + imm_sign * 0xffff0000u;
    int64_t unaligned_signed =
        static_cast<int64_t>(rs1_val) + static_cast<int64_t>(static_cast<int32_t>(imm_extended));
    if (unaligned_signed < 0 ||
        static_cast<uint64_t>(unaligned_signed) >= (uint64_t(1) << PC_BITS)) {
        preflight_set_error(error, 209);
        return;
    }
    constexpr uint32_t MAX_PC = (1u << PC_BITS) - 1;
    if (from.pc > MAX_PC - DEFAULT_PC_STEP) {
        preflight_set_error(error, 209);
        return;
    }
    uint32_t unaligned_to_pc = static_cast<uint32_t>(unaligned_signed);
    if (to.pc != (unaligned_to_pc & ~1u)) {
        preflight_set_error(error, 207);
        return;
    }

    uint16_t expected_rd[BLOCK_FE_WIDTH] = {
        static_cast<uint16_t>(from.pc + DEFAULT_PC_STEP),
        static_cast<uint16_t>((from.pc + DEFAULT_PC_STEP) >> U16_BITS),
        0,
        0,
    };
    ReplayPreviousValue write_previous = {};
    if (needs_write) {
        auto const &write = memory[write_index];
        uint16_t logged_rd[BLOCK_FE_WIDTH];
        if (!replay_u16_block(write.value, logged_rd)) {
            preflight_set_error(error, 206);
            return;
        }
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
    Rv64JalrAdapter adapter(checker, timestamp_max_bits);
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
    Rv64JalrCore core(checker);
    core.fill_trace_row(
        row.slice_from(COL_INDEX(Rv64JalrCols, core)),
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
    assert(width == sizeof(Rv64JalrCols<uint8_t>));
    assert(d_memory.len() == d_predecessors.len());
    assert(step_start <= d_steps.len());
    assert(num_steps <= d_steps.len() - step_start);
    assert(height >= num_steps);

    auto [grid, block] = kernel_launch_params(height, 512);
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
