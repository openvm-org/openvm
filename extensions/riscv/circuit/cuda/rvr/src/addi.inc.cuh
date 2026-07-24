#include "riscv/replay.cuh"


__global__ void addi_replay_tracegen(
    Fp *d_trace,
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
    uint32_t addi_opcode,
    uint32_t register_address_space,
    uint32_t immediate_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height) {
        return;
    }
    RowSlice row(d_trace + idx, height);
    row.fill_zero(0, sizeof(Rv64AddICols<uint8_t>));
    if (idx >= num_steps) {
        return;
    }

    auto const &step = steps[step_start + idx];
    ReplayProgramTransition transition;
    if (!replay_program_transition(
            instructions,
            pc_base,
            program,
            step.program_index,
            2,
            ReplayPcEffect::Sequential,
            transition,
            error,
            1
        )) {
        return;
    }
    auto const &from = *transition.from;
    auto const &to = *transition.to;
    auto const &instruction = *transition.instruction;
    uint32_t rd_ptr = instruction.words[1];
    uint32_t rs1_ptr = instruction.words[2];
    uint32_t encoded_imm = instruction.words[3];
    if (instruction.words[0] != addi_opcode || instruction.words[4] != register_address_space ||
        instruction.words[5] != immediate_address_space || rd_ptr == 0 || (rd_ptr & 1) != 0 ||
        (rs1_ptr & 1) != 0) {
        preflight_set_error(error, 4);
        return;
    }

    size_t read_index = step.memory_start;
    size_t write_index = read_index + 1;
    if (write_index >= memory.len() || write_index >= predecessors.len()) {
        preflight_set_error(error, 5);
        return;
    }
    auto const &read = memory[read_index];
    auto const &write = memory[write_index];
    if (read.timestamp != from.timestamp || preflight_is_write(read) ||
        preflight_address_space(read) != register_address_space || read.pointer != rs1_ptr / 2 ||
        write.timestamp != from.timestamp + 1 || !preflight_is_write(write) ||
        preflight_address_space(write) != register_address_space || write.pointer != rd_ptr / 2 ||
        (write_index + 1 < memory.len() && memory[write_index + 1].timestamp < to.timestamp)) {
        preflight_set_error(error, 6);
        return;
    }

    uint32_t imm_low11 = encoded_imm & 0x7ff;
    uint32_t imm_sign = (encoded_imm >> 11) & 1;
    if (encoded_imm != imm_low11 + imm_sign * 0xfff800) {
        preflight_set_error(error, 7);
        return;
    }
    uint16_t rs1[BLOCK_FE_WIDTH];
    uint16_t logged_rd[BLOCK_FE_WIDTH];
    if (!replay_u16_block(read.value, rs1) || !replay_u16_block(write.value, logged_rd)) {
        preflight_set_error(error, 8);
        return;
    }
    uint16_t expected_rd[BLOCK_FE_WIDTH];
    uint32_t overflow = static_cast<uint32_t>(rs1[0]) + imm_low11 +
                        imm_sign * ((1u << U16_BITS) - (1u << 11));
    uint32_t carry = overflow >> U16_BITS;
    expected_rd[0] = overflow & ((1u << U16_BITS) - 1);
#pragma unroll
    for (size_t i = 1; i < BLOCK_FE_WIDTH; i++) {
        overflow = static_cast<uint32_t>(rs1[i]) +
                   imm_sign * ((1u << U16_BITS) - 1) + carry;
        carry = overflow >> U16_BITS;
        expected_rd[i] = overflow & ((1u << U16_BITS) - 1);
    }
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        if (expected_rd[i] != logged_rd[i]) {
            preflight_set_error(error, 9);
            return;
        }
    }

    ReplayPreviousValue read_previous;
    ReplayPreviousValue write_previous;
    if (!replay_previous_value(
            read_index, read, predecessors[read_index], memory, seeds, read_previous
        ) ||
        !replay_previous_value(
            write_index, write, predecessors[write_index], memory, seeds, write_previous
        )) {
        preflight_set_error(error, 10);
        return;
    }

    auto checker = VariableRangeChecker(range_checker, range_checker_num_bins);
    auto adapter = Rv64BaseAluImmU16Adapter(checker, timestamp_max_bits);
    adapter.fill_trace_row(
        row,
        from.pc,
        from.timestamp,
        rd_ptr,
        rs1_ptr,
        read_previous.timestamp,
        write_previous.timestamp,
        write_previous.value
    );
    auto core = Rv64AddICore(checker);
    core.fill_trace_row(
        row.slice_from(COL_INDEX(Rv64AddICols, core)),
        rs1,
        static_cast<uint16_t>(imm_low11),
        static_cast<uint16_t>(imm_sign)
    );
}



extern "C" int _addi_replay_tracegen(
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
    size_t step_start,
    size_t num_steps,
    uint32_t *d_error,
    uint32_t addi_opcode,
    uint32_t register_address_space,
    uint32_t immediate_address_space,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddICols<uint8_t>));
    assert(d_memory_log.len() == d_memory_predecessors.len());
    assert(step_start <= d_steps.len());
    assert(num_steps <= d_steps.len() - step_start);
    assert(height >= num_steps);
    auto [grid, block] = kernel_launch_params(height);
    addi_replay_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_instructions,
        pc_base,
        d_program_log,
        d_memory_log,
        d_initial_write_log,
        d_memory_predecessors,
        d_steps,
        step_start,
        num_steps,
        d_error,
        addi_opcode,
        register_address_space,
        immediate_address_space,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
