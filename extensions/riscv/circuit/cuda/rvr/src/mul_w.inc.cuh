#include "riscv/reg_reg_write_replay.cuh"


__global__ void rv64_mul_w_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t *error,
    uint32_t opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    size_t range_checker_bins,
    uint32_t *bitwise_lookup,
    uint32_t *range_tuple,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(Rv64MulWCols<uint8_t>));
    if (idx >= num_steps) return;
    auto const &step = steps[step_start + idx];
    ReplayProgramTransition transition;
    if (!replay_program_transition(
            instructions,
            pc_base,
            program_log,
            step.program_index,
            3u,
            ReplayPcEffect::Sequential,
            transition,
            error,
            621
        )) {
        return;
    }
    Rv64RegRegWriteReplay replay;
    if (!replay_reg_reg_write(
            transition, opcode, register_address_space, step,
            memory, seeds, predecessors, replay, error, 624
        )) return;

    uint8_t expected_low[RV64_WORD_NUM_LIMBS];
    uint32_t carry[RV64_WORD_NUM_LIMBS];
    run_mul<RV64_WORD_NUM_LIMBS>(replay.rs1, replay.rs2, expected_low, carry);
    uint8_t sign = expected_low[RV64_WORD_NUM_LIMBS - 1] >> 7;
#pragma unroll
    for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
        uint8_t expected = i < RV64_WORD_NUM_LIMBS ? expected_low[i] : (sign ? 0xff : 0);
        if (replay.result[i] != expected) {
            preflight_set_error(error, 629);
            return;
        }
    }

    Rv64MultWAdapterRecord adapter_record{};
    adapter_record.from_pc = replay.from_pc;
    adapter_record.from_timestamp = replay.from_timestamp;
    adapter_record.rd_ptr = replay.rd_ptr;
    adapter_record.rs1_ptr = replay.rs1_ptr;
    adapter_record.rs2_ptr = replay.rs2_ptr;
    adapter_record.result_sign = sign;
    adapter_record.result_word_msl = expected_low[RV64_WORD_NUM_LIMBS - 1];
    adapter_record.reads_aux[0].prev_timestamp = replay.rs1_previous_timestamp;
    adapter_record.reads_aux[1].prev_timestamp = replay.rs2_previous_timestamp;
    adapter_record.writes_aux.prev_timestamp = replay.result_previous_timestamp;
#pragma unroll
    for (size_t i = 0; i < RV64_WORD_NUM_LIMBS; i++) {
        adapter_record.rs1_high[i] = replay.rs1[RV64_WORD_NUM_LIMBS + i];
        adapter_record.rs2_high[i] = replay.rs2[RV64_WORD_NUM_LIMBS + i];
    }
#pragma unroll
    for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++)
        adapter_record.writes_aux.prev_data[i] = replay.previous_result[i];
    Rv64MulWCoreRecord core_record{};
#pragma unroll
    for (size_t i = 0; i < RV64_WORD_NUM_LIMBS; i++) {
        core_record.b[i] = replay.rs1[i];
        core_record.c[i] = replay.rs2[i];
    }
    auto bitwise = BitwiseOperationLookup(bitwise_lookup);
    Rv64MultWAdapter adapter(
        VariableRangeChecker(range_checker, range_checker_bins), bitwise, timestamp_max_bits
    );
    adapter.fill_trace_row(row, adapter_record);
    Rv64MulWCore core(
        RangeTupleChecker<2>(
            range_tuple, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        ), bitwise
    );
    core.fill_trace_row(row.slice_from(COL_INDEX(Rv64MulWCols, core)), core_record);
}



extern "C" int _rv64_mul_w_replay_tracegen(
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
    size_t step_start,
    size_t num_steps,
    uint32_t *error,
    uint32_t opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    size_t range_checker_bins,
    uint32_t *bitwise_lookup,
    uint32_t *range_tuple,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64MulWCols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(step_start <= steps.len());
    assert(num_steps <= steps.len() - step_start);
    assert(height >= num_steps);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_mul_w_replay_tracegen<<<grid, block, 0, stream>>>(
        trace, height, instructions, pc_base, program_log, memory, seeds, predecessors, steps,
        step_start, num_steps, error, opcode, register_address_space, range_checker,
        range_checker_bins, bitwise_lookup, range_tuple, range_tuple_sizes, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
