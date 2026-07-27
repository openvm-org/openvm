#include "riscv/reg_reg_write_replay.cuh"


__global__ void mul_replay_tracegen(
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
    row.fill_zero(0, sizeof(Rv64MultiplicationCols<uint8_t>));
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
            601
        )) {
        return;
    }
    Rv64RegRegWriteReplay replay;
    if (!replay_reg_reg_write(
            transition,
            opcode,
            register_address_space,
            step,
            memory,
            seeds,
            predecessors,
            replay,
            error,
            604
        )) {
        return;
    }
    uint8_t expected[RV64_REGISTER_NUM_LIMBS];
    uint32_t carry[RV64_REGISTER_NUM_LIMBS];
    run_mul<RV64_REGISTER_NUM_LIMBS>(replay.rs1, replay.rs2, expected, carry);
#pragma unroll
    for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
        if (replay.result[i] != expected[i]) {
            preflight_set_error(error, 609);
            return;
        }
    }

    Rv64MultiplicationCoreRecord core_record{};
#pragma unroll
    for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
        core_record.b[i] = replay.rs1[i];
        core_record.c[i] = replay.rs2[i];
    }
    Rv64MultAdapter adapter(
        VariableRangeChecker(range_checker, range_checker_bins), timestamp_max_bits
    );
    adapter.fill_trace_row(row, replay_mult_adapter_record(replay));
    Rv64MultiplicationCore core(
        RangeTupleChecker<2>(
            range_tuple, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        ),
        BitwiseOperationLookup(bitwise_lookup)
    );
    core.fill_trace_row(row.slice_from(COL_INDEX(Rv64MultiplicationCols, core)), core_record);
}



extern "C" int _mul_replay_tracegen(
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
    assert(width == sizeof(Rv64MultiplicationCols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(step_start <= steps.len());
    assert(num_steps <= steps.len() - step_start);
    assert(height >= num_steps);
    auto [grid, block] = kernel_launch_params(height, RV64_REPLAY_THREADS);
    mul_replay_tracegen<<<grid, block, 0, stream>>>(
        trace,
        height,
        instructions,
        pc_base,
        program_log,
        memory,
        seeds,
        predecessors,
        steps,
        step_start,
        num_steps,
        error,
        opcode,
        register_address_space,
        range_checker,
        range_checker_bins,
        bitwise_lookup,
        range_tuple,
        range_tuple_sizes,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
