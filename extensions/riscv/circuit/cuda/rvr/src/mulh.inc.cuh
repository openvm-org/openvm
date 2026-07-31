#include "riscv/reg_reg_write_replay.cuh"


__global__ void mulh_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t mulh_start, size_t mulh_count,
    size_t mulhsu_start, size_t mulhsu_count,
    size_t mulhu_start, size_t mulhu_count,
    uint32_t *error,
    uint32_t mulh_opcode, uint32_t mulhsu_opcode, uint32_t mulhu_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker, size_t range_checker_bins,
    uint32_t *bitwise_lookup,
    uint32_t *range_tuple, uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(MulHCols<uint8_t>));
    size_t total = mulh_count + mulhsu_count + mulhu_count;
    if (idx >= total) return;

    size_t step_index;
    uint32_t expected_opcode;
    MulHOpcode local_opcode;
    if (idx < mulh_count) {
        step_index = mulh_start + idx;
        expected_opcode = mulh_opcode;
        local_opcode = MULH;
    } else if (idx < mulh_count + mulhsu_count) {
        step_index = mulhsu_start + idx - mulh_count;
        expected_opcode = mulhsu_opcode;
        local_opcode = MULHSU;
    } else {
        step_index = mulhu_start + idx - mulh_count - mulhsu_count;
        expected_opcode = mulhu_opcode;
        local_opcode = MULHU;
    }
    auto const &step = steps[step_index];
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
            641
        )) {
        return;
    }
    Rv64RegRegWriteReplay replay;
    if (!replay_reg_reg_write(
            transition, expected_opcode, register_address_space,
            step, memory, seeds, predecessors, replay, error, 644
        )) return;

    uint32_t b[RV64_REGISTER_NUM_LIMBS];
    uint32_t c[RV64_REGISTER_NUM_LIMBS];
    uint32_t expected[RV64_REGISTER_NUM_LIMBS];
    uint32_t low[RV64_REGISTER_NUM_LIMBS];
    uint32_t carry[2 * RV64_REGISTER_NUM_LIMBS];
    uint32_t b_ext, c_ext;
#pragma unroll
    for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
        b[i] = replay.rs1[i];
        c[i] = replay.rs2[i];
    }
    run_mulh<RV64_REGISTER_NUM_LIMBS>(
        local_opcode, b, c, expected, low, carry, b_ext, c_ext
    );
#pragma unroll
    for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
        if (replay.result[i] != expected[i]) {
            preflight_set_error(error, 649);
            return;
        }
    }
    MulHCoreRecord<RV64_REGISTER_NUM_LIMBS> core_record{};
    core_record.local_opcode = static_cast<uint8_t>(local_opcode);
#pragma unroll
    for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
        core_record.b[i] = replay.rs1[i];
        core_record.c[i] = replay.rs2[i];
    }
    Rv64MultAdapter adapter(
        VariableRangeChecker(range_checker, range_checker_bins), timestamp_max_bits
    );
    adapter.fill_trace_row(row, replay_mult_adapter_record(replay));
    MulHCore<RV64_REGISTER_NUM_LIMBS> core(
        range_tuple, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y},
        BitwiseOperationLookup(bitwise_lookup)
    );
    core.fill_trace_row(row.slice_from(COL_INDEX(MulHCols, core)), core_record);
}



extern "C" int _mulh_replay_tracegen(
    Fp *trace, size_t height, size_t width,
    DeviceBufferConstView<RvrReplayInstruction> instructions, uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t mulh_start, size_t mulh_count,
    size_t mulhsu_start, size_t mulhsu_count,
    size_t mulhu_start, size_t mulhu_count,
    uint32_t *error,
    uint32_t mulh_opcode, uint32_t mulhsu_opcode, uint32_t mulhu_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker, size_t range_checker_bins,
    uint32_t *bitwise_lookup,
    uint32_t *range_tuple, uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits, cudaStream_t stream
) {
    assert(width == sizeof(MulHCols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(mulh_start <= steps.len() && mulh_count <= steps.len() - mulh_start);
    assert(mulhsu_start <= steps.len() && mulhsu_count <= steps.len() - mulhsu_start);
    assert(mulhu_start <= steps.len() && mulhu_count <= steps.len() - mulhu_start);
    assert(mulh_count <= SIZE_MAX - mulhsu_count);
    assert(mulh_count + mulhsu_count <= SIZE_MAX - mulhu_count);
    assert(height >= mulh_count + mulhsu_count + mulhu_count);
    auto [grid, block] = kernel_launch_params(height, RV64_REPLAY_THREADS);
    mulh_replay_tracegen<<<grid, block, 0, stream>>>(
        trace, height, instructions, pc_base, program_log, memory, seeds, predecessors, steps,
        mulh_start, mulh_count, mulhsu_start, mulhsu_count, mulhu_start, mulhu_count, error,
        mulh_opcode, mulhsu_opcode, mulhu_opcode, register_address_space, range_checker,
        range_checker_bins, bitwise_lookup, range_tuple, range_tuple_sizes, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
