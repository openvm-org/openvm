#include "riscv/divrem_replay.cuh"
#include "riscv/reg_reg_write_replay.cuh"


__global__ void rv64_div_rem_w_replay_tracegen(
    Fp *trace, size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions, uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors, DeviceBufferConstView<RvrReplayStep> steps,
    size_t div_start, size_t div_count, size_t divu_start, size_t divu_count,
    size_t rem_start, size_t rem_count, size_t remu_start, size_t remu_count,
    uint32_t *error, uint32_t div_opcode, uint32_t divu_opcode,
    uint32_t rem_opcode, uint32_t remu_opcode, uint32_t register_address_space,
    uint32_t *range_checker, uint32_t range_checker_bins, uint32_t *bitwise_lookup,
    uint32_t *range_tuple, uint2 range_tuple_sizes, uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(Rv64DivRemWCols<uint8_t>));
    size_t total = div_count + divu_count + rem_count + remu_count;
    if (idx >= total) return;
    size_t step_index;
    uint32_t expected_opcode;
    DivRemOpcode local_opcode;
    if (idx < div_count) {
        step_index = div_start + idx; expected_opcode = div_opcode; local_opcode = DIV;
    } else if (idx < div_count + divu_count) {
        step_index = divu_start + idx - div_count; expected_opcode = divu_opcode; local_opcode = DIVU;
    } else if (idx < div_count + divu_count + rem_count) {
        step_index = rem_start + idx - div_count - divu_count;
        expected_opcode = rem_opcode; local_opcode = REM;
    } else {
        step_index = remu_start + idx - div_count - divu_count - rem_count;
        expected_opcode = remu_opcode; local_opcode = REMU;
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
            681
        )) {
        return;
    }
    Rv64RegRegWriteReplay replay;
    if (!replay_reg_reg_write(
            transition, expected_opcode, register_address_space,
            step, memory, seeds, predecessors, replay, error, 684
        )) return;
    uint8_t quotient[RV64_WORD_NUM_LIMBS];
    uint8_t remainder[RV64_WORD_NUM_LIMBS];
    replay_divrem_values<RV64_WORD_NUM_LIMBS>(
        replay.rs1, replay.rs2, local_opcode, quotient, remainder
    );
    uint8_t *expected_low = (local_opcode == DIV || local_opcode == DIVU) ? quotient : remainder;
    uint8_t sign = expected_low[RV64_WORD_NUM_LIMBS - 1] >> 7;
#pragma unroll
    for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
        uint8_t expected = i < RV64_WORD_NUM_LIMBS ? expected_low[i] : (sign ? 0xff : 0);
        if (replay.result[i] != expected) { preflight_set_error(error, 689); return; }
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
    DivRemCoreRecords<RV64_WORD_NUM_LIMBS> core_record{};
    core_record.local_opcode = static_cast<uint8_t>(local_opcode);
#pragma unroll
    for (size_t i = 0; i < RV64_WORD_NUM_LIMBS; i++) {
        core_record.b[i] = replay.rs1[i]; core_record.c[i] = replay.rs2[i];
    }
    auto bitwise = BitwiseOperationLookup(bitwise_lookup);
    Rv64MultWAdapter adapter(
        VariableRangeChecker(range_checker, range_checker_bins), bitwise, timestamp_max_bits
    );
    adapter.fill_trace_row(row, adapter_record);
    DivRemCore<RV64_WORD_NUM_LIMBS> core(
        bitwise,
        RangeTupleChecker<2>(
            range_tuple, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        )
    );
    core.fill_trace_row(row.slice_from(COL_INDEX(Rv64DivRemWCols, core)), core_record);
}



extern "C" int _rv64_div_rem_w_replay_tracegen(
    Fp *trace, size_t height, size_t width,
    DeviceBufferConstView<RvrReplayInstruction> instructions, uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors, DeviceBufferConstView<RvrReplayStep> steps,
    size_t div_start, size_t div_count, size_t divu_start, size_t divu_count,
    size_t rem_start, size_t rem_count, size_t remu_start, size_t remu_count,
    uint32_t *error, uint32_t div_opcode, uint32_t divu_opcode,
    uint32_t rem_opcode, uint32_t remu_opcode, uint32_t register_address_space,
    uint32_t *range_checker, uint32_t range_checker_bins, uint32_t *bitwise_lookup,
    uint32_t *range_tuple, uint2 range_tuple_sizes, uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64DivRemWCols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(div_start <= steps.len() && div_count <= steps.len() - div_start);
    assert(divu_start <= steps.len() && divu_count <= steps.len() - divu_start);
    assert(rem_start <= steps.len() && rem_count <= steps.len() - rem_start);
    assert(remu_start <= steps.len() && remu_count <= steps.len() - remu_start);
    size_t total = div_count + divu_count + rem_count + remu_count;
    assert(total >= div_count && total >= divu_count && total >= rem_count && total >= remu_count);
    assert(height >= total);
    auto [grid, block] = kernel_launch_params(height, 256);
    rv64_div_rem_w_replay_tracegen<<<grid, block, 0, stream>>>(
        trace, height, instructions, pc_base, program_log, memory, seeds, predecessors, steps,
        div_start, div_count, divu_start, divu_count, rem_start, rem_count, remu_start, remu_count,
        error, div_opcode, divu_opcode, rem_opcode, remu_opcode, register_address_space,
        range_checker, range_checker_bins, bitwise_lookup, range_tuple, range_tuple_sizes,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
