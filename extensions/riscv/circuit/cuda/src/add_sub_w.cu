#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_reg_u16.cuh"
#include "riscv/cores/add_sub.cuh"
#include "riscv/replay.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// Concrete type aliases for the 32-bit word variant on RV64. The low word is two u16 limbs and
// reuses the add_sub core; the adapter rebuilds the sign-extended 64-bit register write.
using Rv64AddSubWCoreRecord = AddSubCoreRecord<RV64_WORD_U16_LIMBS>;
using Rv64AddSubWCore = AddSubCore<RV64_WORD_U16_LIMBS, U16_BITS, false>;
template <typename T> using Rv64AddSubWCoreCols = AddSubCoreCols<T, RV64_WORD_U16_LIMBS>;

template <typename T> struct Rv64AddSubWCols {
    Rv64BaseAluWRegU16AdapterCols<T> adapter;
    Rv64AddSubWCoreCols<T> core;
};

struct Rv64AddSubWRecord {
    Rv64BaseAluWRegU16AdapterRecord adapter;
    Rv64AddSubWCoreRecord core;
};

__global__ void rv64_add_sub_w_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64AddSubWRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Rv64BaseAluWRegU16Adapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        Rv64AddSubWCore core(VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64AddSubWCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64AddSubWCols<uint8_t>));
    }
}

extern "C" int _rv64_add_sub_w_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64AddSubWRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddSubWCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_add_sub_w_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_range_checker_ptr, range_checker_num_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

__global__ void add_sub_w_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t addw_step_start,
    size_t num_addw_steps,
    size_t subw_step_start,
    size_t num_subw_steps,
    uint32_t *error,
    uint32_t addw_opcode,
    uint32_t subw_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(Rv64AddSubWCols<uint8_t>));
    size_t total_steps = num_addw_steps + num_subw_steps;
    if (idx >= total_steps) return;

    bool is_addw = idx < num_addw_steps;
    size_t group_index = is_addw ? idx : idx - num_addw_steps;
    size_t step_index = (is_addw ? addw_step_start : subw_step_start) + group_index;
    uint32_t expected_opcode = is_addw ? addw_opcode : subw_opcode;
    uint8_t local_opcode = is_addw ? 0 : 1;
    auto const &step = steps[step_index];
    size_t program_index = step.program_index;
    if (program_index + 1 >= program_log.len()) {
        preflight_set_error(error, 111);
        return;
    }
    auto const &from = program_log[program_index];
    auto const &to = program_log[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % 4 != 0 ||
        from.timestamp + 3 != to.timestamp || to.pc != from.pc + 4) {
        preflight_set_error(error, 112);
        return;
    }
    size_t instruction_index = (from.pc - pc_base) / 4;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 113);
        return;
    }
    auto const &instruction = instructions[instruction_index];
    uint32_t rd_ptr = instruction.words[1];
    uint32_t rs1_ptr = instruction.words[2];
    uint32_t rs2_ptr = instruction.words[3];
    if (instruction.words[0] != expected_opcode ||
        instruction.words[4] != register_address_space ||
        instruction.words[5] != register_address_space || rd_ptr == 0 || (rd_ptr & 1) != 0 ||
        (rs1_ptr & 1) != 0 || (rs2_ptr & 1) != 0) {
        preflight_set_error(error, 114);
        return;
    }

    size_t rs1_index = step.memory_start;
    size_t rs2_index = rs1_index + 1;
    size_t write_index = rs1_index + 2;
    if (write_index >= memory.len() || write_index >= predecessors.len()) {
        preflight_set_error(error, 115);
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
        preflight_set_error(error, 116);
        return;
    }

    uint16_t rs1_value[BLOCK_FE_WIDTH];
    uint16_t rs2_value[BLOCK_FE_WIDTH];
    uint16_t logged_result[BLOCK_FE_WIDTH];
    if (!replay_u16_block(rs1.value, rs1_value) ||
        !replay_u16_block(rs2.value, rs2_value) ||
        !replay_u16_block(write.value, logged_result)) {
        preflight_set_error(error, 117);
        return;
    }
    uint16_t b[RV64_WORD_U16_LIMBS] = {rs1_value[0], rs1_value[1]};
    uint16_t c[RV64_WORD_U16_LIMBS] = {rs2_value[0], rs2_value[1]};
    uint16_t expected_word[RV64_WORD_U16_LIMBS];
    uint32_t carry[RV64_WORD_U16_LIMBS];
    if (is_addw) {
        run_add<RV64_WORD_U16_LIMBS, U16_BITS>(b, c, expected_word, carry);
    } else {
        run_sub<RV64_WORD_U16_LIMBS, U16_BITS>(b, c, expected_word, carry);
    }
    uint16_t sign_extension =
        expected_word[RV64_WORD_U16_LIMBS - 1] >> (U16_BITS - 1) ? UINT16_MAX : 0;
    if (logged_result[0] != expected_word[0] || logged_result[1] != expected_word[1] ||
        logged_result[2] != sign_extension || logged_result[3] != sign_extension) {
        preflight_set_error(error, 118);
        return;
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
        preflight_set_error(error, 119);
        return;
    }

    uint16_t rs1_high[RV64_WORD_U16_LIMBS] = {rs1_value[2], rs1_value[3]};
    uint16_t rs2_high[RV64_WORD_U16_LIMBS] = {rs2_value[2], rs2_value[3]};
    auto checker = VariableRangeChecker(range_checker, range_checker_num_bins);
    auto adapter = Rv64BaseAluWRegU16Adapter(checker, timestamp_max_bits);
    adapter.fill_trace_row(
        row,
        from.pc,
        from.timestamp,
        rd_ptr,
        rs1_ptr,
        rs1_high,
        rs2_ptr,
        rs2_high,
        expected_word[RV64_WORD_U16_LIMBS - 1],
        rs1_previous.timestamp,
        rs2_previous.timestamp,
        write_previous.timestamp,
        write_previous.value
    );
    auto core = Rv64AddSubWCore(checker);
    core.fill_trace_row(
        row.slice_from(COL_INDEX(Rv64AddSubWCols, core)), b, c, local_opcode
    );
}

extern "C" int _add_sub_w_replay_tracegen(
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
    size_t addw_step_start,
    size_t num_addw_steps,
    size_t subw_step_start,
    size_t num_subw_steps,
    uint32_t *error,
    uint32_t addw_opcode,
    uint32_t subw_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddSubWCols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(addw_step_start <= steps.len());
    assert(num_addw_steps <= steps.len() - addw_step_start);
    assert(subw_step_start <= steps.len());
    assert(num_subw_steps <= steps.len() - subw_step_start);
    assert(num_addw_steps <= SIZE_MAX - num_subw_steps);
    assert(height >= num_addw_steps + num_subw_steps);
    auto [grid, block] = kernel_launch_params(height, 512);
    add_sub_w_replay_tracegen<<<grid, block, 0, stream>>>(
        trace,
        height,
        instructions,
        pc_base,
        program_log,
        memory,
        seeds,
        predecessors,
        steps,
        addw_step_start,
        num_addw_steps,
        subw_step_start,
        num_subw_steps,
        error,
        addw_opcode,
        subw_opcode,
        register_address_space,
        range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
