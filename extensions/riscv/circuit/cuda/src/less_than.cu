#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_reg_u16.cuh"
#include "riscv/cores/less_than.cuh"
#include "riscv/replay.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

using Rv64LessThanCoreRecord = LessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>;
using Rv64LessThanCore = LessThanCore<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64LessThanCoreCols =
    LessThanCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct LessThanCols {
    Rv64BaseAluRegU16AdapterCols<T> adapter;
    Rv64LessThanCoreCols<T> core;
};

struct LessThanRecord {
    Rv64BaseAluRegU16AdapterRecord adapter;
    Rv64LessThanCoreRecord core;
};

__global__ void rv64_less_than_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<LessThanRecord> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = Rv64BaseAluRegU16Adapter(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core =
            Rv64LessThanCore(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(LessThanCols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(LessThanCols<uint8_t>));
    }
}

extern "C" int _rv64_less_than_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<LessThanRecord> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(LessThanCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_less_than_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_range_checker, range_checker_num_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

__global__ void rv64_less_than_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t slt_step_start,
    size_t num_slt_steps,
    size_t sltu_step_start,
    size_t num_sltu_steps,
    uint32_t *error,
    uint32_t slt_opcode,
    uint32_t sltu_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(LessThanCols<uint8_t>));
    size_t total_steps = num_slt_steps + num_sltu_steps;
    if (idx >= total_steps) return;

    bool is_slt = idx < num_slt_steps;
    size_t group_index = is_slt ? idx : idx - num_slt_steps;
    size_t step_index = (is_slt ? slt_step_start : sltu_step_start) + group_index;
    uint32_t expected_opcode = is_slt ? slt_opcode : sltu_opcode;
    uint8_t local_opcode = is_slt ? 0 : 1;
    auto const &step = steps[step_index];
    size_t program_index = step.program_index;
    if (program_index + 1 >= program_log.len()) {
        preflight_set_error(error, 121);
        return;
    }
    auto const &from = program_log[program_index];
    auto const &to = program_log[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % 4 != 0 ||
        from.timestamp + 3 != to.timestamp || to.pc != from.pc + 4) {
        preflight_set_error(error, 122);
        return;
    }
    size_t instruction_index = (from.pc - pc_base) / 4;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 123);
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
        preflight_set_error(error, 124);
        return;
    }

    size_t rs1_index = step.memory_start;
    size_t rs2_index = rs1_index + 1;
    size_t write_index = rs1_index + 2;
    if (write_index >= memory.len() || write_index >= predecessors.len()) {
        preflight_set_error(error, 125);
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
        preflight_set_error(error, 126);
        return;
    }

    uint16_t b[BLOCK_FE_WIDTH];
    uint16_t c[BLOCK_FE_WIDTH];
    uint16_t logged_result[BLOCK_FE_WIDTH];
    if (!replay_u16_block(rs1.value, b) || !replay_u16_block(rs2.value, c) ||
        !replay_u16_block(write.value, logged_result)) {
        preflight_set_error(error, 127);
        return;
    }
    bool expected_result =
        run_less_than<BLOCK_FE_WIDTH, U16_BITS>(is_slt, b, c).cmp_result;
    if (logged_result[0] != expected_result || logged_result[1] != 0 ||
        logged_result[2] != 0 || logged_result[3] != 0) {
        preflight_set_error(error, 128);
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
        preflight_set_error(error, 129);
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
    auto core = Rv64LessThanCore(checker);
    core.fill_trace_row(
        row.slice_from(COL_INDEX(LessThanCols, core)), b, c, local_opcode
    );
}

extern "C" int _rv64_less_than_replay_tracegen(
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
    size_t slt_step_start,
    size_t num_slt_steps,
    size_t sltu_step_start,
    size_t num_sltu_steps,
    uint32_t *error,
    uint32_t slt_opcode,
    uint32_t sltu_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(LessThanCols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(slt_step_start <= steps.len());
    assert(num_slt_steps <= steps.len() - slt_step_start);
    assert(sltu_step_start <= steps.len());
    assert(num_sltu_steps <= steps.len() - sltu_step_start);
    assert(num_slt_steps <= SIZE_MAX - num_sltu_steps);
    assert(height >= num_slt_steps + num_sltu_steps);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_less_than_replay_tracegen<<<grid, block, 0, stream>>>(
        trace,
        height,
        instructions,
        pc_base,
        program_log,
        memory,
        seeds,
        predecessors,
        steps,
        slt_step_start,
        num_slt_steps,
        sltu_step_start,
        num_sltu_steps,
        error,
        slt_opcode,
        sltu_opcode,
        register_address_space,
        range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
