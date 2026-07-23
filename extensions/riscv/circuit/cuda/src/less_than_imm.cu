#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm_u16.cuh"
#include "riscv/cores/less_than_imm.cuh"
#include "riscv/replay.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// SLTI/SLTIU use u16 limbs and the single-read immediate adapter.
using Rv64LessThanImmCoreRecord = LessThanImmCoreRecord<BLOCK_FE_WIDTH, U16_BITS>;
using Rv64LessThanImmCore = LessThanImmCore<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64LessThanImmCoreCols = LessThanImmCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct LessThanImmCols {
    Rv64BaseAluImmU16AdapterCols<T> adapter;
    Rv64LessThanImmCoreCols<T> core;
};

struct LessThanImmRecord {
    Rv64BaseAluImmU16AdapterRecord adapter;
    Rv64LessThanImmCoreRecord core;
};

static_assert(sizeof(LessThanImmRecord) == 44);
static_assert(offsetof(LessThanImmRecord, core) == 32);

__global__ void less_than_imm_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<LessThanImmRecord> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = Rv64BaseAluImmU16Adapter(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core =
            Rv64LessThanImmCore(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(LessThanImmCols, core)), record.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _less_than_imm_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<LessThanImmRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(LessThanImmCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    less_than_imm_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

__global__ void less_than_imm_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t slti_step_start,
    size_t num_slti_steps,
    size_t sltiu_step_start,
    size_t num_sltiu_steps,
    uint32_t *error,
    uint32_t slti_opcode,
    uint32_t sltiu_opcode,
    uint32_t register_address_space,
    uint32_t immediate_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(LessThanImmCols<uint8_t>));
    size_t total_steps = num_slti_steps + num_sltiu_steps;
    if (idx >= total_steps) return;

    bool is_slti = idx < num_slti_steps;
    size_t group_index = is_slti ? idx : idx - num_slti_steps;
    size_t step_index = (is_slti ? slti_step_start : sltiu_step_start) + group_index;
    uint32_t expected_opcode = is_slti ? slti_opcode : sltiu_opcode;
    uint8_t local_opcode = is_slti ? 0 : 1;
    auto const &step = steps[step_index];
    size_t program_index = step.program_index;
    if (program_index + 1 >= program.len()) {
        preflight_set_error(error, 41);
        return;
    }
    auto const &from = program[program_index];
    auto const &to = program[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % 4 != 0 ||
        from.timestamp + 2 != to.timestamp || to.pc != from.pc + 4) {
        preflight_set_error(error, 42);
        return;
    }
    size_t instruction_index = (from.pc - pc_base) / 4;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 43);
        return;
    }
    auto const &instruction = instructions[instruction_index];
    uint32_t rd_ptr = instruction.words[1];
    uint32_t rs1_ptr = instruction.words[2];
    uint32_t encoded_imm = instruction.words[3];
    if (instruction.words[0] != expected_opcode ||
        instruction.words[4] != register_address_space ||
        instruction.words[5] != immediate_address_space || rd_ptr == 0 || (rd_ptr & 1) != 0 ||
        (rs1_ptr & 1) != 0) {
        preflight_set_error(error, 44);
        return;
    }

    size_t read_index = step.memory_start;
    size_t write_index = read_index + 1;
    if (write_index >= memory.len() || write_index >= predecessors.len()) {
        preflight_set_error(error, 45);
        return;
    }
    auto const &read = memory[read_index];
    auto const &write = memory[write_index];
    if (read.timestamp != from.timestamp || preflight_is_write(read) ||
        preflight_address_space(read) != register_address_space || read.pointer != rs1_ptr / 2 ||
        write.timestamp != from.timestamp + 1 || !preflight_is_write(write) ||
        preflight_address_space(write) != register_address_space || write.pointer != rd_ptr / 2 ||
        (write_index + 1 < memory.len() && memory[write_index + 1].timestamp < to.timestamp)) {
        preflight_set_error(error, 46);
        return;
    }

    uint32_t imm_low11 = encoded_imm & 0x7ff;
    uint32_t imm_sign = (encoded_imm >> 11) & 1;
    if (encoded_imm != imm_low11 + imm_sign * 0xfff800) {
        preflight_set_error(error, 47);
        return;
    }
    uint16_t rs1[BLOCK_FE_WIDTH];
    uint16_t logged_rd[BLOCK_FE_WIDTH];
    if (!replay_u16_block(read.value, rs1) || !replay_u16_block(write.value, logged_rd)) {
        preflight_set_error(error, 48);
        return;
    }
    uint16_t imm[BLOCK_FE_WIDTH];
    imm[0] = static_cast<uint16_t>(imm_low11 + imm_sign * 0xf800);
#pragma unroll
    for (size_t i = 1; i < BLOCK_FE_WIDTH; i++) {
        imm[i] = static_cast<uint16_t>(imm_sign * 0xffff);
    }
    bool expected_result =
        run_less_than<BLOCK_FE_WIDTH, U16_BITS>(is_slti, rs1, imm).cmp_result;
    if (logged_rd[0] != expected_result || logged_rd[1] != 0 || logged_rd[2] != 0 ||
        logged_rd[3] != 0) {
        preflight_set_error(error, 49);
        return;
    }

    ReplayPreviousValue read_previous;
    ReplayPreviousValue write_previous;
    if (!replay_previous_value(
            read_index, read, predecessors[read_index], memory, seeds, read_previous
        ) ||
        !replay_previous_value(
            write_index, write, predecessors[write_index], memory, seeds, write_previous
        )) {
        preflight_set_error(error, 50);
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
    auto core = Rv64LessThanImmCore(checker);
    core.fill_trace_row(
        row.slice_from(COL_INDEX(LessThanImmCols, core)),
        rs1,
        static_cast<uint16_t>(imm_low11),
        static_cast<uint8_t>(imm_sign),
        local_opcode
    );
}

extern "C" int _less_than_imm_replay_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t slti_step_start,
    size_t num_slti_steps,
    size_t sltiu_step_start,
    size_t num_sltiu_steps,
    uint32_t *error,
    uint32_t slti_opcode,
    uint32_t sltiu_opcode,
    uint32_t register_address_space,
    uint32_t immediate_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(LessThanImmCols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(slti_step_start <= steps.len());
    assert(num_slti_steps <= steps.len() - slti_step_start);
    assert(sltiu_step_start <= steps.len());
    assert(num_sltiu_steps <= steps.len() - sltiu_step_start);
    assert(num_slti_steps <= SIZE_MAX - num_sltiu_steps);
    assert(height >= num_slti_steps + num_sltiu_steps);
    auto [grid, block] = kernel_launch_params(height, 512);
    less_than_imm_replay_tracegen<<<grid, block, 0, stream>>>(
        trace,
        height,
        instructions,
        pc_base,
        program,
        memory,
        seeds,
        predecessors,
        steps,
        slti_step_start,
        num_slti_steps,
        sltiu_step_start,
        num_sltiu_steps,
        error,
        slti_opcode,
        sltiu_opcode,
        register_address_space,
        immediate_address_space,
        range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
