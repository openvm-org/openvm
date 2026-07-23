#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_reg_u16.cuh"
#include "riscv/cores/add_sub.cuh"
#include "riscv/replay.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// Concrete type aliases for RV64
using Rv64AddSubCoreRecord = AddSubCoreRecord<BLOCK_FE_WIDTH>;
using Rv64AddSubCore = AddSubCore<BLOCK_FE_WIDTH, U16_BITS, true>;
template <typename T> using Rv64AddSubCoreCols = AddSubCoreCols<T, BLOCK_FE_WIDTH>;

template <typename T> struct Rv64AddSubCols {
    Rv64BaseAluRegU16AdapterCols<T> adapter;
    Rv64AddSubCoreCols<T> core;
};

struct Rv64AddSubRecord {
    Rv64BaseAluRegU16AdapterRecord adapter;
    Rv64AddSubCoreRecord core;
};

static_assert(sizeof(Rv64AddSubCoreRecord) == 18);
static_assert(sizeof(Rv64AddSubRecord) == 60);
static_assert(offsetof(Rv64AddSubRecord, core) == 40);

__global__ void add_sub_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64AddSubRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        auto adapter = Rv64BaseAluRegU16Adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        auto core =
            Rv64AddSubCore(VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64AddSubCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64AddSubCols<uint8_t>));
    }
}

extern "C" int _add_sub_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64AddSubRecord> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddSubCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    add_sub_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_range_checker, range_checker_num_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

__global__ void add_sub_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t add_step_start,
    size_t num_add_steps,
    size_t sub_step_start,
    size_t num_sub_steps,
    uint32_t *error,
    uint32_t add_opcode,
    uint32_t sub_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(Rv64AddSubCols<uint8_t>));
    size_t total_steps = num_add_steps + num_sub_steps;
    if (idx >= total_steps) return;

    bool is_add = idx < num_add_steps;
    size_t group_index = is_add ? idx : idx - num_add_steps;
    size_t step_index = (is_add ? add_step_start : sub_step_start) + group_index;
    uint32_t expected_opcode = is_add ? add_opcode : sub_opcode;
    uint8_t local_opcode = is_add ? 0 : 1;
    auto const &step = steps[step_index];
    size_t program_index = step.program_index;
    if (program_index + 1 >= program_log.len()) {
        preflight_set_error(error, 101);
        return;
    }
    auto const &from = program_log[program_index];
    auto const &to = program_log[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % 4 != 0 ||
        from.timestamp + 3 != to.timestamp || to.pc != from.pc + 4) {
        preflight_set_error(error, 102);
        return;
    }
    size_t instruction_index = (from.pc - pc_base) / 4;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 103);
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
        preflight_set_error(error, 104);
        return;
    }

    size_t rs1_index = step.memory_start;
    size_t rs2_index = rs1_index + 1;
    size_t write_index = rs1_index + 2;
    if (write_index >= memory.len() || write_index >= predecessors.len()) {
        preflight_set_error(error, 105);
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
        preflight_set_error(error, 106);
        return;
    }

    uint16_t b[BLOCK_FE_WIDTH];
    uint16_t c[BLOCK_FE_WIDTH];
    uint16_t logged_result[BLOCK_FE_WIDTH];
    if (!replay_u16_block(rs1.value, b) || !replay_u16_block(rs2.value, c) ||
        !replay_u16_block(write.value, logged_result)) {
        preflight_set_error(error, 107);
        return;
    }
    uint16_t expected_result[BLOCK_FE_WIDTH];
    uint32_t carry[BLOCK_FE_WIDTH];
    if (is_add) {
        run_add<BLOCK_FE_WIDTH, U16_BITS>(b, c, expected_result, carry);
    } else {
        run_sub<BLOCK_FE_WIDTH, U16_BITS>(b, c, expected_result, carry);
    }
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        if (logged_result[i] != expected_result[i]) {
            preflight_set_error(error, 108);
            return;
        }
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
        preflight_set_error(error, 109);
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
    auto core = Rv64AddSubCore(checker);
    core.fill_trace_row(
        row.slice_from(COL_INDEX(Rv64AddSubCols, core)), b, c, local_opcode
    );
}

extern "C" int _add_sub_replay_tracegen(
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
    size_t add_step_start,
    size_t num_add_steps,
    size_t sub_step_start,
    size_t num_sub_steps,
    uint32_t *error,
    uint32_t add_opcode,
    uint32_t sub_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddSubCols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(add_step_start <= steps.len());
    assert(num_add_steps <= steps.len() - add_step_start);
    assert(sub_step_start <= steps.len());
    assert(num_sub_steps <= steps.len() - sub_step_start);
    assert(num_add_steps <= SIZE_MAX - num_sub_steps);
    assert(height >= num_add_steps + num_sub_steps);
    auto [grid, block] = kernel_launch_params(height, 512);
    add_sub_replay_tracegen<<<grid, block, 0, stream>>>(
        trace,
        height,
        instructions,
        pc_base,
        program_log,
        memory,
        seeds,
        predecessors,
        steps,
        add_step_start,
        num_add_steps,
        sub_step_start,
        num_sub_steps,
        error,
        add_opcode,
        sub_opcode,
        register_address_space,
        range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
