#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/constants.h"
#include "riscv/adapters/branch.cuh" // Rv64BranchAdapterCols, Rv64BranchAdapterRecord, Rv64BranchAdapter
#include "riscv/cores/blt.cuh"
#include "riscv/cores/less_than.cuh"
#include "riscv/replay.cuh"
#include "system/memory/params.cuh" // BLOCK_FE_WIDTH

using namespace riscv;

using Rv64BranchLessThanCoreRecord =
    BranchLessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>;
using Rv64BranchLessThanCore = BranchLessThanCore<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64BranchLessThanCoreCols =
    BranchLessThanCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct BranchLessThanCols {
    Rv64BranchAdapterCols<T> adapter;
    Rv64BranchLessThanCoreCols<T> core;
};

struct BranchLessThanRecord {
    Rv64BranchAdapterRecord adapter;
    Rv64BranchLessThanCoreRecord core;
};

__global__ void blt_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<BranchLessThanRecord> records,
    uint32_t *rc_ptr,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < records.len()) {
        auto const &full_record = records[idx];

        Rv64BranchAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, full_record.adapter);

        Rv64BranchLessThanCore core{VariableRangeChecker(rc_ptr, rc_bins)};
        core.fill_trace_row(row.slice_from(COL_INDEX(BranchLessThanCols, core)), full_record.core);
    } else {
        row.fill_zero(0, sizeof(BranchLessThanCols<uint8_t>));
    }
}

extern "C" int _blt_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BranchLessThanRecord> d_records,
    uint32_t *d_rc,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(BranchLessThanCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height, 512);
    blt_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_rc, rc_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

__global__ void blt_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t blt_step_start,
    size_t num_blt_steps,
    size_t bltu_step_start,
    size_t num_bltu_steps,
    size_t bge_step_start,
    size_t num_bge_steps,
    size_t bgeu_step_start,
    size_t num_bgeu_steps,
    uint32_t *error,
    uint32_t blt_opcode,
    uint32_t bltu_opcode,
    uint32_t bge_opcode,
    uint32_t bgeu_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(BranchLessThanCols<uint8_t>));

    size_t total_steps = num_blt_steps + num_bltu_steps + num_bge_steps + num_bgeu_steps;
    if (idx >= total_steps) return;

    size_t group_index;
    size_t step_index;
    uint32_t expected_opcode;
    uint8_t local_opcode;
    if (idx < num_blt_steps) {
        group_index = idx;
        step_index = blt_step_start + group_index;
        expected_opcode = blt_opcode;
        local_opcode = 0;
    } else if (idx < num_blt_steps + num_bltu_steps) {
        group_index = idx - num_blt_steps;
        step_index = bltu_step_start + group_index;
        expected_opcode = bltu_opcode;
        local_opcode = 1;
    } else if (idx < num_blt_steps + num_bltu_steps + num_bge_steps) {
        group_index = idx - num_blt_steps - num_bltu_steps;
        step_index = bge_step_start + group_index;
        expected_opcode = bge_opcode;
        local_opcode = 2;
    } else {
        group_index = idx - num_blt_steps - num_bltu_steps - num_bge_steps;
        step_index = bgeu_step_start + group_index;
        expected_opcode = bgeu_opcode;
        local_opcode = 3;
    }

    auto const &step = steps[step_index];
    size_t program_index = step.program_index;
    if (program_index + 1 >= program.len()) {
        preflight_set_error(error, 11);
        return;
    }
    auto const &from = program[program_index];
    auto const &to = program[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % 4 != 0 ||
        from.timestamp + 2 != to.timestamp) {
        preflight_set_error(error, 12);
        return;
    }
    size_t instruction_index = (from.pc - pc_base) / 4;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 13);
        return;
    }
    auto const &instruction = instructions[instruction_index];
    uint32_t rs1_ptr = instruction.words[1];
    uint32_t rs2_ptr = instruction.words[2];
    uint32_t encoded_imm = instruction.words[3];
    if (instruction.words[0] != expected_opcode ||
        instruction.words[4] != register_address_space ||
        instruction.words[5] != register_address_space || (rs1_ptr & 1) != 0 ||
        (rs2_ptr & 1) != 0) {
        preflight_set_error(error, 14);
        return;
    }

    size_t first_read_index = step.memory_start;
    size_t second_read_index = first_read_index + 1;
    if (second_read_index >= memory.len() || second_read_index >= predecessors.len()) {
        preflight_set_error(error, 15);
        return;
    }
    auto const &first_read = memory[first_read_index];
    auto const &second_read = memory[second_read_index];
    if (first_read.timestamp != from.timestamp || preflight_is_write(first_read) ||
        preflight_address_space(first_read) != register_address_space ||
        first_read.pointer != rs1_ptr / 2 || second_read.timestamp != from.timestamp + 1 ||
        preflight_is_write(second_read) ||
        preflight_address_space(second_read) != register_address_space ||
        second_read.pointer != rs2_ptr / 2 ||
        (second_read_index + 1 < memory.len() &&
         memory[second_read_index + 1].timestamp < to.timestamp)) {
        preflight_set_error(error, 16);
        return;
    }

    uint16_t rs1[BLOCK_FE_WIDTH];
    uint16_t rs2[BLOCK_FE_WIDTH];
    if (!replay_u16_block(first_read.value, rs1) || !replay_u16_block(second_read.value, rs2)) {
        preflight_set_error(error, 17);
        return;
    }

    bool signed_op = local_opcode == 0 || local_opcode == 2;
    bool ge_op = local_opcode == 2 || local_opcode == 3;
    bool cmp_lt =
        run_less_than<BLOCK_FE_WIDTH, U16_BITS>(signed_op, rs1, rs2).cmp_result;
    bool should_branch = ge_op ? !cmp_lt : cmp_lt;
    Fp expected_next_pc_field(from.pc);
    expected_next_pc_field += Fp(should_branch ? encoded_imm : 4);
    if (to.pc != expected_next_pc_field.asUInt32()) {
        preflight_set_error(error, 18);
        return;
    }

    ReplayPreviousValue first_previous;
    ReplayPreviousValue second_previous;
    if (!replay_previous_value(
            first_read_index,
            first_read,
            predecessors[first_read_index],
            memory,
            seeds,
            first_previous
        ) ||
        !replay_previous_value(
            second_read_index,
            second_read,
            predecessors[second_read_index],
            memory,
            seeds,
            second_previous
        )) {
        preflight_set_error(error, 19);
        return;
    }

    Rv64BranchAdapter adapter(
        VariableRangeChecker(range_checker, range_checker_num_bins), timestamp_max_bits
    );
    adapter.fill_trace_row(
        row,
        from.pc,
        from.timestamp,
        rs1_ptr,
        rs2_ptr,
        first_previous.timestamp,
        second_previous.timestamp
    );
    Rv64BranchLessThanCoreRecord core_record{};
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        core_record.a[i] = rs1[i];
        core_record.b[i] = rs2[i];
    }
    core_record.imm = encoded_imm;
    core_record.local_opcode = local_opcode;
    Rv64BranchLessThanCore core{VariableRangeChecker(range_checker, range_checker_num_bins)};
    core.fill_trace_row(row.slice_from(COL_INDEX(BranchLessThanCols, core)), core_record);
}

extern "C" int _blt_replay_tracegen(
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
    size_t blt_step_start,
    size_t num_blt_steps,
    size_t bltu_step_start,
    size_t num_bltu_steps,
    size_t bge_step_start,
    size_t num_bge_steps,
    size_t bgeu_step_start,
    size_t num_bgeu_steps,
    uint32_t *d_error,
    uint32_t blt_opcode,
    uint32_t bltu_opcode,
    uint32_t bge_opcode,
    uint32_t bgeu_opcode,
    uint32_t register_address_space,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(BranchLessThanCols<uint8_t>));
    assert(d_memory_log.len() == d_memory_predecessors.len());
    assert(blt_step_start <= d_steps.len());
    assert(num_blt_steps <= d_steps.len() - blt_step_start);
    assert(bltu_step_start <= d_steps.len());
    assert(num_bltu_steps <= d_steps.len() - bltu_step_start);
    assert(bge_step_start <= d_steps.len());
    assert(num_bge_steps <= d_steps.len() - bge_step_start);
    assert(bgeu_step_start <= d_steps.len());
    assert(num_bgeu_steps <= d_steps.len() - bgeu_step_start);
    assert(num_blt_steps <= SIZE_MAX - num_bltu_steps);
    size_t total_steps = num_blt_steps + num_bltu_steps;
    assert(total_steps <= SIZE_MAX - num_bge_steps);
    total_steps += num_bge_steps;
    assert(total_steps <= SIZE_MAX - num_bgeu_steps);
    total_steps += num_bgeu_steps;
    assert(height >= total_steps);
    auto [grid, block] = kernel_launch_params(height, 512);
    blt_replay_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_instructions,
        pc_base,
        d_program_log,
        d_memory_log,
        d_initial_write_log,
        d_memory_predecessors,
        d_steps,
        blt_step_start,
        num_blt_steps,
        bltu_step_start,
        num_bltu_steps,
        bge_step_start,
        num_bge_steps,
        bgeu_step_start,
        num_bgeu_steps,
        d_error,
        blt_opcode,
        bltu_opcode,
        bge_opcode,
        bgeu_opcode,
        register_address_space,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
