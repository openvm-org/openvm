#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_reg.cuh"
#include "riscv/cores/bitwise_logic.cuh"
#include "riscv/replay.cuh"

using namespace riscv;

// Concrete type aliases for RV64
using Rv64BitwiseLogicCoreRecord = BitwiseLogicCoreRecord<RV64_REGISTER_NUM_LIMBS>;
using Rv64BitwiseLogicCore = BitwiseLogicCore<RV64_REGISTER_NUM_LIMBS>;
template <typename T> using Rv64BitwiseLogicCoreCols = BitwiseLogicCoreCols<T, RV64_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv64BitwiseLogicCols {
    Rv64BaseAluRegAdapterCols<T> adapter;
    Rv64BitwiseLogicCoreCols<T> core;
};

struct Rv64BitwiseLogicRecord {
    Rv64BaseAluRegAdapterRecord adapter;
    Rv64BitwiseLogicCoreRecord core;
};

__global__ void bitwise_logic_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64BitwiseLogicRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Rv64BaseAluRegAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        Rv64BitwiseLogicCore core{BitwiseOperationLookup(d_bitwise_lookup_ptr)};
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64BitwiseLogicCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64BitwiseLogicCols<uint8_t>));
    }
}

extern "C" int _bitwise_logic_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64BitwiseLogicRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64BitwiseLogicCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    bitwise_logic_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

__global__ void bitwise_logic_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t xor_step_start,
    size_t num_xor_steps,
    size_t or_step_start,
    size_t num_or_steps,
    size_t and_step_start,
    size_t num_and_steps,
    uint32_t *error,
    uint32_t xor_opcode,
    uint32_t or_opcode,
    uint32_t and_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(Rv64BitwiseLogicCols<uint8_t>));
    size_t total_steps = num_xor_steps + num_or_steps + num_and_steps;
    if (idx >= total_steps) return;

    size_t step_index;
    uint32_t expected_opcode;
    uint8_t local_opcode;
    if (idx < num_xor_steps) {
        step_index = xor_step_start + idx;
        expected_opcode = xor_opcode;
        local_opcode = 2;
    } else if (idx < num_xor_steps + num_or_steps) {
        size_t group_index = idx - num_xor_steps;
        step_index = or_step_start + group_index;
        expected_opcode = or_opcode;
        local_opcode = 3;
    } else {
        size_t group_index = idx - num_xor_steps - num_or_steps;
        step_index = and_step_start + group_index;
        expected_opcode = and_opcode;
        local_opcode = 4;
    }
    auto const &step = steps[step_index];
    size_t program_index = step.program_index;
    if (program_index + 1 >= program_log.len()) {
        preflight_set_error(error, 131);
        return;
    }
    auto const &from = program_log[program_index];
    auto const &to = program_log[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % 4 != 0 ||
        from.timestamp + 3 != to.timestamp || to.pc != from.pc + 4) {
        preflight_set_error(error, 132);
        return;
    }
    size_t instruction_index = (from.pc - pc_base) / 4;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 133);
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
        preflight_set_error(error, 134);
        return;
    }

    size_t rs1_index = step.memory_start;
    size_t rs2_index = rs1_index + 1;
    size_t write_index = rs1_index + 2;
    if (write_index >= memory.len() || write_index >= predecessors.len()) {
        preflight_set_error(error, 135);
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
        preflight_set_error(error, 136);
        return;
    }

    uint16_t rs1_cells[BLOCK_FE_WIDTH];
    uint16_t rs2_cells[BLOCK_FE_WIDTH];
    uint16_t logged_result_cells[BLOCK_FE_WIDTH];
    if (!replay_u16_block(rs1.value, rs1_cells) || !replay_u16_block(rs2.value, rs2_cells) ||
        !replay_u16_block(write.value, logged_result_cells)) {
        preflight_set_error(error, 137);
        return;
    }
    uint8_t b[RV64_REGISTER_NUM_LIMBS];
    uint8_t c[RV64_REGISTER_NUM_LIMBS];
    uint8_t logged_result[RV64_REGISTER_NUM_LIMBS];
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        b[2 * i] = static_cast<uint8_t>(rs1_cells[i]);
        b[2 * i + 1] = static_cast<uint8_t>(rs1_cells[i] >> 8);
        c[2 * i] = static_cast<uint8_t>(rs2_cells[i]);
        c[2 * i + 1] = static_cast<uint8_t>(rs2_cells[i] >> 8);
        logged_result[2 * i] = static_cast<uint8_t>(logged_result_cells[i]);
        logged_result[2 * i + 1] = static_cast<uint8_t>(logged_result_cells[i] >> 8);
    }
    uint8_t expected_result[RV64_REGISTER_NUM_LIMBS];
    if (local_opcode == 2) {
        run_xor<RV64_REGISTER_NUM_LIMBS>(b, c, expected_result);
    } else if (local_opcode == 3) {
        run_or<RV64_REGISTER_NUM_LIMBS>(b, c, expected_result);
    } else {
        run_and<RV64_REGISTER_NUM_LIMBS>(b, c, expected_result);
    }
#pragma unroll
    for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
        if (logged_result[i] != expected_result[i]) {
            preflight_set_error(error, 138);
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
        preflight_set_error(error, 139);
        return;
    }

    auto adapter = Rv64BaseAluRegAdapter(
        VariableRangeChecker(range_checker, range_checker_num_bins), timestamp_max_bits
    );
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
    auto core = Rv64BitwiseLogicCore(BitwiseOperationLookup(bitwise_lookup));
    core.fill_trace_row(
        row.slice_from(COL_INDEX(Rv64BitwiseLogicCols, core)), b, c, local_opcode
    );
}

extern "C" int _bitwise_logic_replay_tracegen(
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
    size_t xor_step_start,
    size_t num_xor_steps,
    size_t or_step_start,
    size_t num_or_steps,
    size_t and_step_start,
    size_t num_and_steps,
    uint32_t *error,
    uint32_t xor_opcode,
    uint32_t or_opcode,
    uint32_t and_opcode,
    uint32_t register_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64BitwiseLogicCols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(xor_step_start <= steps.len());
    assert(num_xor_steps <= steps.len() - xor_step_start);
    assert(or_step_start <= steps.len());
    assert(num_or_steps <= steps.len() - or_step_start);
    assert(and_step_start <= steps.len());
    assert(num_and_steps <= steps.len() - and_step_start);
    assert(num_xor_steps <= SIZE_MAX - num_or_steps);
    assert(num_xor_steps + num_or_steps <= SIZE_MAX - num_and_steps);
    assert(height >= num_xor_steps + num_or_steps + num_and_steps);
    auto [grid, block] = kernel_launch_params(height, 512);
    bitwise_logic_replay_tracegen<<<grid, block, 0, stream>>>(
        trace,
        height,
        instructions,
        pc_base,
        program_log,
        memory,
        seeds,
        predecessors,
        steps,
        xor_step_start,
        num_xor_steps,
        or_step_start,
        num_or_steps,
        and_step_start,
        num_and_steps,
        error,
        xor_opcode,
        or_opcode,
        and_opcode,
        register_address_space,
        range_checker,
        range_checker_num_bins,
        bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
