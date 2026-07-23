#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm.cuh"
#include "riscv/cores/bitwise_logic_imm.cuh"
#include "riscv/replay.cuh"

using namespace riscv;

// XORI/ORI/ANDI use byte limbs and the immediate-only byte ALU adapter.
using Rv64BitwiseLogicImmCoreRecord = BitwiseLogicImmCoreRecord<RV64_REGISTER_NUM_LIMBS>;
using Rv64BitwiseLogicImmCore = BitwiseLogicImmCore<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>;
template <typename T>
using Rv64BitwiseLogicImmCoreCols = BitwiseLogicImmCoreCols<T, RV64_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv64BitwiseLogicImmCols {
    Rv64BaseAluImmAdapterCols<T> adapter;
    Rv64BitwiseLogicImmCoreCols<T> core;
};

struct Rv64BitwiseLogicImmRecord {
    Rv64BaseAluImmAdapterRecord adapter;
    Rv64BitwiseLogicImmCoreRecord core;
};

static_assert(sizeof(Rv64BitwiseLogicImmRecord) == 44);
static_assert(offsetof(Rv64BitwiseLogicImmRecord, core) == 32);

__global__ void bitwise_logic_imm_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64BitwiseLogicImmRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        auto adapter = Rv64BaseAluImmAdapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        auto core = Rv64BitwiseLogicImmCore(BitwiseOperationLookup(d_bitwise_lookup_ptr));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64BitwiseLogicImmCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64BitwiseLogicImmCols<uint8_t>));
    }
}

extern "C" int _bitwise_logic_imm_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64BitwiseLogicImmRecord> d_records,
    uint32_t *d_range_checker,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64BitwiseLogicImmCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    bitwise_logic_imm_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker,
        range_checker_bins,
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

__global__ void bitwise_logic_imm_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program_log,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t xori_step_start,
    size_t num_xori_steps,
    size_t ori_step_start,
    size_t num_ori_steps,
    size_t andi_step_start,
    size_t num_andi_steps,
    uint32_t *error,
    uint32_t xori_opcode,
    uint32_t ori_opcode,
    uint32_t andi_opcode,
    uint32_t register_address_space,
    uint32_t immediate_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(Rv64BitwiseLogicImmCols<uint8_t>));
    size_t total_steps = num_xori_steps + num_ori_steps + num_andi_steps;
    if (idx >= total_steps) return;

    size_t step_index;
    uint32_t expected_opcode;
    uint8_t local_opcode;
    if (idx < num_xori_steps) {
        step_index = xori_step_start + idx;
        expected_opcode = xori_opcode;
        local_opcode = 1;
    } else if (idx < num_xori_steps + num_ori_steps) {
        size_t group_index = idx - num_xori_steps;
        step_index = ori_step_start + group_index;
        expected_opcode = ori_opcode;
        local_opcode = 2;
    } else {
        size_t group_index = idx - num_xori_steps - num_ori_steps;
        step_index = andi_step_start + group_index;
        expected_opcode = andi_opcode;
        local_opcode = 3;
    }
    auto const &step = steps[step_index];
    size_t program_index = step.program_index;
    if (program_index + 1 >= program_log.len()) {
        preflight_set_error(error, 91);
        return;
    }
    auto const &from = program_log[program_index];
    auto const &to = program_log[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % 4 != 0 ||
        from.timestamp + 2 != to.timestamp || to.pc != from.pc + 4) {
        preflight_set_error(error, 92);
        return;
    }
    size_t instruction_index = (from.pc - pc_base) / 4;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 93);
        return;
    }
    auto const &instruction = instructions[instruction_index];
    uint32_t rd_ptr = instruction.words[1];
    uint32_t rs1_ptr = instruction.words[2];
    uint32_t immediate = instruction.words[3];
    uint32_t imm_sign = (immediate >> 11) & 1;
    if (instruction.words[0] != expected_opcode ||
        instruction.words[4] != register_address_space ||
        instruction.words[5] != immediate_address_space || rd_ptr == 0 || (rd_ptr & 1) != 0 ||
        (rs1_ptr & 1) != 0 || immediate >= (1u << 24) ||
        (immediate >> 12) != (imm_sign ? 0xfffu : 0)) {
        preflight_set_error(error, 94);
        return;
    }

    size_t read_index = step.memory_start;
    size_t write_index = read_index + 1;
    if (write_index >= memory.len() || write_index >= predecessors.len()) {
        preflight_set_error(error, 95);
        return;
    }
    auto const &read = memory[read_index];
    auto const &write = memory[write_index];
    if (read.timestamp != from.timestamp || preflight_is_write(read) ||
        preflight_address_space(read) != register_address_space || read.pointer != rs1_ptr / 2 ||
        write.timestamp != from.timestamp + 1 || !preflight_is_write(write) ||
        preflight_address_space(write) != register_address_space || write.pointer != rd_ptr / 2 ||
        (write_index + 1 < memory.len() && memory[write_index + 1].timestamp < to.timestamp)) {
        preflight_set_error(error, 96);
        return;
    }

    uint16_t source_cells[BLOCK_FE_WIDTH];
    uint16_t logged_result_cells[BLOCK_FE_WIDTH];
    if (!replay_u16_block(read.value, source_cells) ||
        !replay_u16_block(write.value, logged_result_cells)) {
        preflight_set_error(error, 97);
        return;
    }
    uint8_t source[RV64_REGISTER_NUM_LIMBS];
    uint8_t logged_result[RV64_REGISTER_NUM_LIMBS];
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        source[2 * i] = static_cast<uint8_t>(source_cells[i]);
        source[2 * i + 1] = static_cast<uint8_t>(source_cells[i] >> 8);
        logged_result[2 * i] = static_cast<uint8_t>(logged_result_cells[i]);
        logged_result[2 * i + 1] = static_cast<uint8_t>(logged_result_cells[i] >> 8);
    }
    uint8_t c_low[2] = {
        static_cast<uint8_t>(immediate), static_cast<uint8_t>((immediate >> 8) & 0x07)
    };
    uint8_t c[RV64_REGISTER_NUM_LIMBS];
    c[0] = c_low[0];
    c[1] = c_low[1] + static_cast<uint8_t>(imm_sign) * 0xf8;
#pragma unroll
    for (size_t i = 2; i < RV64_REGISTER_NUM_LIMBS; i++) {
        c[i] = imm_sign ? 0xff : 0;
    }
    uint8_t expected_result[RV64_REGISTER_NUM_LIMBS];
    if (local_opcode == 1) {
        run_xor<RV64_REGISTER_NUM_LIMBS>(source, c, expected_result);
    } else if (local_opcode == 2) {
        run_or<RV64_REGISTER_NUM_LIMBS>(source, c, expected_result);
    } else {
        run_and<RV64_REGISTER_NUM_LIMBS>(source, c, expected_result);
    }
#pragma unroll
    for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
        if (logged_result[i] != expected_result[i]) {
            preflight_set_error(error, 98);
            return;
        }
    }

    ReplayPreviousValue read_previous;
    ReplayPreviousValue write_previous;
    if (!replay_previous_value(
            read_index, read, predecessors[read_index], memory, seeds, read_previous
        ) ||
        !replay_previous_value(
            write_index, write, predecessors[write_index], memory, seeds, write_previous
        )) {
        preflight_set_error(error, 99);
        return;
    }

    auto adapter = Rv64BaseAluImmAdapter(
        VariableRangeChecker(range_checker, range_checker_num_bins), timestamp_max_bits
    );
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
    auto core = Rv64BitwiseLogicImmCore(BitwiseOperationLookup(bitwise_lookup));
    core.fill_trace_row(
        row.slice_from(COL_INDEX(Rv64BitwiseLogicImmCols, core)),
        source,
        c_low,
        static_cast<uint8_t>(imm_sign),
        local_opcode
    );
}

extern "C" int _bitwise_logic_imm_replay_tracegen(
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
    size_t xori_step_start,
    size_t num_xori_steps,
    size_t ori_step_start,
    size_t num_ori_steps,
    size_t andi_step_start,
    size_t num_andi_steps,
    uint32_t *error,
    uint32_t xori_opcode,
    uint32_t ori_opcode,
    uint32_t andi_opcode,
    uint32_t register_address_space,
    uint32_t immediate_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64BitwiseLogicImmCols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(xori_step_start <= steps.len());
    assert(num_xori_steps <= steps.len() - xori_step_start);
    assert(ori_step_start <= steps.len());
    assert(num_ori_steps <= steps.len() - ori_step_start);
    assert(andi_step_start <= steps.len());
    assert(num_andi_steps <= steps.len() - andi_step_start);
    assert(num_xori_steps <= SIZE_MAX - num_ori_steps);
    assert(num_xori_steps + num_ori_steps <= SIZE_MAX - num_andi_steps);
    assert(height >= num_xori_steps + num_ori_steps + num_andi_steps);
    auto [grid, block] = kernel_launch_params(height, 512);
    bitwise_logic_imm_replay_tracegen<<<grid, block, 0, stream>>>(
        trace,
        height,
        instructions,
        pc_base,
        program_log,
        memory,
        seeds,
        predecessors,
        steps,
        xori_step_start,
        num_xori_steps,
        ori_step_start,
        num_ori_steps,
        andi_step_start,
        num_andi_steps,
        error,
        xori_opcode,
        ori_opcode,
        andi_opcode,
        register_address_space,
        immediate_address_space,
        range_checker,
        range_checker_num_bins,
        bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
