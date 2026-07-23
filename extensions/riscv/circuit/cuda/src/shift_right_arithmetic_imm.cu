#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm_u16.cuh"
#include "riscv/cores/shift_right_arithmetic_imm.cuh"
#include "riscv/replay.cuh"
#include "system/memory/params.cuh"

using namespace riscv;
using namespace program;

// SRAI uses u16 limbs (4 limbs of 16 bits) and the immediate-only u16 ALU adapter.
using Rv64ShiftRightArithmeticImmCoreRecord =
    ShiftRightArithmeticImmCoreRecord<BLOCK_FE_WIDTH, U16_BITS>;
using Rv64ShiftRightArithmeticImmCore =
    ShiftRightArithmeticImmCore<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64ShiftRightArithmeticImmCoreCols =
    ShiftRightArithmeticImmCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct ShiftRightArithmeticImmCols {
    Rv64BaseAluImmU16AdapterCols<T> adapter;
    Rv64ShiftRightArithmeticImmCoreCols<T> core;
};

struct ShiftRightArithmeticImmRecord {
    Rv64BaseAluImmU16AdapterRecord adapter;
    Rv64ShiftRightArithmeticImmCoreRecord core;
};

static_assert(sizeof(ShiftRightArithmeticImmRecord) == 44);
static_assert(offsetof(ShiftRightArithmeticImmRecord, core) == 32);

__global__ void shift_right_arithmetic_imm_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftRightArithmeticImmRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluImmU16Adapter(
            VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Rv64ShiftRightArithmeticImmCore(
            VariableRangeChecker(range_ptr, range_bins)
        );
        core.fill_trace_row(
            row.slice_from(COL_INDEX(ShiftRightArithmeticImmCols, core)), rec.core
        );
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _shift_right_arithmetic_imm_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftRightArithmeticImmRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(ShiftRightArithmeticImmCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    shift_right_arithmetic_imm_tracegen<<<grid, block, 0, stream>>>(
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

__global__ void shift_right_arithmetic_imm_replay_tracegen(
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
    uint32_t immediate_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(ShiftRightArithmeticImmCols<uint8_t>));
    if (idx >= num_steps) return;

    size_t step_index = step_start + idx;
    auto const &step = steps[step_index];
    size_t program_index = step.program_index;
    if (program_index + 1 >= program_log.len()) {
        preflight_set_error(error, 61);
        return;
    }
    auto const &from = program_log[program_index];
    auto const &to = program_log[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % 4 != 0 ||
        from.timestamp + 2 != to.timestamp || to.pc != from.pc + 4) {
        preflight_set_error(error, 62);
        return;
    }
    size_t instruction_index = (from.pc - pc_base) / 4;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 63);
        return;
    }
    auto const &instruction = instructions[instruction_index];
    uint32_t rd_ptr = instruction.words[1];
    uint32_t rs1_ptr = instruction.words[2];
    uint32_t shamt = instruction.words[3];
    if (instruction.words[0] != opcode ||
        instruction.words[4] != register_address_space ||
        instruction.words[5] != immediate_address_space || rd_ptr == 0 || (rd_ptr & 1) != 0 ||
        (rs1_ptr & 1) != 0 || shamt >= BLOCK_FE_WIDTH * U16_BITS) {
        preflight_set_error(error, 64);
        return;
    }

    size_t read_index = step.memory_start;
    size_t write_index = read_index + 1;
    if (write_index >= memory.len() || write_index >= predecessors.len()) {
        preflight_set_error(error, 65);
        return;
    }
    auto const &read = memory[read_index];
    auto const &write = memory[write_index];
    if (read.timestamp != from.timestamp || preflight_is_write(read) ||
        preflight_address_space(read) != register_address_space || read.pointer != rs1_ptr / 2 ||
        write.timestamp != from.timestamp + 1 || !preflight_is_write(write) ||
        preflight_address_space(write) != register_address_space || write.pointer != rd_ptr / 2 ||
        (write_index + 1 < memory.len() && memory[write_index + 1].timestamp < to.timestamp)) {
        preflight_set_error(error, 66);
        return;
    }

    uint16_t source[BLOCK_FE_WIDTH];
    uint16_t logged_result[BLOCK_FE_WIDTH];
    if (!replay_u16_block(read.value, source) ||
        !replay_u16_block(write.value, logged_result)) {
        preflight_set_error(error, 67);
        return;
    }
    uint16_t shamt_limbs[BLOCK_FE_WIDTH] = {0};
    shamt_limbs[0] = static_cast<uint16_t>(shamt);
    uint16_t expected_result[BLOCK_FE_WIDTH];
    size_t limb_shift = 0;
    size_t bit_shift = 0;
    run_shift_right_arithmetic<BLOCK_FE_WIDTH, U16_BITS>(
        source, shamt_limbs, expected_result, limb_shift, bit_shift
    );
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        if (logged_result[i] != expected_result[i]) {
            preflight_set_error(error, 68);
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
        preflight_set_error(error, 69);
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
    auto core = Rv64ShiftRightArithmeticImmCore(checker);
    core.fill_trace_row(
        row.slice_from(COL_INDEX(ShiftRightArithmeticImmCols, core)),
        source,
        static_cast<uint8_t>(shamt)
    );
}

extern "C" int _shift_right_arithmetic_imm_replay_tracegen(
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
    uint32_t immediate_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(ShiftRightArithmeticImmCols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(step_start <= steps.len());
    assert(num_steps <= steps.len() - step_start);
    assert(height >= num_steps);
    auto [grid, block] = kernel_launch_params(height, 512);
    shift_right_arithmetic_imm_replay_tracegen<<<grid, block, 0, stream>>>(
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
        immediate_address_space,
        range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
