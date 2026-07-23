#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/rdwrite.cuh"
#include "riscv/replay.cuh"

using namespace riscv;
using namespace program;

constexpr uint32_t LUI_IMM_LOW_BITS = U16_BITS - RV_IS_TYPE_IMM_BITS;
constexpr uint32_t LUI_IMM_BITS = 32 - RV_IS_TYPE_IMM_BITS;
constexpr uint32_t PC_HIGH_U16_SHIFT = 2 * U16_BITS - PC_BITS;

template <typename T> struct Rv64JalLuiCoreCols {
    T imm;                             // core_row.imm
    T rd_data[RV64_PTR_U16_LIMBS];     // low-32 bits of rd_data as u16 cells
    T imm_low_4;                       // low 4 bits of imm for LUI
    T is_jal;                          // core_row.is_jal
    T is_lui;                          // core_row.is_lui
    T is_sign_extend;                  // 1 if upper cells are 0xFFFF, 0 if 0x0000
};

struct Rv64JalLuiCoreRecord {
    uint32_t imm;
    uint16_t rd_data[BLOCK_FE_WIDTH];
    bool is_jal;
};

struct Rv64JalLuiCore {
    VariableRangeChecker range_checker;

    __device__ Rv64JalLuiCore(VariableRangeChecker rc) : range_checker(rc) {}

    __device__ void fill_trace_row(
        RowSlice row, uint32_t imm, const uint16_t rd_data[BLOCK_FE_WIDTH], bool is_jal
    ) {
        uint32_t rd_lo = rd_data[0];
        uint32_t rd_hi = rd_data[1];

        bool is_sign_extend = (rd_hi >> (U16_BITS - 1)) & 1;
        uint32_t imm_low_4 = is_jal ? 0u : (imm & 0xfu);

        range_checker.add_count(rd_lo, U16_BITS);
        range_checker.add_count(rd_hi, U16_BITS);
        range_checker.add_count(
            2u * rd_hi - ((uint32_t)is_sign_extend << U16_BITS), U16_BITS
        );

        if (!is_jal) {
            range_checker.add_count(imm_low_4, LUI_IMM_LOW_BITS);
        } else {
            range_checker.add_count(rd_hi << PC_HIGH_U16_SHIFT, U16_BITS);
        }

        uint32_t rd_u16[2] = {rd_lo, rd_hi};
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, is_sign_extend, is_sign_extend);
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, is_lui, !is_jal);
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, is_jal, is_jal);
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, imm_low_4, imm_low_4);
        COL_WRITE_ARRAY(row, Rv64JalLuiCoreCols, rd_data, rd_u16);
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, imm, imm);
    }

    __device__ void fill_trace_row(RowSlice row, Rv64JalLuiCoreRecord record) {
        fill_trace_row(row, record.imm, record.rd_data, record.is_jal);
    }
};

template <typename T> struct Rv64JalLuiCols {
    Rv64CondRdWriteAdapterCols<T> adapter;
    Rv64JalLuiCoreCols<T> core;
};

struct Rv64JalLuiRecord {
    Rv64RdWriteAdapterRecord adapter;
    Rv64JalLuiCoreRecord core;
};

__global__ void jal_lui_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<Rv64JalLuiRecord> records,
    uint32_t *rc_ptr,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < records.len()) {
        auto const &full = records[idx];

        Rv64CondRdWriteAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, full.adapter);
        Rv64JalLuiCore core(VariableRangeChecker(rc_ptr, rc_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64JalLuiCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(Rv64JalLuiCols<uint8_t>));
    }
}

__global__ void jal_lui_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t jal_step_start,
    size_t num_jal_steps,
    size_t lui_step_start,
    size_t num_lui_steps,
    uint32_t *error,
    uint32_t jal_opcode,
    uint32_t lui_opcode,
    uint32_t register_as,
    uint32_t *rc_ptr,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(Rv64JalLuiCols<uint8_t>));

    size_t total_steps = num_jal_steps + num_lui_steps;
    if (idx >= total_steps) return;

    bool is_jal = idx < num_jal_steps;
    size_t group_index = is_jal ? idx : idx - num_jal_steps;
    size_t step_index = (is_jal ? jal_step_start : lui_step_start) + group_index;
    auto const &step = steps[step_index];
    size_t program_index = step.program_index;
    if (program_index + 1 >= program.len()) {
        preflight_set_error(error, 181);
        return;
    }

    auto const &from = program[program_index];
    auto const &to = program[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % 4 != 0 ||
        to.timestamp != from.timestamp + 1) {
        preflight_set_error(error, 182);
        return;
    }
    size_t instruction_index = (from.pc - pc_base) / 4;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 183);
        return;
    }

    auto const &instruction = instructions[instruction_index];
    uint32_t expected_opcode = is_jal ? jal_opcode : lui_opcode;
    uint32_t rd_ptr = instruction.words[1];
    uint32_t encoded_imm = instruction.words[3];
    uint32_t needs_write = instruction.words[6];
    if (instruction.words[0] != expected_opcode || instruction.words[2] != 0 ||
        instruction.words[4] != register_as || instruction.words[5] != 0 ||
        needs_write > 1 || (rd_ptr & 1) != 0 ||
        needs_write != (rd_ptr != 0)) {
        preflight_set_error(error, 184);
        return;
    }

    uint32_t rd_low;
    uint32_t expected_pc;
    if (is_jal) {
        constexpr uint32_t MAX_PC = (1u << PC_BITS) - 1;
        if (from.pc > MAX_PC - 4) {
            preflight_set_error(error, 189);
            return;
        }
        rd_low = from.pc + 4;
        Fp target(from.pc);
        target += Fp(encoded_imm);
        expected_pc = target.asUInt32();
    } else {
        if (encoded_imm >= (1u << LUI_IMM_BITS)) {
            preflight_set_error(error, 189);
            return;
        }
        rd_low = encoded_imm << 12;
        expected_pc = from.pc + 4;
    }
    if (to.pc != expected_pc) {
        preflight_set_error(error, 187);
        return;
    }

    uint16_t sign = !is_jal && (rd_low >> 31) ? UINT16_MAX : 0;
    uint16_t expected_data[BLOCK_FE_WIDTH] = {
        static_cast<uint16_t>(rd_low),
        static_cast<uint16_t>(rd_low >> U16_BITS),
        sign,
        sign,
    };
    ReplayPreviousValue previous = {};
    if (needs_write) {
        size_t memory_idx = step.memory_start;
        if (memory_idx >= memory.len() || memory_idx >= predecessors.len()) {
            preflight_set_error(error, 185);
            return;
        }
        auto const &event = memory[memory_idx];
        uint16_t logged_data[BLOCK_FE_WIDTH];
        if (event.timestamp != from.timestamp || !preflight_is_write(event) ||
            preflight_address_space(event) != register_as || event.pointer != rd_ptr / 2 ||
            (memory_idx + 1 < memory.len() && memory[memory_idx + 1].timestamp < to.timestamp)) {
            preflight_set_error(error, 185);
            return;
        }
        if (!replay_u16_block(event.value, logged_data)) {
            preflight_set_error(error, 186);
            return;
        }
        bool matches = true;
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            matches &= logged_data[i] == expected_data[i];
        }
        if (!matches) {
            preflight_set_error(error, 186);
            return;
        }
        if (!replay_previous_value(
                memory_idx, event, predecessors[memory_idx], memory, seeds, previous
            )) {
            preflight_set_error(error, 188);
            return;
        }
    } else if (
        step.memory_start < memory.len() &&
        memory[step.memory_start].timestamp < to.timestamp
    ) {
        preflight_set_error(error, 185);
        return;
    }

    Rv64CondRdWriteAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins), timestamp_max_bits);
    adapter.fill_trace_row(
        row,
        from.pc,
        from.timestamp,
        rd_ptr,
        needs_write,
        previous.timestamp,
        previous.value
    );
    Rv64JalLuiCore core(VariableRangeChecker(rc_ptr, rc_bins));
    core.fill_trace_row(
        row.slice_from(COL_INDEX(Rv64JalLuiCols, core)),
        encoded_imm,
        expected_data,
        is_jal
    );
}

extern "C" int _jal_lui_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64JalLuiRecord> d_records,
    uint32_t *d_rc,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64JalLuiCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height, 512);

    jal_lui_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_rc, rc_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

extern "C" int _jal_lui_replay_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrReplayInstruction> d_instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> d_program,
    DeviceBufferConstView<PreflightMemoryEvent> d_memory,
    DeviceBufferConstView<PreflightInitialWrite> d_seeds,
    DeviceBufferConstView<uint32_t> d_predecessors,
    DeviceBufferConstView<RvrReplayStep> d_steps,
    size_t jal_step_start,
    size_t num_jal_steps,
    size_t lui_step_start,
    size_t num_lui_steps,
    uint32_t *d_error,
    uint32_t jal_opcode,
    uint32_t lui_opcode,
    uint32_t register_as,
    uint32_t *d_rc,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64JalLuiCols<uint8_t>));
    assert(d_memory.len() == d_predecessors.len());
    assert(jal_step_start <= d_steps.len());
    assert(num_jal_steps <= d_steps.len() - jal_step_start);
    assert(lui_step_start <= d_steps.len());
    assert(num_lui_steps <= d_steps.len() - lui_step_start);
    assert(num_jal_steps <= SIZE_MAX - num_lui_steps);
    assert(height >= num_jal_steps + num_lui_steps);

    auto [grid, block] = kernel_launch_params(height, 512);
    jal_lui_replay_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_instructions,
        pc_base,
        d_program,
        d_memory,
        d_seeds,
        d_predecessors,
        d_steps,
        jal_step_start,
        num_jal_steps,
        lui_step_start,
        num_lui_steps,
        d_error,
        jal_opcode,
        lui_opcode,
        register_as,
        d_rc,
        rc_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
