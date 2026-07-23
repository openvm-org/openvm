#include <assert.h>

#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv/adapters/rdwrite.cuh"
#include "riscv/replay.cuh"

using namespace riscv;
using namespace program;

template <typename T> struct Rv64AuipcCoreCols {
    T is_valid;
    T is_sign_extend;
    // The immediate is split around the byte shift in AUIPC's `imm << 8`.
    T imm_low_8;
    T imm_high_16;
    T pc_high;
    T rd_data[RV64_PTR_U16_LIMBS];
};

struct Rv64AuipcCoreRecord {
    uint32_t from_pc;
    uint32_t imm;
};

__device__ uint64_t run_auipc(uint32_t pc, uint32_t imm) {
    uint32_t offset = imm << RV64_BYTE_BITS;
    int64_t signed_offset = (int64_t)(int32_t)offset;
    return (uint64_t)pc + (uint64_t)signed_offset;
}

struct Rv64AuipcCore {
    VariableRangeChecker range_checker;

    __device__ Rv64AuipcCore(VariableRangeChecker range_checker) : range_checker(range_checker) {}

    __device__ void fill_trace_row(RowSlice row, uint32_t from_pc, uint32_t imm) {
        uint32_t imm_low_8 = imm & ((1u << RV64_BYTE_BITS) - 1u);
        uint32_t imm_high_16 = (imm >> RV64_BYTE_BITS) & uint32_t(UINT16_MAX);
        uint16_t pc_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(pc_limbs, from_pc);
        uint64_t auipc = run_auipc(from_pc, imm);
        uint64_t auipc_hi = auipc >> 32;
        assert(auipc_hi == 0ull || auipc_hi == 0xffffffffull);
        uint32_t auipc_lo = (uint32_t)auipc;
        uint16_t rd_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rd_limbs, auipc_lo);
        uint32_t rd_lo = rd_limbs[0];
        uint32_t rd_hi = rd_limbs[1];
        uint32_t is_sign_ext = (auipc_hi != 0) ? 1u : 0u;
        uint32_t imm_sign = (imm_high_16 >> (U16_BITS - 1)) & 1u;

        range_checker.add_count(pc_limbs[0], U16_BITS);
        range_checker.add_count(pc_limbs[1], PC_BITS - U16_BITS);
        range_checker.add_count(imm_low_8, RV64_BYTE_BITS);
        range_checker.add_count(imm_high_16, U16_BITS);
        range_checker.add_count(rd_lo, U16_BITS);
        range_checker.add_count(rd_hi, U16_BITS);
        // Check that imm_sign matches the top bit of imm_high_16.
        range_checker.add_count(2u * imm_high_16 - (imm_sign << U16_BITS), U16_BITS);

        uint32_t rd_u16[RV64_PTR_U16_LIMBS] = {rd_lo, rd_hi};
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, imm_low_8, imm_low_8);
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, imm_high_16, imm_high_16);
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, pc_high, pc_limbs[1]);
        COL_WRITE_ARRAY(row, Rv64AuipcCoreCols, rd_data, rd_u16);
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, is_sign_extend, is_sign_ext);
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, is_valid, 1);
    }

    __device__ void fill_trace_row(RowSlice row, Rv64AuipcCoreRecord record) {
        fill_trace_row(row, record.from_pc, record.imm);
    }
};

template <typename T> struct Rv64AuipcCols {
    Rv64RdWriteAdapterCols<T> adapter;
    Rv64AuipcCoreCols<T> core;
};

struct Rv64AuipcRecord {
    Rv64RdWriteAdapterRecord adapter;
    Rv64AuipcCoreRecord core;
};

__global__ void auipc_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<Rv64AuipcRecord> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = Rv64RdWriteAdapter(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core =
            Rv64AuipcCore(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64AuipcCols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(Rv64AuipcCols<uint8_t>));
    }
}

__global__ void auipc_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t *error,
    uint32_t auipc_opcode,
    uint32_t register_as,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(Rv64AuipcCols<uint8_t>));
    if (idx >= num_steps) return;

    auto const &step = steps[step_start + idx];
    size_t program_index = step.program_index;
    if (program_index + 1 >= program.len()) {
        preflight_set_error(error, 191);
        return;
    }

    auto const &from = program[program_index];
    auto const &to = program[program_index + 1];
    constexpr uint32_t MAX_PC = (1u << PC_BITS) - 1;
    if (from.pc < pc_base || (from.pc - pc_base) % 4 != 0 ||
        from.timestamp == UINT32_MAX || from.pc > MAX_PC - 4 ||
        to.timestamp != from.timestamp + 1 || to.pc != from.pc + 4) {
        preflight_set_error(error, 192);
        return;
    }

    size_t instruction_index = (from.pc - pc_base) / 4;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 193);
        return;
    }

    auto const &instruction = instructions[instruction_index];
    uint32_t rd_ptr = instruction.words[1];
    uint32_t imm = instruction.words[3];
    constexpr uint32_t REGISTER_FILE_BYTES = 32 * RV64_REGISTER_NUM_LIMBS;
    bool rd_is_canonical =
        rd_ptr != 0 && rd_ptr < REGISTER_FILE_BYTES && rd_ptr % RV64_REGISTER_NUM_LIMBS == 0;
    if (instruction.words[0] != auipc_opcode || !rd_is_canonical ||
        instruction.words[2] != 0 || imm >= (1u << 24) ||
        instruction.words[4] != register_as || instruction.words[5] != 0 ||
        instruction.words[6] != 0 || instruction.words[7] != 0) {
        preflight_set_error(error, 194);
        return;
    }

    size_t write_index = step.memory_start;
    if (write_index >= memory.len() || write_index >= predecessors.len()) {
        preflight_set_error(error, 195);
        return;
    }
    auto const &write = memory[write_index];
    if (write.timestamp != from.timestamp || !preflight_is_write(write) ||
        preflight_address_space(write) != register_as || write.pointer != rd_ptr / 2 ||
        (write_index + 1 < memory.len() &&
         memory[write_index + 1].timestamp < to.timestamp)) {
        preflight_set_error(error, 195);
        return;
    }

    uint16_t logged_data[BLOCK_FE_WIDTH];
    if (!replay_u16_block(write.value, logged_data)) {
        preflight_set_error(error, 196);
        return;
    }
    uint64_t expected_result = run_auipc(from.pc, imm);
    uint64_t expected_high = expected_result >> 32;
    if (expected_high != 0 && expected_high != UINT32_MAX) {
        preflight_set_error(error, 199);
        return;
    }
    uint16_t expected_data[BLOCK_FE_WIDTH] = {
        static_cast<uint16_t>(expected_result),
        static_cast<uint16_t>(expected_result >> U16_BITS),
        static_cast<uint16_t>(expected_result >> (2 * U16_BITS)),
        static_cast<uint16_t>(expected_result >> (3 * U16_BITS)),
    };
    bool matches = true;
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        matches &= logged_data[i] == expected_data[i];
    }
    if (!matches) {
        preflight_set_error(error, 197);
        return;
    }

    ReplayPreviousValue previous = {};
    if (!replay_previous_value(
            write_index, write, predecessors[write_index], memory, seeds, previous
        )) {
        preflight_set_error(error, 198);
        return;
    }

    Rv64RdWriteAdapter adapter(
        VariableRangeChecker(range_checker, range_checker_num_bins), timestamp_max_bits
    );
    adapter.fill_trace_row(
        row, from.pc, from.timestamp, rd_ptr, previous.timestamp, previous.value
    );
    Rv64AuipcCore core(VariableRangeChecker(range_checker, range_checker_num_bins));
    core.fill_trace_row(row.slice_from(COL_INDEX(Rv64AuipcCols, core)), from.pc, imm);
}

extern "C" int _auipc_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64AuipcRecord> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AuipcCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    auipc_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_range_checker, range_checker_num_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

extern "C" int _auipc_replay_tracegen(
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
    size_t step_start,
    size_t num_steps,
    uint32_t *d_error,
    uint32_t auipc_opcode,
    uint32_t register_as,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AuipcCols<uint8_t>));
    assert(d_memory.len() == d_predecessors.len());
    assert(step_start <= d_steps.len());
    assert(num_steps <= d_steps.len() - step_start);
    assert(height >= num_steps);

    auto [grid, block] = kernel_launch_params(height, 512);
    auipc_replay_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_instructions,
        pc_base,
        d_program,
        d_memory,
        d_seeds,
        d_predecessors,
        d_steps,
        step_start,
        num_steps,
        d_error,
        auipc_opcode,
        register_as,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
