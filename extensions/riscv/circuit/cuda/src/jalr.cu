#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv/adapters/jalr.cuh"
#include "riscv/replay.cuh"

using namespace riscv;
using namespace program;

template <typename T> struct Rv64JalrCoreCols {
    T imm;                                  // 2 bytes
    T rs1_data[RV64_PTR_U16_LIMBS];         // low 32 bits of rs1 as u16 cells
    T rd_high[RV64_PTR_U16_LIMBS - 1];      // high u16 limb of low-32 rd
    T is_valid;                             // 1 byte
    T to_pc_least_sig_bit;                  // 1 byte
    T to_pc_limbs[RV64_PTR_U16_LIMBS];      // `to_pc * 2` after the low-bit split
    T imm_sign;                             // 1 byte
};

struct Rv64JalrCoreRecord {
    uint16_t imm;
    uint32_t from_pc;
    uint32_t rs1_val;
    uint8_t imm_sign; // 0 or 1
};

__device__ void run_jalr(
    uint32_t pc,
    uint32_t rs1,
    uint16_t imm,
    bool imm_sign,
    uint32_t &out_pc,
    uint16_t rd_data[BLOCK_FE_WIDTH]
) {
    uint32_t offset = imm + (imm_sign ? (uint32_t(UINT16_MAX) << U16_BITS) : 0);
    int64_t signed_offset = (int64_t)(int32_t)offset;
    uint64_t to_pc = uint64_t(rs1) + signed_offset;

    assert(to_pc < (uint64_t(1) << PC_BITS));
    out_pc = uint32_t(to_pc);
    uint32_t rd_val = pc + DEFAULT_PC_STEP;
    rd_data[0] = uint16_t(rd_val);
    rd_data[1] = uint16_t(rd_val >> U16_BITS);
#pragma unroll
    for (size_t i = RV64_PTR_U16_LIMBS; i < BLOCK_FE_WIDTH; i++) {
        rd_data[i] = 0;
    }
}

struct Rv64JalrCore {
    VariableRangeChecker rc;

    __device__ Rv64JalrCore(VariableRangeChecker rc) : rc(rc) {}

    __device__ void fill_trace_row(
        RowSlice row, uint32_t from_pc, uint32_t rs1_val, uint16_t imm, bool imm_sign
    ) {
        uint32_t to_pc;
        uint16_t rd_data[BLOCK_FE_WIDTH];
        run_jalr(from_pc, rs1_val, imm, imm_sign, to_pc, rd_data);

        uint16_t to_pc_u16[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(to_pc_u16, to_pc);
        uint32_t to_pc_limbs[2] = {uint32_t(to_pc_u16[0] >> 1), uint32_t(to_pc_u16[1])};
        // to_pc_limbs[0] is 15 bits because it is doubled to reconstruct
        // the aligned JALR target.
        rc.add_count(to_pc_limbs[0], U16_BITS - 1);
        rc.add_count(to_pc_limbs[1], PC_BITS - U16_BITS);

        uint32_t rd_low_u16_lo = rd_data[0];
        uint32_t rd_low_u16_hi = rd_data[1];

        // rd writes the low 32 bits of from_pc + DEFAULT_PC_STEP. The high
        // limb is narrowed to the remaining PC bits because from_pc is program-bus bounded.
        rc.add_count(rd_low_u16_lo, U16_BITS);
        rc.add_count(rd_low_u16_hi, PC_BITS - U16_BITS);

        uint16_t rs1_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rs1_limbs, rs1_val);

        COL_WRITE_VALUE(row, Rv64JalrCoreCols, imm_sign, imm_sign);
        COL_WRITE_ARRAY(row, Rv64JalrCoreCols, to_pc_limbs, to_pc_limbs);
        COL_WRITE_VALUE(row, Rv64JalrCoreCols, to_pc_least_sig_bit, (to_pc & 1) == 1 ? 1 : 0);
        COL_WRITE_VALUE(row, Rv64JalrCoreCols, is_valid, 1);

        COL_WRITE_ARRAY(row, Rv64JalrCoreCols, rs1_data, rs1_limbs);
        uint32_t rd_limbs[RV64_PTR_U16_LIMBS - 1] = {rd_low_u16_hi};
        COL_WRITE_ARRAY(row, Rv64JalrCoreCols, rd_high, rd_limbs);
        COL_WRITE_VALUE(row, Rv64JalrCoreCols, imm, imm);
    }

    __device__ void fill_trace_row(RowSlice row, Rv64JalrCoreRecord record) {
        fill_trace_row(row, record.from_pc, record.rs1_val, record.imm, record.imm_sign);
    }
};

template <typename T> struct Rv64JalrCols {
    Rv64JalrAdapterCols<T> adapter;
    Rv64JalrCoreCols<T> core;
};

struct Rv64JalrRecord {
    Rv64JalrAdapterRecord adapter;
    Rv64JalrCoreRecord core;
};

__global__ void jalr_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<Rv64JalrRecord> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < records.len()) {
        auto full = records[idx];

        Rv64JalrAdapter adapter(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, full.adapter);

        Rv64JalrCore core(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64JalrCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(Rv64JalrCols<uint8_t>));
    }
}

__global__ void jalr_replay_tracegen(
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
    uint32_t jalr_opcode,
    uint32_t register_as,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(Rv64JalrCols<uint8_t>));
    if (idx >= num_steps) return;

    auto const &step = steps[step_start + idx];
    size_t program_index = step.program_index;
    if (program_index + 1 >= program.len()) {
        preflight_set_error(error, 201);
        return;
    }
    auto const &from = program[program_index];
    auto const &to = program[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % 4 != 0 ||
        from.timestamp > UINT32_MAX - 2 || to.timestamp != from.timestamp + 2) {
        preflight_set_error(error, 202);
        return;
    }
    size_t instruction_index = (from.pc - pc_base) / 4;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 203);
        return;
    }

    auto const &instruction = instructions[instruction_index];
    uint32_t rd_ptr = instruction.words[1];
    uint32_t rs1_ptr = instruction.words[2];
    uint32_t imm = instruction.words[3];
    uint32_t needs_write = instruction.words[6];
    uint32_t imm_sign = instruction.words[7];
    constexpr uint32_t REGISTER_FILE_BYTES = 32 * RV64_REGISTER_NUM_LIMBS;
    bool rd_is_canonical =
        rd_ptr < REGISTER_FILE_BYTES && rd_ptr % RV64_REGISTER_NUM_LIMBS == 0;
    bool rs1_is_canonical =
        rs1_ptr < REGISTER_FILE_BYTES && rs1_ptr % RV64_REGISTER_NUM_LIMBS == 0;
    if (instruction.words[0] != jalr_opcode || instruction.words[4] != register_as ||
        instruction.words[5] != 0 || imm > UINT16_MAX || needs_write > 1 || imm_sign > 1 ||
        needs_write != (rd_ptr != 0) || !rd_is_canonical || !rs1_is_canonical) {
        preflight_set_error(error, 204);
        return;
    }

    size_t read_index = step.memory_start;
    if (read_index >= memory.len() || read_index >= predecessors.len()) {
        preflight_set_error(error, 205);
        return;
    }
    auto const &read = memory[read_index];
    size_t write_index = read_index + 1;
    if (read.timestamp != from.timestamp || preflight_is_write(read) ||
        preflight_address_space(read) != register_as || read.pointer != rs1_ptr / 2) {
        preflight_set_error(error, 205);
        return;
    }
    if (needs_write) {
        if (write_index >= memory.len() || write_index >= predecessors.len()) {
            preflight_set_error(error, 205);
            return;
        }
        auto const &write = memory[write_index];
        if (write.timestamp != from.timestamp + 1 || !preflight_is_write(write) ||
            preflight_address_space(write) != register_as || write.pointer != rd_ptr / 2 ||
            (write_index + 1 < memory.len() &&
             memory[write_index + 1].timestamp < to.timestamp)) {
            preflight_set_error(error, 205);
            return;
        }
    } else if (write_index < memory.len() && memory[write_index].timestamp < to.timestamp) {
        preflight_set_error(error, 205);
        return;
    }

    uint16_t rs1[BLOCK_FE_WIDTH];
    if (!replay_u16_block(read.value, rs1) || rs1[2] != 0 || rs1[3] != 0) {
        preflight_set_error(error, 206);
        return;
    }
    uint32_t rs1_val = static_cast<uint32_t>(rs1[0]) |
                       (static_cast<uint32_t>(rs1[1]) << U16_BITS);
    uint32_t imm_extended = imm + imm_sign * 0xffff0000u;
    int64_t unaligned_signed =
        static_cast<int64_t>(rs1_val) + static_cast<int64_t>(static_cast<int32_t>(imm_extended));
    if (unaligned_signed < 0 ||
        static_cast<uint64_t>(unaligned_signed) >= (uint64_t(1) << PC_BITS)) {
        preflight_set_error(error, 209);
        return;
    }
    constexpr uint32_t MAX_PC = (1u << PC_BITS) - 1;
    if (from.pc > MAX_PC - DEFAULT_PC_STEP) {
        preflight_set_error(error, 209);
        return;
    }
    uint32_t unaligned_to_pc = static_cast<uint32_t>(unaligned_signed);
    if (to.pc != (unaligned_to_pc & ~1u)) {
        preflight_set_error(error, 207);
        return;
    }

    uint16_t expected_rd[BLOCK_FE_WIDTH] = {
        static_cast<uint16_t>(from.pc + DEFAULT_PC_STEP),
        static_cast<uint16_t>((from.pc + DEFAULT_PC_STEP) >> U16_BITS),
        0,
        0,
    };
    ReplayPreviousValue write_previous = {};
    if (needs_write) {
        auto const &write = memory[write_index];
        uint16_t logged_rd[BLOCK_FE_WIDTH];
        if (!replay_u16_block(write.value, logged_rd)) {
            preflight_set_error(error, 206);
            return;
        }
        bool matches = true;
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            matches &= logged_rd[i] == expected_rd[i];
        }
        if (!matches) {
            preflight_set_error(error, 208);
            return;
        }
    }

    ReplayPreviousValue read_previous;
    if (!replay_previous_value(
            read_index, read, predecessors[read_index], memory, seeds, read_previous
        ) ||
        (needs_write &&
         !replay_previous_value(
             write_index,
             memory[write_index],
             predecessors[write_index],
             memory,
             seeds,
             write_previous
         ))) {
        preflight_set_error(error, 210);
        return;
    }

    auto checker = VariableRangeChecker(range_checker, range_checker_num_bins);
    Rv64JalrAdapter adapter(checker, timestamp_max_bits);
    adapter.fill_trace_row(
        row,
        from.pc,
        from.timestamp,
        rs1_ptr,
        rd_ptr,
        needs_write,
        read_previous.timestamp,
        write_previous.timestamp,
        write_previous.value
    );
    Rv64JalrCore core(checker);
    core.fill_trace_row(
        row.slice_from(COL_INDEX(Rv64JalrCols, core)),
        from.pc,
        rs1_val,
        static_cast<uint16_t>(imm),
        imm_sign
    );
}

extern "C" int _jalr_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64JalrRecord> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64JalrCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height, 512);

    jalr_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

extern "C" int _jalr_replay_tracegen(
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
    uint32_t jalr_opcode,
    uint32_t register_as,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64JalrCols<uint8_t>));
    assert(d_memory.len() == d_predecessors.len());
    assert(step_start <= d_steps.len());
    assert(num_steps <= d_steps.len() - step_start);
    assert(height >= num_steps);

    auto [grid, block] = kernel_launch_params(height, 512);
    jalr_replay_tracegen<<<grid, block, 0, stream>>>(
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
        jalr_opcode,
        register_as,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
