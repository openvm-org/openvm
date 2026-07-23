#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_imm_u16.cuh"
#include "riscv/cores/addi.cuh"
#include "riscv/replay.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

using Rv64AddIWCoreRecord = AddICoreRecord<RV64_WORD_U16_LIMBS>;
using Rv64AddIWCore = AddICore<RV64_WORD_U16_LIMBS, U16_BITS, false>;
template <typename T> using Rv64AddIWCoreCols = AddICoreCols<T, RV64_WORD_U16_LIMBS>;

template <typename T> struct Rv64AddIWCols {
    Rv64BaseAluWImmU16AdapterCols<T> adapter;
    Rv64AddIWCoreCols<T> core;
};

struct Rv64AddIWRecord {
    Rv64BaseAluWImmU16AdapterRecord adapter;
    Rv64AddIWCoreRecord core;
};

static_assert(sizeof(Rv64AddIWRecord) == 48);
static_assert(offsetof(Rv64AddIWRecord, core) == 40);

__global__ void addi_w_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64AddIWRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluWImmU16Adapter(
            VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Rv64AddIWCore(VariableRangeChecker(range_ptr, range_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64AddIWCols, core)), rec.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _addi_w_tracegen(
    Fp *__restrict__ trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64AddIWRecord> records,
    uint32_t *__restrict__ range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddIWCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    addi_w_tracegen<<<grid, block, 0, stream>>>(
        trace, height, width, records, range_ptr, range_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

__global__ void addi_w_replay_tracegen(
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
    uint32_t addiw_opcode,
    uint32_t register_address_space,
    uint32_t immediate_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(Rv64AddIWCols<uint8_t>));
    if (idx >= num_steps) return;

    auto const &step = steps[step_start + idx];
    size_t program_index = step.program_index;
    if (program_index + 1 >= program.len()) {
        preflight_set_error(error, 31);
        return;
    }
    auto const &from = program[program_index];
    auto const &to = program[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % 4 != 0 ||
        from.timestamp + 2 != to.timestamp || to.pc != from.pc + 4) {
        preflight_set_error(error, 32);
        return;
    }
    size_t instruction_index = (from.pc - pc_base) / 4;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 33);
        return;
    }
    auto const &instruction = instructions[instruction_index];
    uint32_t rd_ptr = instruction.words[1];
    uint32_t rs1_ptr = instruction.words[2];
    uint32_t encoded_imm = instruction.words[3];
    // The RV64 transpiler replaces I-type writes to x0 with NOP. Reject a
    // hand-built ADDIW/x0 instead of inventing a memory event for its disabled
    // destination-write clock slot.
    if (instruction.words[0] != addiw_opcode ||
        instruction.words[4] != register_address_space ||
        instruction.words[5] != immediate_address_space || rd_ptr == 0 || (rd_ptr & 1) != 0 ||
        (rs1_ptr & 1) != 0) {
        preflight_set_error(error, 34);
        return;
    }

    size_t read_index = step.memory_start;
    size_t write_index = read_index + 1;
    if (write_index >= memory.len() || write_index >= predecessors.len()) {
        preflight_set_error(error, 35);
        return;
    }
    auto const &read = memory[read_index];
    auto const &write = memory[write_index];
    if (read.timestamp != from.timestamp || preflight_is_write(read) ||
        preflight_address_space(read) != register_address_space || read.pointer != rs1_ptr / 2 ||
        write.timestamp != from.timestamp + 1 || !preflight_is_write(write) ||
        preflight_address_space(write) != register_address_space || write.pointer != rd_ptr / 2 ||
        (write_index + 1 < memory.len() && memory[write_index + 1].timestamp < to.timestamp)) {
        preflight_set_error(error, 36);
        return;
    }

    uint32_t imm_low11 = encoded_imm & 0x7ff;
    uint32_t imm_sign = (encoded_imm >> 11) & 1;
    if (encoded_imm != imm_low11 + imm_sign * 0xfff800) {
        preflight_set_error(error, 37);
        return;
    }
    uint16_t rs1[BLOCK_FE_WIDTH];
    uint16_t logged_rd[BLOCK_FE_WIDTH];
    if (!replay_u16_block(read.value, rs1) || !replay_u16_block(write.value, logged_rd)) {
        preflight_set_error(error, 38);
        return;
    }
    uint16_t expected_low[RV64_WORD_U16_LIMBS];
    uint32_t overflow = static_cast<uint32_t>(rs1[0]) + imm_low11 +
                        imm_sign * ((1u << U16_BITS) - (1u << 11));
    uint32_t carry = overflow >> U16_BITS;
    expected_low[0] = overflow & ((1u << U16_BITS) - 1);
    overflow = static_cast<uint32_t>(rs1[1]) +
               imm_sign * ((1u << U16_BITS) - 1) + carry;
    expected_low[1] = overflow & ((1u << U16_BITS) - 1);
    uint16_t sign_extension = (expected_low[1] >> (U16_BITS - 1)) != 0 ? 0xffffu : 0u;
    if (logged_rd[0] != expected_low[0] || logged_rd[1] != expected_low[1] ||
        logged_rd[2] != sign_extension || logged_rd[3] != sign_extension) {
        preflight_set_error(error, 39);
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
        preflight_set_error(error, 40);
        return;
    }

    uint16_t rs1_high[RV64_WORD_U16_LIMBS] = {rs1[2], rs1[3]};
    uint16_t rs1_low[RV64_WORD_U16_LIMBS] = {rs1[0], rs1[1]};
    auto checker = VariableRangeChecker(range_checker, range_checker_num_bins);
    auto adapter = Rv64BaseAluWImmU16Adapter(checker, timestamp_max_bits);
    adapter.fill_trace_row(
        row,
        from.pc,
        from.timestamp,
        rd_ptr,
        rs1_ptr,
        rs1_high,
        expected_low[1],
        read_previous.timestamp,
        write_previous.timestamp,
        write_previous.value
    );
    auto core = Rv64AddIWCore(checker);
    core.fill_trace_row(
        row.slice_from(COL_INDEX(Rv64AddIWCols, core)),
        rs1_low,
        static_cast<uint16_t>(imm_low11),
        static_cast<uint16_t>(imm_sign)
    );
}

extern "C" int _addi_w_replay_tracegen(
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
    size_t step_start,
    size_t num_steps,
    uint32_t *error,
    uint32_t addiw_opcode,
    uint32_t register_address_space,
    uint32_t immediate_address_space,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddIWCols<uint8_t>));
    assert(memory.len() == predecessors.len());
    assert(step_start <= steps.len());
    assert(num_steps <= steps.len() - step_start);
    assert(height >= num_steps);
    auto [grid, block] = kernel_launch_params(height, 512);
    addi_w_replay_tracegen<<<grid, block, 0, stream>>>(
        trace,
        height,
        instructions,
        pc_base,
        program,
        memory,
        seeds,
        predecessors,
        steps,
        step_start,
        num_steps,
        error,
        addiw_opcode,
        register_address_space,
        immediate_address_space,
        range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
