#pragma once

#include "primitives/constants.h"
#include "riscv/replay.cuh"

using namespace program;
using namespace riscv;

struct ReplayLoadByteInput {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rs1_ptr;
    uint32_t rd_ptr;
    uint32_t rs1_val;
    uint32_t rs1_prev_timestamp;
    uint32_t read_prev_timestamp;
    uint32_t write_prev_timestamp;
    uint16_t imm;
    uint8_t imm_sign;
    uint8_t shift;
    bool needs_write;
    uint16_t read_data[BLOCK_FE_WIDTH];
    uint16_t write_prev_data[BLOCK_FE_WIDTH];
};

static __device__ bool replay_load_byte(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    RvrReplayStep const &step,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    size_t pointer_max_bits,
    bool sign_extend,
    ReplayLoadByteInput &out,
    uint32_t *error
) {
    size_t program_index = step.program_index;
    if (program_index + 1 >= program.len()) {
        preflight_set_error(error, 221);
        return false;
    }
    auto const &from = program[program_index];
    auto const &to = program[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % DEFAULT_PC_STEP != 0 ||
        from.pc > UINT32_MAX - DEFAULT_PC_STEP || from.timestamp > UINT32_MAX - 3 ||
        to.pc != from.pc + DEFAULT_PC_STEP || to.timestamp != from.timestamp + 3) {
        preflight_set_error(error, 222);
        return false;
    }

    size_t instruction_index = (from.pc - pc_base) / DEFAULT_PC_STEP;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 223);
        return false;
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
    if (instruction.words[0] != expected_opcode || instruction.words[4] != register_as ||
        instruction.words[5] != memory_as || imm > UINT16_MAX || needs_write > 1 ||
        imm_sign > 1 || needs_write != (rd_ptr != 0) || !rd_is_canonical ||
        !rs1_is_canonical) {
        preflight_set_error(error, 224);
        return false;
    }

    size_t rs1_index = step.memory_start;
    size_t read_index = rs1_index + 1;
    size_t next_index = read_index + 1;
    if (read_index >= memory.len() || read_index >= predecessors.len()) {
        preflight_set_error(error, 225);
        return false;
    }
    auto const &rs1_read = memory[rs1_index];
    auto const &read = memory[read_index];
    if (rs1_read.timestamp != from.timestamp || preflight_is_write(rs1_read) ||
        preflight_address_space(rs1_read) != register_as || rs1_read.pointer != rs1_ptr / 2 ||
        read.timestamp != from.timestamp + 1 || preflight_is_write(read) ||
        preflight_address_space(read) != memory_as) {
        preflight_set_error(error, 225);
        return false;
    }

    if (needs_write) {
        if (next_index >= memory.len() || next_index >= predecessors.len()) {
            preflight_set_error(error, 225);
            return false;
        }
        auto const &write = memory[next_index];
        if (write.timestamp != from.timestamp + 2 || !preflight_is_write(write) ||
            preflight_address_space(write) != register_as || write.pointer != rd_ptr / 2) {
            preflight_set_error(error, 225);
            return false;
        }
        next_index++;
    }
    if (next_index < memory.len() && memory[next_index].timestamp < to.timestamp) {
        preflight_set_error(error, 225);
        return false;
    }

    uint16_t rs1[BLOCK_FE_WIDTH];
    uint16_t read_data[BLOCK_FE_WIDTH];
    if (!replay_u16_block(rs1_read.value, rs1) || !replay_u16_block(read.value, read_data) ||
        rs1[2] != 0 || rs1[3] != 0) {
        preflight_set_error(error, 226);
        return false;
    }
    uint32_t rs1_val =
        static_cast<uint32_t>(rs1[0]) | (static_cast<uint32_t>(rs1[1]) << U16_BITS);
    int64_t signed_imm = imm_sign ? static_cast<int64_t>(imm) - (int64_t(1) << U16_BITS)
                                  : static_cast<int64_t>(imm);
    int64_t effective = static_cast<int64_t>(rs1_val) + signed_imm;
    bool exceeds_configured_width =
        pointer_max_bits < 32 &&
        static_cast<uint64_t>(effective) >= (uint64_t(1) << pointer_max_bits);
    if (effective < 0 || effective > UINT32_MAX || exceeds_configured_width) {
        preflight_set_error(error, 227);
        return false;
    }
    uint32_t ptr = static_cast<uint32_t>(effective);
    uint32_t aligned_ptr = ptr & ~(uint32_t(MEMORY_BLOCK_BYTES) - 1);
    if (read.pointer != aligned_ptr / U16_CELL_SIZE) {
        preflight_set_error(error, 228);
        return false;
    }

    uint8_t shift = ptr - aligned_ptr;
    uint16_t selected = load_byte_from_cell(read_data[shift >> 1], shift & 1);
    uint64_t expected_value = sign_extend && (selected & (1u << (RV64_BYTE_BITS - 1)))
                                  ? UINT64_MAX - UINT8_MAX + selected
                                  : selected;
    uint16_t expected_rd[BLOCK_FE_WIDTH];
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        expected_rd[i] = static_cast<uint16_t>(expected_value >> (i * U16_BITS));
    }
    if (needs_write) {
        uint16_t logged_rd[BLOCK_FE_WIDTH];
        if (!replay_u16_block(memory[next_index - 1].value, logged_rd)) {
            preflight_set_error(error, 226);
            return false;
        }
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            if (logged_rd[i] != expected_rd[i]) {
                preflight_set_error(error, 229);
                return false;
            }
        }
    }

    ReplayPreviousValue rs1_previous;
    ReplayPreviousValue read_previous;
    ReplayPreviousValue write_previous = {};
    if (!replay_previous_value(
            rs1_index, rs1_read, predecessors[rs1_index], memory, seeds, rs1_previous
        ) ||
        !replay_previous_value(
            read_index, read, predecessors[read_index], memory, seeds, read_previous
        ) ||
        (needs_write &&
         !replay_previous_value(
             next_index - 1,
             memory[next_index - 1],
             predecessors[next_index - 1],
             memory,
             seeds,
             write_previous
         ))) {
        preflight_set_error(error, 230);
        return false;
    }

    out.from_pc = from.pc;
    out.from_timestamp = from.timestamp;
    out.rs1_ptr = rs1_ptr;
    out.rd_ptr = rd_ptr;
    out.rs1_val = rs1_val;
    out.rs1_prev_timestamp = rs1_previous.timestamp;
    out.read_prev_timestamp = read_previous.timestamp;
    out.write_prev_timestamp = write_previous.timestamp;
    out.imm = static_cast<uint16_t>(imm);
    out.imm_sign = imm_sign;
    out.shift = shift;
    out.needs_write = needs_write;
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        out.read_data[i] = read_data[i];
        out.write_prev_data[i] = write_previous.value[i];
    }
    return true;
}
