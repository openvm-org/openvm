#pragma once

#include "primitives/constants.h"
#include "riscv/replay.cuh"

using namespace program;
using namespace riscv;

struct ReplayLoadMultiByteInput {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rs1_ptr;
    uint32_t rd_ptr;
    uint32_t rs1_val;
    uint32_t rs1_prev_timestamp;
    uint32_t block0_prev_timestamp;
    uint32_t block1_prev_timestamp;
    uint32_t write_prev_timestamp;
    uint16_t imm;
    uint8_t imm_sign;
    uint8_t shift;
    bool crosses;
    bool needs_write;
    uint16_t read_data[2][BLOCK_FE_WIDTH];
    uint16_t write_prev_data[BLOCK_FE_WIDTH];
};

static __device__ __forceinline__ uint8_t replay_load_byte(
    uint16_t const (&read_data)[2][BLOCK_FE_WIDTH],
    uint32_t byte
) {
    uint32_t cell = byte >> 1;
    uint16_t value = read_data[cell / BLOCK_FE_WIDTH][cell % BLOCK_FE_WIDTH];
    return static_cast<uint8_t>((value >> ((byte & 1) * RV64_BYTE_BITS)) & UINT8_MAX);
}

template <size_t WIDTH_BYTES, bool SIGN_EXTEND>
static __device__ bool replay_load_multibyte(
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
    ReplayLoadMultiByteInput &out,
    uint32_t *error
) {
    static_assert(WIDTH_BYTES == WORD_ACCESS_WIDTH || WIDTH_BYTES == DOUBLEWORD_ACCESS_WIDTH);
    static_assert(!SIGN_EXTEND || WIDTH_BYTES == WORD_ACCESS_WIDTH);

    size_t program_index = step.program_index;
    if (program_index + 1 >= program.len()) {
        preflight_set_error(error, 241);
        return false;
    }
    auto const &from = program[program_index];
    auto const &to = program[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % DEFAULT_PC_STEP != 0 ||
        from.pc > UINT32_MAX - DEFAULT_PC_STEP || from.timestamp > UINT32_MAX - 4 ||
        to.pc != from.pc + DEFAULT_PC_STEP || to.timestamp != from.timestamp + 4) {
        preflight_set_error(error, 242);
        return false;
    }

    size_t instruction_index = (from.pc - pc_base) / DEFAULT_PC_STEP;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, 243);
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
        preflight_set_error(error, 244);
        return false;
    }

    size_t rs1_index = step.memory_start;
    size_t block0_index = rs1_index + 1;
    if (block0_index >= memory.len() || block0_index >= predecessors.len()) {
        preflight_set_error(error, 245);
        return false;
    }
    auto const &rs1_read = memory[rs1_index];
    auto const &block0_read = memory[block0_index];
    if (rs1_read.timestamp != from.timestamp || preflight_is_write(rs1_read) ||
        preflight_address_space(rs1_read) != register_as || rs1_read.pointer != rs1_ptr / 2 ||
        block0_read.timestamp != from.timestamp + 1 || preflight_is_write(block0_read) ||
        preflight_address_space(block0_read) != memory_as) {
        preflight_set_error(error, 245);
        return false;
    }

    uint16_t rs1[BLOCK_FE_WIDTH];
    uint16_t read_data[2][BLOCK_FE_WIDTH] = {};
    if (!replay_u16_block(rs1_read.value, rs1) ||
        !replay_u16_block(block0_read.value, read_data[0]) || rs1[2] != 0 || rs1[3] != 0) {
        preflight_set_error(error, 246);
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
        preflight_set_error(error, 247);
        return false;
    }
    uint32_t ptr = static_cast<uint32_t>(effective);
    uint32_t aligned_ptr = ptr & ~(uint32_t(MEMORY_BLOCK_BYTES) - 1);
    uint8_t shift = ptr - aligned_ptr;
    bool crosses = shift + WIDTH_BYTES > MEMORY_BLOCK_BYTES;
    if (block0_read.pointer != aligned_ptr / U16_CELL_SIZE) {
        preflight_set_error(error, 248);
        return false;
    }

    size_t next_index = block0_index + 1;
    size_t block1_index = SIZE_MAX;
    if (crosses) {
        uint64_t block1_ptr = static_cast<uint64_t>(aligned_ptr) + MEMORY_BLOCK_BYTES;
        bool block1_exceeds_configured_width =
            pointer_max_bits < 32 &&
            block1_ptr + MEMORY_BLOCK_BYTES > (uint64_t(1) << pointer_max_bits);
        if (block1_ptr > UINT32_MAX || block1_exceeds_configured_width ||
            next_index >= memory.len() || next_index >= predecessors.len()) {
            preflight_set_error(error, 248);
            return false;
        }
        auto const &block1_read = memory[next_index];
        if (block1_read.timestamp != from.timestamp + 2 || preflight_is_write(block1_read) ||
            preflight_address_space(block1_read) != memory_as ||
            block1_read.pointer != block1_ptr / U16_CELL_SIZE ||
            !replay_u16_block(block1_read.value, read_data[1])) {
            preflight_set_error(error, 248);
            return false;
        }
        block1_index = next_index;
        next_index++;
    } else if (next_index < memory.len() &&
               memory[next_index].timestamp < from.timestamp + 3) {
        preflight_set_error(error, 245);
        return false;
    }

    size_t write_index = SIZE_MAX;
    if (needs_write) {
        if (next_index >= memory.len() || next_index >= predecessors.len()) {
            preflight_set_error(error, 245);
            return false;
        }
        auto const &write = memory[next_index];
        if (write.timestamp != from.timestamp + 3 || !preflight_is_write(write) ||
            preflight_address_space(write) != register_as || write.pointer != rd_ptr / 2) {
            preflight_set_error(error, 245);
            return false;
        }
        write_index = next_index;
        next_index++;
    }
    if (next_index < memory.len() && memory[next_index].timestamp < to.timestamp) {
        preflight_set_error(error, 245);
        return false;
    }

    uint16_t expected_rd[BLOCK_FE_WIDTH] = {};
#pragma unroll
    for (size_t byte = 0; byte < WIDTH_BYTES; byte++) {
        expected_rd[byte >> 1] |=
            static_cast<uint16_t>(replay_load_byte(read_data, shift + byte))
            << ((byte & 1) * RV64_BYTE_BITS);
    }
    if constexpr (SIGN_EXTEND) {
        if (expected_rd[WIDTH_BYTES / U16_CELL_SIZE - 1] & (uint16_t(1) << (U16_BITS - 1))) {
#pragma unroll
            for (size_t cell = WIDTH_BYTES / U16_CELL_SIZE; cell < BLOCK_FE_WIDTH; cell++) {
                expected_rd[cell] = UINT16_MAX;
            }
        }
    }
    if (needs_write) {
        uint16_t logged_rd[BLOCK_FE_WIDTH];
        if (!replay_u16_block(memory[write_index].value, logged_rd)) {
            preflight_set_error(error, 246);
            return false;
        }
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            if (logged_rd[i] != expected_rd[i]) {
                preflight_set_error(error, 249);
                return false;
            }
        }
    }

    ReplayPreviousValue rs1_previous;
    ReplayPreviousValue block0_previous;
    ReplayPreviousValue block1_previous = {};
    ReplayPreviousValue write_previous = {};
    if (!replay_previous_value(
            rs1_index, rs1_read, predecessors[rs1_index], memory, seeds, rs1_previous
        ) ||
        !replay_previous_value(
            block0_index,
            block0_read,
            predecessors[block0_index],
            memory,
            seeds,
            block0_previous
        ) ||
        (crosses &&
         !replay_previous_value(
             block1_index,
             memory[block1_index],
             predecessors[block1_index],
             memory,
             seeds,
             block1_previous
         )) ||
        (needs_write &&
         !replay_previous_value(
             write_index,
             memory[write_index],
             predecessors[write_index],
             memory,
             seeds,
             write_previous
         ))) {
        preflight_set_error(error, 250);
        return false;
    }

    out.from_pc = from.pc;
    out.from_timestamp = from.timestamp;
    out.rs1_ptr = rs1_ptr;
    out.rd_ptr = rd_ptr;
    out.rs1_val = rs1_val;
    out.rs1_prev_timestamp = rs1_previous.timestamp;
    out.block0_prev_timestamp = block0_previous.timestamp;
    out.block1_prev_timestamp = block1_previous.timestamp;
    out.write_prev_timestamp = write_previous.timestamp;
    out.imm = static_cast<uint16_t>(imm);
    out.imm_sign = imm_sign;
    out.shift = shift;
    out.crosses = crosses;
    out.needs_write = needs_write;
#pragma unroll
    for (size_t block = 0; block < 2; block++) {
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            out.read_data[block][i] = read_data[block][i];
        }
    }
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        out.write_prev_data[i] = write_previous.value[i];
    }
    return true;
}
