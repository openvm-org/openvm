#pragma once

#include "primitives/constants.h"
#include "arch/rvr/replay.cuh"

using namespace program;
using namespace riscv;

struct ReplayStoreMultiByteInput {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rs1_ptr;
    uint32_t rs2_ptr;
    uint32_t rs1_val;
    uint32_t rs1_prev_timestamp;
    uint32_t rs2_prev_timestamp;
    uint32_t write_prev_timestamps[2];
    uint32_t memory_as;
    uint16_t imm;
    uint8_t imm_sign;
    uint8_t shift;
    uint16_t read_data[BLOCK_FE_WIDTH];
    uint16_t prev_data[2][BLOCK_FE_WIDTH];
};

static __device__ __forceinline__ uint8_t
replay_store_source_byte(uint16_t const (&value)[BLOCK_FE_WIDTH], size_t byte) {
    uint16_t cell = value[byte / U16_CELL_SIZE];
    return static_cast<uint8_t>(
        (cell >> ((byte % U16_CELL_SIZE) * RV64_BYTE_BITS)) & UINT8_MAX
    );
}

template <size_t WIDTH_BYTES>
static __device__ bool replay_store_multibyte(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    RvrReplayStep const &step,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t main_memory_as,
    uint32_t public_values_as,
    size_t pointer_max_bits,
    ReplayStoreMultiByteInput &out,
    uint32_t *error
) {
    static_assert(
        WIDTH_BYTES == HALFWORD_ACCESS_WIDTH || WIDTH_BYTES == WORD_ACCESS_WIDTH ||
        WIDTH_BYTES == DOUBLEWORD_ACCESS_WIDTH
    );

    ReplayProgramTransition transition;
    if (!replay_program_transition(
            instructions,
            pc_base,
            program,
            step.program_index,
            4u,
            ReplayPcEffect::Sequential,
            transition,
            error,
            261
        )) {
        return false;
    }
    auto const &from = *transition.from;
    auto const &to = *transition.to;
    auto const &instruction = *transition.instruction;
    uint32_t rs2_ptr = instruction.words[1];
    uint32_t rs1_ptr = instruction.words[2];
    uint32_t imm = instruction.words[3];
    uint32_t memory_as = instruction.words[5];
    uint32_t is_valid = instruction.words[6];
    uint32_t imm_sign = instruction.words[7];
    constexpr uint32_t REGISTER_FILE_BYTES = 32 * RV64_REGISTER_NUM_LIMBS;
    bool rs1_is_canonical =
        rs1_ptr < REGISTER_FILE_BYTES && rs1_ptr % RV64_REGISTER_NUM_LIMBS == 0;
    bool rs2_is_canonical =
        rs2_ptr < REGISTER_FILE_BYTES && rs2_ptr % RV64_REGISTER_NUM_LIMBS == 0;
    if (instruction.words[0] != expected_opcode || instruction.words[4] != register_as ||
        (memory_as != main_memory_as && memory_as != public_values_as) || imm > UINT16_MAX ||
        is_valid != 1 ||
        imm_sign > 1 || !rs1_is_canonical || !rs2_is_canonical) {
        preflight_set_error(error, 264);
        return false;
    }

    size_t rs1_index = step.memory_start;
    if (rs1_index >= memory.len() || memory.len() - rs1_index < 3 ||
        rs1_index >= predecessors.len() || predecessors.len() - rs1_index < 3) {
        preflight_set_error(error, 265);
        return false;
    }
    size_t rs2_index = rs1_index + 1;
    size_t write0_index = rs1_index + 2;
    auto const &rs1_read = memory[rs1_index];
    auto const &rs2_read = memory[rs2_index];
    auto const &write0 = memory[write0_index];
    if (rs1_read.timestamp != from.timestamp || preflight_is_write(rs1_read) ||
        preflight_address_space(rs1_read) != register_as || rs1_read.pointer != rs1_ptr / 2 ||
        rs2_read.timestamp != from.timestamp + 1 || preflight_is_write(rs2_read) ||
        preflight_address_space(rs2_read) != register_as || rs2_read.pointer != rs2_ptr / 2 ||
        write0.timestamp != from.timestamp + 2 || !preflight_is_write(write0) ||
        preflight_address_space(write0) != memory_as) {
        preflight_set_error(error, 265);
        return false;
    }

    uint16_t rs1[BLOCK_FE_WIDTH];
    uint16_t rs2[BLOCK_FE_WIDTH];
    uint16_t logged_post[2][BLOCK_FE_WIDTH] = {};
    if (!replay_u16_block(rs1_read.value, rs1) || !replay_u16_block(rs2_read.value, rs2) ||
        !replay_u16_block(write0.value, logged_post[0]) || rs1[2] != 0 || rs1[3] != 0) {
        preflight_set_error(error, 266);
        return false;
    }

    uint32_t rs1_val =
        static_cast<uint32_t>(rs1[0]) | (static_cast<uint32_t>(rs1[1]) << U16_BITS);
    int64_t signed_imm = imm_sign ? static_cast<int64_t>(imm) - (int64_t(1) << U16_BITS)
                                  : static_cast<int64_t>(imm);
    int64_t effective = static_cast<int64_t>(rs1_val) + signed_imm;
    if (effective < 0 || effective > UINT32_MAX) {
        preflight_set_error(error, 267);
        return false;
    }
    uint64_t domain_end =
        pointer_max_bits < 32 ? (uint64_t(1) << pointer_max_bits) : (uint64_t(1) << 32);
    uint64_t access_end = static_cast<uint64_t>(effective) + WIDTH_BYTES;
    if (access_end > domain_end) {
        preflight_set_error(error, 267);
        return false;
    }

    uint32_t ptr = static_cast<uint32_t>(effective);
    uint32_t aligned_ptr = ptr & ~(uint32_t(MEMORY_BLOCK_BYTES) - 1);
    uint8_t shift = ptr - aligned_ptr;
    bool crosses = shift + WIDTH_BYTES > MEMORY_BLOCK_BYTES;
    if (write0.pointer != aligned_ptr / U16_CELL_SIZE) {
        preflight_set_error(error, 268);
        return false;
    }

    size_t next_index = write0_index + 1;
    size_t write1_index = SIZE_MAX;
    if (crosses) {
        uint64_t block1_ptr = static_cast<uint64_t>(aligned_ptr) + MEMORY_BLOCK_BYTES;
        if (block1_ptr + MEMORY_BLOCK_BYTES > domain_end || next_index >= memory.len() ||
            next_index >= predecessors.len()) {
            preflight_set_error(error, 268);
            return false;
        }
        auto const &write1 = memory[next_index];
        if (write1.timestamp != from.timestamp + 3 || !preflight_is_write(write1) ||
            preflight_address_space(write1) != memory_as ||
            write1.pointer != block1_ptr / U16_CELL_SIZE ||
            !replay_u16_block(write1.value, logged_post[1])) {
            preflight_set_error(error, 268);
            return false;
        }
        write1_index = next_index;
        next_index++;
    }
    if (next_index < memory.len() && memory[next_index].timestamp < to.timestamp) {
        preflight_set_error(error, 265);
        return false;
    }

    ReplayPreviousValue rs1_previous;
    ReplayPreviousValue rs2_previous;
    ReplayPreviousValue write0_previous;
    ReplayPreviousValue write1_previous = {};
    if (!replay_previous_value(
            rs1_index, rs1_read, predecessors[rs1_index], memory, seeds, rs1_previous
        ) ||
        !replay_previous_value(
            rs2_index, rs2_read, predecessors[rs2_index], memory, seeds, rs2_previous
        ) ||
        !replay_previous_value(
            write0_index, write0, predecessors[write0_index], memory, seeds, write0_previous
        ) ||
        (crosses &&
         !replay_previous_value(
             write1_index,
             memory[write1_index],
             predecessors[write1_index],
             memory,
             seeds,
             write1_previous
         ))) {
        preflight_set_error(error, 269);
        return false;
    }

    uint16_t expected_post[2][BLOCK_FE_WIDTH] = {};
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        expected_post[0][i] = write0_previous.value[i];
        if (crosses) expected_post[1][i] = write1_previous.value[i];
    }
#pragma unroll
    for (size_t byte = 0; byte < WIDTH_BYTES; byte++) {
        size_t destination = shift + byte;
        size_t block = destination / MEMORY_BLOCK_BYTES;
        size_t within_block = destination % MEMORY_BLOCK_BYTES;
        size_t cell = within_block / U16_CELL_SIZE;
        size_t byte_in_cell = within_block % U16_CELL_SIZE;
        uint16_t mask = static_cast<uint16_t>(UINT8_MAX << (byte_in_cell * RV64_BYTE_BITS));
        expected_post[block][cell] = static_cast<uint16_t>(
            (expected_post[block][cell] & ~mask) |
            (static_cast<uint16_t>(replay_store_source_byte(rs2, byte))
             << (byte_in_cell * RV64_BYTE_BITS))
        );
    }
#pragma unroll
    for (size_t block = 0; block < 2; block++) {
        if (block == 1 && !crosses) break;
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            if (logged_post[block][i] != expected_post[block][i]) {
                preflight_set_error(error, 270);
                return false;
            }
        }
    }

    out.from_pc = from.pc;
    out.from_timestamp = from.timestamp;
    out.rs1_ptr = rs1_ptr;
    out.rs2_ptr = rs2_ptr;
    out.rs1_val = rs1_val;
    out.rs1_prev_timestamp = rs1_previous.timestamp;
    out.rs2_prev_timestamp = rs2_previous.timestamp;
    out.write_prev_timestamps[0] = write0_previous.timestamp;
    out.write_prev_timestamps[1] = crosses ? write1_previous.timestamp : UINT32_MAX;
    out.memory_as = memory_as;
    out.imm = static_cast<uint16_t>(imm);
    out.imm_sign = imm_sign;
    out.shift = shift;
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        out.read_data[i] = rs2[i];
        out.prev_data[0][i] = write0_previous.value[i];
        out.prev_data[1][i] = crosses ? write1_previous.value[i] : 0;
    }
    return true;
}
