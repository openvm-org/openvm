#pragma once

#include "arch/rvr/replay.cuh"
#include "primitives/constants.h"

using namespace program;
using namespace riscv;

static constexpr uint32_t REVEAL_REPLAY_ERROR = 271;

struct ReplayRevealInput {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t src_ptr;
    uint32_t base_ptr;
    uint32_t base_value;
    uint32_t base_prev_timestamp;
    uint32_t src_prev_timestamp;
    uint32_t write_prev_timestamp[2];
    uint16_t imm;
    uint8_t imm_sign;
    uint16_t src_data[BLOCK_FE_WIDTH];
    uint8_t src_bytes[MEMORY_BLOCK_BYTES];
    uint8_t write_prev_data[2][BLOCK_FE_WIDTH];
};

static __device__ bool replay_reveal(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    RvrReplayStep const &step,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t public_values_as,
    size_t pointer_max_bits,
    ReplayRevealInput &out,
    uint32_t *error
) {
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
            REVEAL_REPLAY_ERROR
        )) {
        return false;
    }
    auto const &from = *transition.from;
    auto const &to = *transition.to;
    auto const &instruction = *transition.instruction;
    uint32_t src_ptr = instruction.words[1];
    uint32_t base_ptr = instruction.words[2];
    uint32_t imm = instruction.words[3];
    uint32_t is_valid = instruction.words[6];
    uint32_t imm_sign = instruction.words[7];
    if (instruction.words[0] != expected_opcode ||
        instruction.words[4] != register_as ||
        instruction.words[5] != public_values_as || imm > UINT16_MAX ||
        is_valid != 1 || imm_sign > 1 || !replay_canonical_register_pointer(src_ptr) ||
        !replay_canonical_register_pointer(base_ptr)) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 1);
        return false;
    }

    size_t base_index = step.memory_start;
    if (base_index >= memory.len() || memory.len() - base_index < 4 ||
        base_index >= predecessors.len() || predecessors.len() - base_index < 4) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 2);
        return false;
    }
    size_t src_index = base_index + 1;
    size_t write_index = base_index + 2;
    size_t write_second_index = base_index + 3;
    auto const &base_read = memory[base_index];
    auto const &src_read = memory[src_index];
    auto const &write = memory[write_index];
    auto const &write_second = memory[write_second_index];
    if (base_read.timestamp != from.timestamp || preflight_is_write(base_read) ||
        preflight_address_space(base_read) != register_as ||
        base_read.pointer != base_ptr / U16_CELL_SIZE ||
        src_read.timestamp != from.timestamp + 1 || preflight_is_write(src_read) ||
        preflight_address_space(src_read) != register_as ||
        src_read.pointer != src_ptr / U16_CELL_SIZE ||
        write.timestamp != from.timestamp + 2 || !preflight_is_write(write) ||
        preflight_address_space(write) != public_values_as ||
        write_second.timestamp != from.timestamp + 3 || !preflight_is_write(write_second) ||
        preflight_address_space(write_second) != public_values_as) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 2);
        return false;
    }
    size_t next_index = write_second_index + 1;
    if (next_index < memory.len() && memory[next_index].timestamp < to.timestamp) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 2);
        return false;
    }

    uint16_t base[BLOCK_FE_WIDTH];
    uint16_t src[BLOCK_FE_WIDTH];
    uint8_t logged_post[2][BLOCK_FE_WIDTH];
    replay_u16_block(base_read.value, base);
    replay_u16_block(src_read.value, src);
    preflight_decode_u8_block(write.value, logged_post[0]);
    preflight_decode_u8_block(write_second.value, logged_post[1]);
    if (base[PTR_U16_LIMBS] != 0 || base[PTR_U16_LIMBS + 1] != 0) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 3);
        return false;
    }

    uint32_t base_value =
        static_cast<uint32_t>(base[0]) | (static_cast<uint32_t>(base[1]) << U16_BITS);
    int64_t signed_imm = imm_sign ? static_cast<int64_t>(imm) - (int64_t(1) << U16_BITS)
                                  : static_cast<int64_t>(imm);
    int64_t effective = static_cast<int64_t>(base_value) + signed_imm;
    if (effective < 0 || effective > UINT32_MAX ||
        (static_cast<uint32_t>(effective) & (MEMORY_BLOCK_BYTES - 1)) != 0) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 4);
        return false;
    }
    uint64_t domain_end =
        pointer_max_bits < 32 ? (uint64_t(1) << pointer_max_bits) : (uint64_t(1) << 32);
    uint32_t reveal_ptr = static_cast<uint32_t>(effective);
    if (static_cast<uint64_t>(reveal_ptr) + MEMORY_BLOCK_BYTES > domain_end ||
        write.pointer != reveal_ptr || write_second.pointer != reveal_ptr + BLOCK_FE_WIDTH) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 4);
        return false;
    }
#pragma unroll
    for (size_t i = 0; i < MEMORY_BLOCK_BYTES; i++) {
        uint8_t byte = uint8_t(src[i / U16_CELL_SIZE] >> ((i % U16_CELL_SIZE) * 8));
        if (logged_post[i / BLOCK_FE_WIDTH][i % BLOCK_FE_WIDTH] != byte) {
            preflight_set_error(error, REVEAL_REPLAY_ERROR + 5);
            return false;
        }
        out.src_bytes[i] = byte;
    }

    ReplayPreviousValue base_previous;
    ReplayPreviousValue src_previous;
    ReplayPreviousValue write_previous[2];
    if (!replay_previous_value(
            base_index, base_read, predecessors[base_index], memory, seeds, base_previous
        ) ||
        !replay_previous_value(
            src_index, src_read, predecessors[src_index], memory, seeds, src_previous
        ) ||
        !replay_previous_value(
            write_index, write, predecessors[write_index], memory, seeds, write_previous[0]
        ) ||
        !replay_previous_value(
            write_second_index,
            write_second,
            predecessors[write_second_index],
            memory,
            seeds,
            write_previous[1]
        )) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 6);
        return false;
    }

    out.from_pc = from.pc;
    out.from_timestamp = from.timestamp;
    out.src_ptr = src_ptr;
    out.base_ptr = base_ptr;
    out.base_value = base_value;
    out.base_prev_timestamp = base_previous.timestamp;
    out.src_prev_timestamp = src_previous.timestamp;
    out.imm = static_cast<uint16_t>(imm);
    out.imm_sign = imm_sign;
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        out.src_data[i] = src[i];
    }
    for (size_t block = 0; block < 2; block++) {
        out.write_prev_timestamp[block] = write_previous[block].timestamp;
        preflight_decode_u8_block(write_previous[block].value, out.write_prev_data[block]);
    }
    return true;
}
