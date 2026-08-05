#pragma once

#include "arch/rvr/replay.cuh"
#include "primitives/constants.h"

using namespace program;
using namespace riscv;

static constexpr uint32_t REVEAL_REPLAY_ERROR = 271;
static constexpr size_t REVEAL_ACCESS_WIDTH = 8;

struct ReplayRevealInput {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t src_ptr;
    uint32_t base_ptr;
    uint32_t base_value;
    uint32_t base_prev_timestamp;
    uint32_t src_prev_timestamp;
    uint32_t write_prev_timestamps[2];
    uint16_t imm;
    uint8_t imm_sign;
    uint8_t shift;
    uint16_t src_data[BLOCK_FE_WIDTH];
    uint16_t prev_data[2][BLOCK_FE_WIDTH];
};

static __device__ __forceinline__ uint8_t
replay_reveal_source_byte(uint16_t const (&value)[BLOCK_FE_WIDTH], size_t byte) {
    uint16_t cell = value[byte / U16_CELL_SIZE];
    return static_cast<uint8_t>(
        (cell >> ((byte % U16_CELL_SIZE) * BYTE_BITS)) & UINT8_MAX
    );
}

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
    if (base_index >= memory.len() || memory.len() - base_index < 3 ||
        base_index >= predecessors.len() || predecessors.len() - base_index < 3) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 2);
        return false;
    }
    size_t src_index = base_index + 1;
    size_t write0_index = base_index + 2;
    auto const &base_read = memory[base_index];
    auto const &src_read = memory[src_index];
    auto const &write0 = memory[write0_index];
    if (base_read.timestamp != from.timestamp || preflight_is_write(base_read) ||
        preflight_address_space(base_read) != register_as ||
        base_read.pointer != base_ptr / U16_CELL_SIZE ||
        src_read.timestamp != from.timestamp + 1 || preflight_is_write(src_read) ||
        preflight_address_space(src_read) != register_as ||
        src_read.pointer != src_ptr / U16_CELL_SIZE ||
        write0.timestamp != from.timestamp + 2 || !preflight_is_write(write0) ||
        preflight_address_space(write0) != public_values_as) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 2);
        return false;
    }

    uint16_t base[BLOCK_FE_WIDTH];
    uint16_t src[BLOCK_FE_WIDTH];
    uint16_t logged_post[2][BLOCK_FE_WIDTH] = {};
    replay_u16_block(base_read.value, base);
    replay_u16_block(src_read.value, src);
    replay_u16_block(write0.value, logged_post[0]);
    if (base[PTR_U16_LIMBS] != 0 || base[PTR_U16_LIMBS + 1] != 0) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 3);
        return false;
    }

    uint32_t base_value =
        static_cast<uint32_t>(base[0]) | (static_cast<uint32_t>(base[1]) << U16_BITS);
    int64_t signed_imm = imm_sign ? static_cast<int64_t>(imm) - (int64_t(1) << U16_BITS)
                                  : static_cast<int64_t>(imm);
    int64_t effective = static_cast<int64_t>(base_value) + signed_imm;
    if (effective < 0 || effective > UINT32_MAX) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 4);
        return false;
    }
    uint64_t domain_end =
        pointer_max_bits < 32 ? (uint64_t(1) << pointer_max_bits) : (uint64_t(1) << 32);
    if (static_cast<uint64_t>(effective) + REVEAL_ACCESS_WIDTH > domain_end) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 4);
        return false;
    }

    uint32_t ptr = static_cast<uint32_t>(effective);
    uint32_t aligned_ptr = ptr & ~(uint32_t(MEMORY_BLOCK_BYTES) - 1);
    uint8_t shift = ptr - aligned_ptr;
    bool crosses = shift != 0;
    if (write0.pointer != aligned_ptr / U16_CELL_SIZE) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 5);
        return false;
    }

    size_t next_index = write0_index + 1;
    size_t write1_index = SIZE_MAX;
    if (crosses) {
        uint64_t block1_ptr = static_cast<uint64_t>(aligned_ptr) + MEMORY_BLOCK_BYTES;
        if (block1_ptr + MEMORY_BLOCK_BYTES > domain_end || next_index >= memory.len() ||
            next_index >= predecessors.len()) {
            preflight_set_error(error, REVEAL_REPLAY_ERROR + 5);
            return false;
        }
        auto const &write1 = memory[next_index];
        if (write1.timestamp != from.timestamp + 3 || !preflight_is_write(write1) ||
            preflight_address_space(write1) != public_values_as ||
            write1.pointer != block1_ptr / U16_CELL_SIZE) {
            preflight_set_error(error, REVEAL_REPLAY_ERROR + 5);
            return false;
        }
        replay_u16_block(write1.value, logged_post[1]);
        write1_index = next_index;
        next_index++;
    }
    if (next_index < memory.len() && memory[next_index].timestamp < to.timestamp) {
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 2);
        return false;
    }

    ReplayPreviousValue base_previous;
    ReplayPreviousValue src_previous;
    ReplayPreviousValue write0_previous;
    ReplayPreviousValue write1_previous = {};
    if (!replay_previous_value(
            base_index, base_read, predecessors[base_index], memory, seeds, base_previous
        ) ||
        !replay_previous_value(
            src_index, src_read, predecessors[src_index], memory, seeds, src_previous
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
        preflight_set_error(error, REVEAL_REPLAY_ERROR + 6);
        return false;
    }

    uint16_t expected_post[2][BLOCK_FE_WIDTH] = {};
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        expected_post[0][i] = write0_previous.value[i];
        if (crosses) expected_post[1][i] = write1_previous.value[i];
    }
#pragma unroll
    for (size_t byte = 0; byte < REVEAL_ACCESS_WIDTH; byte++) {
        size_t destination = shift + byte;
        size_t block = destination / MEMORY_BLOCK_BYTES;
        size_t within_block = destination % MEMORY_BLOCK_BYTES;
        size_t cell = within_block / U16_CELL_SIZE;
        size_t byte_in_cell = within_block % U16_CELL_SIZE;
        uint16_t mask = static_cast<uint16_t>(UINT8_MAX << (byte_in_cell * BYTE_BITS));
        expected_post[block][cell] = static_cast<uint16_t>(
            (expected_post[block][cell] & ~mask) |
            (static_cast<uint16_t>(replay_reveal_source_byte(src, byte))
             << (byte_in_cell * BYTE_BITS))
        );
    }
#pragma unroll
    for (size_t block = 0; block < 2; block++) {
        if (block == 1 && !crosses) break;
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            if (logged_post[block][i] != expected_post[block][i]) {
                preflight_set_error(error, REVEAL_REPLAY_ERROR + 7);
                return false;
            }
        }
    }

    out.from_pc = from.pc;
    out.from_timestamp = from.timestamp;
    out.src_ptr = src_ptr;
    out.base_ptr = base_ptr;
    out.base_value = base_value;
    out.base_prev_timestamp = base_previous.timestamp;
    out.src_prev_timestamp = src_previous.timestamp;
    out.write_prev_timestamps[0] = write0_previous.timestamp;
    out.write_prev_timestamps[1] = crosses ? write1_previous.timestamp : UINT32_MAX;
    out.imm = static_cast<uint16_t>(imm);
    out.imm_sign = imm_sign;
    out.shift = shift;
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        out.src_data[i] = src[i];
        out.prev_data[0][i] = write0_previous.value[i];
        out.prev_data[1][i] = crosses ? write1_previous.value[i] : 0;
    }
    return true;
}
