#pragma once

#include "primitives/constants.h"
#include "arch/rvr/replay.cuh"
#include "checkpoint_replay_opcodes.cuh"

using namespace program;
using namespace riscv;

struct ReplayStoreByteInput {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rs1_ptr;
    uint32_t rs2_ptr;
    uint32_t rs1_val;
    uint32_t rs1_prev_timestamp;
    uint32_t rs2_prev_timestamp;
    uint32_t write_prev_timestamp;
    uint16_t imm;
    uint8_t imm_sign;
    uint8_t shift;
    uint16_t read_data[BLOCK_FE_WIDTH];
    uint16_t prev_data[BLOCK_FE_WIDTH];
};

static __device__ bool replay_store_byte(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    RvrReplayStep const &step,
    uint32_t expected_opcode,
    uint32_t register_as,
    size_t pointer_max_bits,
    ReplayStoreByteInput &out,
    uint32_t *error
) {
    ReplayProgramTransition transition;
    if (!replay_program_transition(
            instructions,
            pc_base,
            program,
            step.program_index,
            3u,
            ReplayPcEffect::Sequential,
            transition,
            error,
            251
        )) {
        return false;
    }
    auto const &from = *transition.from;
    auto const &to = *transition.to;
    auto const &instruction = *transition.instruction;
    uint32_t rs2_ptr = instruction.words[1];
    uint32_t rs1_ptr = instruction.words[2];
    uint32_t imm = instruction.words[3];
    uint32_t is_valid = instruction.words[6];
    uint32_t imm_sign = instruction.words[7];
    if (instruction.words[0] != expected_opcode || instruction.words[4] != register_as ||
        instruction.words[5] != MEMORY_ADDRESS_SPACE || imm > UINT16_MAX ||
        is_valid != 1 || imm_sign > 1 || !replay_canonical_register_pointer(rs1_ptr) ||
        !replay_canonical_register_pointer(rs2_ptr)) {
        preflight_set_error(error, 254);
        return false;
    }

    size_t rs1_index = step.memory_start;
    size_t rs2_index = rs1_index + 1;
    size_t write_index = rs2_index + 1;
    size_t next_index = write_index + 1;
    if (write_index >= memory.len() || write_index >= predecessors.len()) {
        preflight_set_error(error, 255);
        return false;
    }
    auto const &rs1_read = memory[rs1_index];
    auto const &rs2_read = memory[rs2_index];
    auto const &write = memory[write_index];
    if (rs1_read.timestamp != from.timestamp || preflight_is_write(rs1_read) ||
        preflight_address_space(rs1_read) != register_as || rs1_read.pointer != rs1_ptr / 2 ||
        rs2_read.timestamp != from.timestamp + 1 || preflight_is_write(rs2_read) ||
        preflight_address_space(rs2_read) != register_as || rs2_read.pointer != rs2_ptr / 2 ||
        write.timestamp != from.timestamp + 2 || !preflight_is_write(write) ||
        preflight_address_space(write) != MEMORY_ADDRESS_SPACE ||
        (next_index < memory.len() && memory[next_index].timestamp < to.timestamp)) {
        preflight_set_error(error, 255);
        return false;
    }

    uint16_t rs1[BLOCK_FE_WIDTH];
    uint16_t rs2[BLOCK_FE_WIDTH];
    uint16_t logged_post[BLOCK_FE_WIDTH];
    replay_u16_block(rs1_read.value, rs1);
    replay_u16_block(rs2_read.value, rs2);
    replay_u16_block(write.value, logged_post);
    if (rs1[2] != 0 || rs1[3] != 0) {
        preflight_set_error(error, 256);
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
        preflight_set_error(error, 257);
        return false;
    }
    uint32_t ptr = static_cast<uint32_t>(effective);
    uint32_t aligned_ptr = ptr & ~(uint32_t(MEMORY_BLOCK_BYTES) - 1);
    if (write.pointer != aligned_ptr / U16_CELL_SIZE) {
        preflight_set_error(error, 258);
        return false;
    }

    ReplayPreviousValue rs1_previous;
    ReplayPreviousValue rs2_previous;
    ReplayPreviousValue write_previous;
    if (!replay_previous_value(
            rs1_index, rs1_read, predecessors[rs1_index], memory, seeds, rs1_previous
        ) ||
        !replay_previous_value(
            rs2_index, rs2_read, predecessors[rs2_index], memory, seeds, rs2_previous
        ) ||
        !replay_previous_value(
            write_index, write, predecessors[write_index], memory, seeds, write_previous
        )) {
        preflight_set_error(error, 259);
        return false;
    }

    uint8_t shift = ptr - aligned_ptr;
    uint16_t expected_post[BLOCK_FE_WIDTH];
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        expected_post[i] = write_previous.value[i];
    }
    uint32_t cell = shift >> 1;
    uint32_t byte = shift & 1;
    uint16_t mask = static_cast<uint16_t>(UINT8_MAX << (byte * BYTE_BITS));
    expected_post[cell] = static_cast<uint16_t>(
        (expected_post[cell] & ~mask) |
        ((rs2[0] & UINT8_MAX) << (byte * BYTE_BITS))
    );
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        if (logged_post[i] != expected_post[i]) {
            preflight_set_error(error, 260);
            return false;
        }
    }

    out.from_pc = from.pc;
    out.from_timestamp = from.timestamp;
    out.rs1_ptr = rs1_ptr;
    out.rs2_ptr = rs2_ptr;
    out.rs1_val = rs1_val;
    out.rs1_prev_timestamp = rs1_previous.timestamp;
    out.rs2_prev_timestamp = rs2_previous.timestamp;
    out.write_prev_timestamp = write_previous.timestamp;
    out.imm = static_cast<uint16_t>(imm);
    out.imm_sign = imm_sign;
    out.shift = shift;
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        out.read_data[i] = rs2[i];
        out.prev_data[i] = write_previous.value[i];
    }
    return true;
}
