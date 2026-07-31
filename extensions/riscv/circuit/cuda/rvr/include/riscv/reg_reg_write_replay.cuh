#pragma once

#include "riscv/adapters/mul.cuh"
#include "arch/rvr/replay.cuh"

struct Rv64RegRegWriteReplay {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint32_t rs2_ptr;
    uint8_t rs1[8];
    uint8_t rs2[8];
    uint8_t result[8];
    uint8_t previous_result[8];
    uint32_t rs1_previous_timestamp;
    uint32_t rs2_previous_timestamp;
    uint32_t result_previous_timestamp;
};

static __device__ __forceinline__ void replay_u16_to_bytes(
    uint16_t const (&source)[BLOCK_FE_WIDTH], uint8_t (&out)[8]
) {
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        out[2 * i] = static_cast<uint8_t>(source[i]);
        out[2 * i + 1] = static_cast<uint8_t>(source[i] >> 8);
    }
}

static __device__ bool replay_reg_reg_write(
    ReplayProgramTransition const &transition,
    uint32_t expected_opcode,
    uint32_t register_address_space,
    RvrReplayStep const &step,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    Rv64RegRegWriteReplay &out,
    uint32_t *error,
    uint32_t error_base
) {
    auto const &instruction = *transition.instruction;
    auto const &from = *transition.from;
    auto const &to = *transition.to;
    uint32_t rd_ptr = instruction.words[1];
    uint32_t rs1_ptr = instruction.words[2];
    uint32_t rs2_ptr = instruction.words[3];
    if (instruction.words[0] != expected_opcode ||
        instruction.words[4] != register_address_space || instruction.words[5] != 0 ||
        instruction.words[6] != 0 || instruction.words[7] != 0 || rd_ptr == 0 ||
        !replay_canonical_register_pointer(rd_ptr) ||
        !replay_canonical_register_pointer(rs1_ptr) ||
        !replay_canonical_register_pointer(rs2_ptr)) {
        preflight_set_error(error, error_base);
        return false;
    }

    size_t rs1_index = step.memory_start;
    size_t rs2_index = rs1_index + 1;
    size_t write_index = rs1_index + 2;
    if (write_index >= memory.len() || write_index >= predecessors.len()) {
        preflight_set_error(error, error_base + 1);
        return false;
    }
    auto const &rs1 = memory[rs1_index];
    auto const &rs2 = memory[rs2_index];
    auto const &write = memory[write_index];
    if (rs1.timestamp != from.timestamp || preflight_is_write(rs1) ||
        preflight_address_space(rs1) != register_address_space || rs1.pointer != rs1_ptr / 2 ||
        rs2.timestamp != from.timestamp + 1 || preflight_is_write(rs2) ||
        preflight_address_space(rs2) != register_address_space || rs2.pointer != rs2_ptr / 2 ||
        write.timestamp != from.timestamp + 2 || !preflight_is_write(write) ||
        preflight_address_space(write) != register_address_space || write.pointer != rd_ptr / 2 ||
        (write_index + 1 < memory.len() && memory[write_index + 1].timestamp < to.timestamp)) {
        preflight_set_error(error, error_base + 2);
        return false;
    }

    uint16_t rs1_u16[BLOCK_FE_WIDTH];
    uint16_t rs2_u16[BLOCK_FE_WIDTH];
    uint16_t result_u16[BLOCK_FE_WIDTH];
    replay_u16_block(rs1.value, rs1_u16);
    replay_u16_block(rs2.value, rs2_u16);
    replay_u16_block(write.value, result_u16);

    ReplayPreviousValue rs1_previous;
    ReplayPreviousValue rs2_previous;
    ReplayPreviousValue result_previous;
    if (!replay_previous_value(
            rs1_index, rs1, predecessors[rs1_index], memory, seeds, rs1_previous
        ) ||
        !replay_previous_value(
            rs2_index, rs2, predecessors[rs2_index], memory, seeds, rs2_previous
        ) ||
        !replay_previous_value(
            write_index, write, predecessors[write_index], memory, seeds, result_previous
        )) {
        preflight_set_error(error, error_base + 4);
        return false;
    }

    out.from_pc = from.pc;
    out.from_timestamp = from.timestamp;
    out.rd_ptr = rd_ptr;
    out.rs1_ptr = rs1_ptr;
    out.rs2_ptr = rs2_ptr;
    replay_u16_to_bytes(rs1_u16, out.rs1);
    replay_u16_to_bytes(rs2_u16, out.rs2);
    replay_u16_to_bytes(result_u16, out.result);
    replay_u16_to_bytes(result_previous.value, out.previous_result);
    out.rs1_previous_timestamp = rs1_previous.timestamp;
    out.rs2_previous_timestamp = rs2_previous.timestamp;
    out.result_previous_timestamp = result_previous.timestamp;
    return true;
}

static __device__ __forceinline__ Rv64MultAdapterRecord replay_mult_adapter_record(
    Rv64RegRegWriteReplay const &replay
) {
    Rv64MultAdapterRecord record{};
    record.from_pc = replay.from_pc;
    record.from_timestamp = replay.from_timestamp;
    record.rd_ptr = replay.rd_ptr;
    record.rs1_ptr = replay.rs1_ptr;
    record.rs2_ptr = replay.rs2_ptr;
    record.reads_aux[0].prev_timestamp = replay.rs1_previous_timestamp;
    record.reads_aux[1].prev_timestamp = replay.rs2_previous_timestamp;
    record.writes_aux.prev_timestamp = replay.result_previous_timestamp;
#pragma unroll
    for (size_t i = 0; i < 8; i++) record.writes_aux.prev_data[i] = replay.previous_result[i];
    return record;
}
