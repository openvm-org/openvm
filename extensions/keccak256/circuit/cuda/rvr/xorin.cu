#include "fp.h"
#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv-adapters/pointer_conv.cuh"
#include "system/memory/controller.cuh"
#include "arch/rvr/replay.cuh"
#include "xorin.cuh"

#include <cassert>
#include <cstddef>
#include <cstdint>

using namespace xorin;
using namespace riscv;
using namespace keccak256;
using namespace program;
using openvm::U16_BITS;

static constexpr uint32_t XORIN_REPLAY_ERROR = 801;

#define XORIN_WRITE(FIELD, VALUE) COL_WRITE_VALUE(row, XorinVmCols, FIELD, VALUE)
#define XORIN_WRITE_ARRAY(FIELD, VALUES) COL_WRITE_ARRAY(row, XorinVmCols, FIELD, VALUES)
#define XORIN_SLICE(FIELD) row.slice_from(COL_INDEX(XorinVmCols, FIELD))

static __device__ bool xorin_replay_event(
    size_t event_idx,
    uint32_t timestamp,
    uint32_t address_space,
    uint32_t pointer,
    bool is_write,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    uint32_t &previous_timestamp,
    uint32_t *error
) {
    if (event_idx >= memory.len() || event_idx >= predecessors.len()) {
        preflight_set_error(error, XORIN_REPLAY_ERROR);
        return false;
    }
    auto const &event = memory[event_idx];
    ReplayPreviousValue previous;
    if (event.timestamp != timestamp || preflight_address_space(event) != address_space ||
        event.pointer != pointer || preflight_is_write(event) != is_write ||
        !replay_previous_value(
            event_idx, event, predecessors[event_idx], memory, seeds, previous
        )) {
        preflight_set_error(error, XORIN_REPLAY_ERROR);
        return false;
    }
    previous_timestamp = previous.timestamp;
    return true;
}

static __device__ void xorin_replay_bytes(
    uint16_t const (&cells)[BLOCK_FE_WIDTH], uint8_t *bytes
) {
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        bytes[2 * i] = static_cast<uint8_t>(cells[i]);
        bytes[2 * i + 1] = static_cast<uint8_t>(cells[i] >> 8);
    }
}

__global__ void xorin_replay_tracegen(
    Fp *d_trace,
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
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;

    RowSlice row(d_trace + idx, height);
    row.fill_zero(0, sizeof(XorinVmCols<uint8_t>));
    if (idx >= num_steps) return;

    auto const &step = steps[step_start + idx];
    size_t program_idx = step.program_index;
    if (program_idx + 1 >= program.len()) {
        preflight_set_error(error, XORIN_REPLAY_ERROR);
        return;
    }
    auto const &from = program[program_idx];
    auto const &to = program[program_idx + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % DEFAULT_PC_STEP != 0 ||
        from.pc > UINT32_MAX - DEFAULT_PC_STEP) {
        preflight_set_error(error, XORIN_REPLAY_ERROR);
        return;
    }
    size_t instruction_idx = (from.pc - pc_base) / DEFAULT_PC_STEP;
    if (instruction_idx >= instructions.len()) {
        preflight_set_error(error, XORIN_REPLAY_ERROR);
        return;
    }
    auto const &instruction = instructions[instruction_idx];
    uint32_t buffer_reg_ptr = instruction.words[1];
    uint32_t input_reg_ptr = instruction.words[2];
    uint32_t len_reg_ptr = instruction.words[3];
    if (instruction.words[0] != expected_opcode || instruction.words[4] != register_as ||
        instruction.words[5] != memory_as || (buffer_reg_ptr & 1) != 0 ||
        (input_reg_ptr & 1) != 0 || (len_reg_ptr & 1) != 0) {
        preflight_set_error(error, XORIN_REPLAY_ERROR);
        return;
    }

    size_t event_idx = step.memory_start;
    uint32_t register_previous_timestamps[XORIN_REGISTER_READS];
    uint32_t register_ptrs[XORIN_REGISTER_READS] = {
        buffer_reg_ptr, input_reg_ptr, len_reg_ptr
    };
    uint32_t register_values[XORIN_REGISTER_READS];
#pragma unroll
    for (size_t i = 0; i < XORIN_REGISTER_READS; i++) {
        if (!xorin_replay_event(
                event_idx + i,
                from.timestamp + static_cast<uint32_t>(i),
                register_as,
                register_ptrs[i] / 2,
                false,
                memory,
                seeds,
                predecessors,
                register_previous_timestamps[i],
                error
            )) {
            return;
        }
        auto const &event = memory[event_idx + i];
        if (event.value[2] != 0 || event.value[3] != 0) {
            preflight_set_error(error, XORIN_REPLAY_ERROR);
            return;
        }
        register_values[i] = static_cast<uint32_t>(event.value[0]) |
                             (static_cast<uint32_t>(event.value[1]) << U16_BITS);
    }
    event_idx += XORIN_REGISTER_READS;

    uint32_t buffer_ptr = register_values[0];
    uint32_t input_ptr = register_values[1];
    uint32_t len = register_values[2];
    uint32_t num_blocks = len / DEFAULT_BLOCK_SIZE;
    uint64_t domain_end = pointer_max_bits < 32 ? (uint64_t(1) << pointer_max_bits)
                                                : (uint64_t(1) << 32);
    // The AIR converts the base byte pointers to cell pointers on every enabled row (padding
    // included), so 2-byte alignment is required even for zero-length XORIN. Mirrors the host
    // replay validation.
    if (len > XORIN_RATE_BYTES || len % DEFAULT_BLOCK_SIZE != 0 ||
        (buffer_ptr & 1) != 0 || (input_ptr & 1) != 0 ||
        buffer_ptr >= domain_end || input_ptr >= domain_end ||
        static_cast<uint64_t>(buffer_ptr) + len > domain_end ||
        static_cast<uint64_t>(input_ptr) + len > domain_end ||
        from.timestamp > UINT32_MAX - (XORIN_REGISTER_READS + 3 * num_blocks) ||
        to.timestamp != from.timestamp + XORIN_REGISTER_READS + 3 * num_blocks ||
        to.pc != from.pc + DEFAULT_PC_STEP) {
        preflight_set_error(error, XORIN_REPLAY_ERROR);
        return;
    }

    uint8_t buffer_bytes[XORIN_RATE_BYTES] = {};
    uint8_t input_bytes[XORIN_RATE_BYTES] = {};
    uint32_t buffer_read_previous_timestamps[keccak256::KECCAK_RATE_MEM_OPS];
    uint32_t input_read_previous_timestamps[keccak256::KECCAK_RATE_MEM_OPS];
    uint32_t buffer_write_previous_timestamps[keccak256::KECCAK_RATE_MEM_OPS];

    for (uint32_t i = 0; i < num_blocks; i++) {
        if (!xorin_replay_event(
                event_idx + i,
                from.timestamp + XORIN_REGISTER_READS + i,
                memory_as,
                buffer_ptr / 2 + i * BLOCK_FE_WIDTH,
                false,
                memory,
                seeds,
                predecessors,
                buffer_read_previous_timestamps[i],
                error
            )) {
            return;
        }
        xorin_replay_bytes(memory[event_idx + i].value, buffer_bytes + i * DEFAULT_BLOCK_SIZE);
    }
    event_idx += num_blocks;
    for (uint32_t i = 0; i < num_blocks; i++) {
        if (!xorin_replay_event(
                event_idx + i,
                from.timestamp + XORIN_REGISTER_READS + num_blocks + i,
                memory_as,
                input_ptr / 2 + i * BLOCK_FE_WIDTH,
                false,
                memory,
                seeds,
                predecessors,
                input_read_previous_timestamps[i],
                error
            )) {
            return;
        }
        xorin_replay_bytes(memory[event_idx + i].value, input_bytes + i * DEFAULT_BLOCK_SIZE);
    }
    event_idx += num_blocks;
    for (uint32_t i = 0; i < num_blocks; i++) {
        if (!xorin_replay_event(
                event_idx + i,
                from.timestamp + XORIN_REGISTER_READS + 2 * num_blocks + i,
                memory_as,
                buffer_ptr / 2 + i * BLOCK_FE_WIDTH,
                true,
                memory,
                seeds,
                predecessors,
                buffer_write_previous_timestamps[i],
                error
            )) {
            return;
        }
        uint8_t written[DEFAULT_BLOCK_SIZE];
        xorin_replay_bytes(memory[event_idx + i].value, written);
#pragma unroll
        for (size_t j = 0; j < DEFAULT_BLOCK_SIZE; j++) {
            size_t byte_idx = i * DEFAULT_BLOCK_SIZE + j;
            if (written[j] != static_cast<uint8_t>(buffer_bytes[byte_idx] ^ input_bytes[byte_idx])) {
                preflight_set_error(error, XORIN_REPLAY_ERROR);
                return;
            }
        }
    }
    event_idx += num_blocks;
    if (event_idx < memory.len() && memory[event_idx].timestamp < to.timestamp) {
        preflight_set_error(error, XORIN_REPLAY_ERROR);
        return;
    }

    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);
    MemoryAuxColsFactory mem_helper(range_checker, timestamp_max_bits);
    BitwiseOperationLookup bitwise_lookup(bitwise_lookup_ptr);

    XORIN_WRITE(instruction.pc, from.pc);
    XORIN_WRITE(instruction.is_enabled, 1);
    XORIN_WRITE(instruction.buffer_reg_ptr, buffer_reg_ptr);
    XORIN_WRITE(instruction.input_reg_ptr, input_reg_ptr);
    XORIN_WRITE(instruction.len_reg_ptr, len_reg_ptr);
    XORIN_WRITE(instruction.start_timestamp, from.timestamp);
    uint16_t buffer_ptr_limbs[PTR_U16_LIMBS];
    uint16_t input_ptr_limbs[PTR_U16_LIMBS];
    ptr_to_u16_limbs(buffer_ptr_limbs, buffer_ptr);
    ptr_to_u16_limbs(input_ptr_limbs, input_ptr);
    XORIN_WRITE_ARRAY(instruction.buffer_ptr_limbs, buffer_ptr_limbs);
    XORIN_WRITE_ARRAY(instruction.input_ptr_limbs, input_ptr_limbs);

    for (uint32_t i = 0; i < keccak256::KECCAK_RATE_MEM_OPS; i++) {
        XORIN_WRITE(sponge.is_padding_bytes[i], i >= num_blocks);
    }
    for (uint32_t i = 0; i < len; i++) {
        XORIN_WRITE(sponge.preimage_buffer_bytes[i], buffer_bytes[i]);
        XORIN_WRITE(sponge.input_bytes[i], input_bytes[i]);
        XORIN_WRITE(sponge.postimage_buffer_bytes[i], buffer_bytes[i] ^ input_bytes[i]);
        bitwise_lookup.add_xor(buffer_bytes[i], input_bytes[i]);
    }

#pragma unroll
    for (size_t i = 0; i < XORIN_REGISTER_READS; i++) {
        mem_helper.fill(
            XORIN_SLICE(mem_oc.register_aux_cols[i].base),
            register_previous_timestamps[i],
            from.timestamp + static_cast<uint32_t>(i)
        );
    }
    for (uint32_t i = 0; i < num_blocks; i++) {
        mem_helper.fill(
            XORIN_SLICE(mem_oc.buffer_bytes_read_aux_cols[i].base),
            buffer_read_previous_timestamps[i],
            from.timestamp + XORIN_REGISTER_READS + i
        );
        mem_helper.fill(
            XORIN_SLICE(mem_oc.input_bytes_read_aux_cols[i].base),
            input_read_previous_timestamps[i],
            from.timestamp + XORIN_REGISTER_READS + num_blocks + i
        );
        mem_helper.fill(
            XORIN_SLICE(mem_oc.buffer_bytes_write_base_aux[i]),
            buffer_write_previous_timestamps[i],
            from.timestamp + XORIN_REGISTER_READS + 2 * num_blocks + i
        );
    }
    // Byte -> cell pointer conversion carries, plus matching range-check counts. Mirrors
    // `xorin/trace.rs`.
    XORIN_WRITE(
        mem_oc.buffer_cell_carry, compute_pointer_carry(range_checker, buffer_ptr, pointer_max_bits)
    );
    XORIN_WRITE(
        mem_oc.input_cell_carry, compute_pointer_carry(range_checker, input_ptr, pointer_max_bits)
    );
}

extern "C" int _xorin_replay_tracegen(
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
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(XorinVmCols<uint8_t>));
    assert(d_memory.len() == d_predecessors.len());
    assert(step_start <= d_steps.len() && num_steps <= d_steps.len() - step_start);
    assert(height >= num_steps);
    auto [grid, block] = kernel_launch_params(height, 256);
    xorin_replay_tracegen<<<grid, block, 0, stream>>>(
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
        expected_opcode,
        register_as,
        memory_as,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
