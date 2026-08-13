#include "fp.h"
#include "keccakf_op.cuh"
#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv-adapters/pointer_conv.cuh"
#include "system/memory/controller.cuh"
#include "arch/rvr/replay.cuh"

#include <cassert>
#include <cstddef>
#include <cstdint>

using namespace keccak256;
using namespace keccakf_op;
using namespace program;
using namespace riscv;
using openvm::U16_BITS;

static constexpr uint32_t KECCAKF_REPLAY_ERROR = 811;

#define KECCAKF_OP_WRITE(FIELD, VALUE) COL_WRITE_VALUE(row, KeccakfOpCols, FIELD, VALUE)
#define KECCAKF_OP_WRITE_ARRAY(FIELD, VALUES) COL_WRITE_ARRAY(row, KeccakfOpCols, FIELD, VALUES)
#define KECCAKF_OP_FILL_ZERO(FIELD) COL_FILL_ZERO(row, KeccakfOpCols, FIELD)
#define KECCAKF_OP_SLICE(FIELD) row.slice_from(COL_INDEX(KeccakfOpCols, FIELD))

static __device__ uint64_t keccakf_replay_u64(uint16_t const (&cells)[BLOCK_FE_WIDTH]) {
    uint64_t value = 0;
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        value |= static_cast<uint64_t>(cells[i]) << (i * 16);
    }
    return value;
}

__global__ void keccakf_op_replay_tracegen(
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
    uint64_t *preimages,
    size_t preimage_words,
    uint32_t *error,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(d_trace + idx, height);
    row.fill_zero(0, sizeof(KeccakfOpCols<uint8_t>));
    if (idx >= num_steps) return;
    if (preimage_words < num_steps * KECCAK_WIDTH_MEM_OPS) {
        preflight_set_error(error, KECCAKF_REPLAY_ERROR);
        return;
    }

    auto const &step = steps[step_start + idx];
    ReplayProgramTransition transition;
    if (resolve_replay_program_transition(
            instructions,
            pc_base,
            program,
            step.program_index,
            26,
            ReplayPcEffect::Sequential,
            transition
        ) != ReplayProgramTransitionError::None) {
        preflight_set_error(error, KECCAKF_REPLAY_ERROR);
        return;
    }
    auto const &from = *transition.from;
    auto const &to = *transition.to;
    auto const &instruction = *transition.instruction;
    uint32_t rd_ptr = instruction.words[1];
    if (instruction.words[0] != expected_opcode || instruction.words[4] != register_as ||
        instruction.words[5] != memory_as || (rd_ptr & 1) != 0) {
        preflight_set_error(error, KECCAKF_REPLAY_ERROR);
        return;
    }

    size_t rd_idx = step.memory_start;
    if (rd_idx >= memory.len() || rd_idx >= predecessors.len()) {
        preflight_set_error(error, KECCAKF_REPLAY_ERROR);
        return;
    }
    auto const &rd = memory[rd_idx];
    ReplayPreviousValue rd_previous;
    if (rd.timestamp != from.timestamp || preflight_is_write(rd) ||
        preflight_address_space(rd) != register_as || rd.pointer != rd_ptr / 2 ||
        rd.value[2] != 0 || rd.value[3] != 0 ||
        !replay_previous_value(rd_idx, rd, predecessors[rd_idx], memory, seeds, rd_previous)) {
        preflight_set_error(error, KECCAKF_REPLAY_ERROR);
        return;
    }
    uint32_t buffer_ptr = static_cast<uint32_t>(rd.value[0]) |
                          (static_cast<uint32_t>(rd.value[1]) << U16_BITS);
    uint64_t domain_end = pointer_max_bits < 32 ? (uint64_t(1) << pointer_max_bits)
                                                : (uint64_t(1) << 32);
    if ((buffer_ptr & 1) != 0 || static_cast<uint64_t>(buffer_ptr) + KECCAK_WIDTH_BYTES > domain_end) {
        preflight_set_error(error, KECCAKF_REPLAY_ERROR);
        return;
    }

    uint32_t write_previous_timestamps[KECCAK_WIDTH_MEM_OPS];
    uint64_t state[KECCAK_WIDTH_MEM_OPS];
    for (uint32_t i = 0; i < KECCAK_WIDTH_MEM_OPS; i++) {
        size_t write_idx = rd_idx + 1 + i;
        if (write_idx >= memory.len() || write_idx >= predecessors.len()) {
            preflight_set_error(error, KECCAKF_REPLAY_ERROR);
            return;
        }
        auto const &write = memory[write_idx];
        ReplayPreviousValue previous;
        if (write.timestamp != from.timestamp + 1 + i || !preflight_is_write(write) ||
            preflight_address_space(write) != memory_as ||
            write.pointer != buffer_ptr / 2 + i * BLOCK_FE_WIDTH ||
            !replay_previous_value(
                write_idx,
                write,
                predecessors[write_idx],
                memory,
                seeds,
                previous
            )) {
            preflight_set_error(error, KECCAKF_REPLAY_ERROR);
            return;
        }
        write_previous_timestamps[i] = previous.timestamp;
        state[i] = keccakf_replay_u64(previous.value);
        preimages[idx * KECCAK_WIDTH_MEM_OPS + i] = state[i];
    }
    size_t event_end = rd_idx + 1 + KECCAK_WIDTH_MEM_OPS;
    if (event_end < memory.len() && memory[event_end].timestamp < to.timestamp) {
        preflight_set_error(error, KECCAKF_REPLAY_ERROR);
        return;
    }

    uint64_t postimage[KECCAK_WIDTH_MEM_OPS];
#pragma unroll
    for (size_t i = 0; i < KECCAK_WIDTH_MEM_OPS; i++) postimage[i] = state[i];
    keccakf_permutation(postimage);
    for (uint32_t i = 0; i < KECCAK_WIDTH_MEM_OPS; i++) {
        if (keccakf_replay_u64(memory[rd_idx + 1 + i].value) != postimage[i]) {
            preflight_set_error(error, KECCAKF_REPLAY_ERROR);
            return;
        }
    }

    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);
    MemoryAuxColsFactory mem_helper(range_checker, timestamp_max_bits);
    KECCAKF_OP_WRITE(pc[0], ::program::pc_lo(from.pc));
    KECCAKF_OP_WRITE(pc[1], ::program::pc_hi(from.pc));
    KECCAKF_OP_WRITE(is_valid, 1);
    KECCAKF_OP_WRITE(timestamp, from.timestamp);
    KECCAKF_OP_WRITE(rd_ptr, rd_ptr);
    uint16_t buffer_ptr_limbs[PTR_U16_LIMBS];
    ptr_to_u16_limbs(buffer_ptr_limbs, buffer_ptr);
    KECCAKF_OP_WRITE_ARRAY(buffer_ptr_limbs, buffer_ptr_limbs);
    KECCAKF_OP_WRITE_ARRAY(preimage, reinterpret_cast<uint16_t const *>(state));
    KECCAKF_OP_WRITE_ARRAY(postimage, reinterpret_cast<uint16_t const *>(postimage));
    mem_helper.fill(KECCAKF_OP_SLICE(rd_aux.base), rd_previous.timestamp, from.timestamp);
    for (uint32_t i = 0; i < KECCAK_WIDTH_MEM_OPS; i++) {
        mem_helper.fill(
            KECCAKF_OP_SLICE(buffer_word_aux[i]),
            write_previous_timestamps[i],
            from.timestamp + 1 + i
        );
    }
    // Byte -> cell pointer conversion carry and per-block cell-offset carries, plus the matching
    // range-check counts (mirrors KeccakfOpChip::fill_trace_inputs).
    uint32_t cell_stride = MEMORY_BLOCK_BYTES / U16_CELL_SIZE;
    uint32_t add_carries[KECCAK_WIDTH_MEM_OPS];
    uint32_t conv_carry = compute_pointer_carries(
        range_checker,
        buffer_ptr,
        pointer_max_bits,
        KECCAK_WIDTH_MEM_OPS,
        cell_stride,
        add_carries
    );
    KECCAKF_OP_WRITE(buffer_cell_carry, conv_carry);
    KECCAKF_OP_WRITE_ARRAY(buffer_word_add_carry, add_carries);
}

extern "C" int _keccakf_op_replay_tracegen(
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
    uint64_t *d_preimages,
    size_t preimage_words,
    uint32_t *d_error,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(KeccakfOpCols<uint8_t>));
    assert(d_memory.len() == d_predecessors.len());
    assert(step_start <= d_steps.len() && num_steps <= d_steps.len() - step_start);
    assert(height >= num_steps);
    assert(preimage_words >= num_steps * KECCAK_WIDTH_MEM_OPS);
    auto [grid, block] = kernel_launch_params(height, 128);
    keccakf_op_replay_tracegen<<<grid, block, 0, stream>>>(
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
        d_preimages,
        preimage_words,
        d_error,
        expected_opcode,
        register_as,
        memory_as,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
