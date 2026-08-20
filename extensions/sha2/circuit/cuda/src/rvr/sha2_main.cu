#include "block_hasher/variant.cuh"
#include "fp.h"
#include "launcher.cuh"
#include "main/columns.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv-adapters/pointer_conv.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"
#include "rvr/replay.cuh"

using namespace riscv;
using namespace sha2;

template <typename V>
static __device__ __forceinline__ void sha2_main_replay_row_body(
    RowSlice row,
    uint32_t row_idx,
    Sha2ReplayInput const &input,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    uint32_t ptr_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);
    MemoryAuxColsFactory mem_helper(range_checker, timestamp_max_bits);

    SHA2_MAIN_WRITE_BLOCK(V, row, request_id, Fp(row_idx));
    Fp message_u16s[V::BLOCK_U16S];
    Fp prev_state_u16s[V::STATE_U16S];
    Fp new_state_u16s[V::STATE_U16S];
#pragma unroll
    for (size_t i = 0; i < V::BLOCK_U16S; i++) {
        message_u16s[i] =
            Fp(memory[input.input_start + i / BLOCK_FE_WIDTH].value[i % BLOCK_FE_WIDTH]);
    }
#pragma unroll
    for (size_t i = 0; i < V::STATE_U16S; i++) {
        prev_state_u16s[i] =
            Fp(memory[input.state_start + i / BLOCK_FE_WIDTH].value[i % BLOCK_FE_WIDTH]);
        new_state_u16s[i] =
            Fp(memory[input.write_start + i / BLOCK_FE_WIDTH].value[i % BLOCK_FE_WIDTH]);
    }
    SHA2_MAIN_WRITE_ARRAY_BLOCK(V, row, message_u16s, message_u16s);
    SHA2_MAIN_WRITE_ARRAY_BLOCK(V, row, prev_state, prev_state_u16s);
    SHA2_MAIN_WRITE_ARRAY_BLOCK(V, row, new_state, new_state_u16s);

    SHA2_MAIN_WRITE_INSTR(V, row, is_enabled, Fp::one());
    SHA2_MAIN_WRITE_INSTR(V, row, from_state.timestamp, input.timestamp);
    SHA2_MAIN_WRITE_INSTR(V, row, from_state.pc, input.from_pc);
    SHA2_MAIN_WRITE_INSTR(V, row, dst_reg_ptr, input.dst_reg_ptr);
    SHA2_MAIN_WRITE_INSTR(V, row, state_reg_ptr, input.state_reg_ptr);
    SHA2_MAIN_WRITE_INSTR(V, row, input_reg_ptr, input.input_reg_ptr);

    uint16_t dst_ptr_u16s[PTR_U16_LIMBS];
    uint16_t state_ptr_u16s[PTR_U16_LIMBS];
    uint16_t input_ptr_u16s[PTR_U16_LIMBS];
    ptr_to_u16_limbs(dst_ptr_u16s, input.dst_ptr);
    ptr_to_u16_limbs(state_ptr_u16s, input.state_ptr);
    ptr_to_u16_limbs(input_ptr_u16s, input.input_ptr);
    SHA2_MAIN_WRITE_ARRAY_INSTR(V, row, dst_ptr_limbs, dst_ptr_u16s);
    SHA2_MAIN_WRITE_ARRAY_INSTR(V, row, state_ptr_limbs, state_ptr_u16s);
    SHA2_MAIN_WRITE_ARRAY_INSTR(V, row, input_ptr_limbs, input_ptr_u16s);
    // Block-index range-check counts for each base heap pointer. Mirrors
    // `add_block_index_range_checks` in `main_chip/trace.rs`.
    add_block_index_range_checks(range_checker, input.input_ptr, ptr_max_bits);
    add_block_index_range_checks(range_checker, input.state_ptr, ptr_max_bits);
    add_block_index_range_checks(range_checker, input.dst_ptr, ptr_max_bits);

#pragma unroll
    for (size_t i = 0; i < SHA2_REGISTER_READS; i++) {
        size_t event_index = input.register_start + i;
        ReplayPreviousValue previous;
        replay_previous_value(
            event_index, memory[event_index], predecessors[event_index], memory, seeds, previous
        );
        RowSlice reg_aux = SHA2_MAIN_SLICE_MEM(V, row, register_aux[i]);
        mem_helper.fill(
            reg_aux.slice_from(COL_INDEX(MemoryReadAuxCols, base)),
            previous.timestamp,
            input.timestamp + i
        );
    }
#pragma unroll
    for (size_t i = 0; i < V::BLOCK_READS; i++) {
        size_t event_index = input.input_start + i;
        ReplayPreviousValue previous;
        replay_previous_value(
            event_index, memory[event_index], predecessors[event_index], memory, seeds, previous
        );
        RowSlice read_aux = SHA2_MAIN_SLICE_MEM(V, row, input_reads[i]);
        mem_helper.fill(
            read_aux.slice_from(COL_INDEX(MemoryReadAuxCols, base)),
            previous.timestamp,
            input.timestamp + SHA2_REGISTER_READS + i
        );
    }
#pragma unroll
    for (size_t i = 0; i < V::STATE_READS; i++) {
        size_t event_index = input.state_start + i;
        ReplayPreviousValue previous;
        replay_previous_value(
            event_index, memory[event_index], predecessors[event_index], memory, seeds, previous
        );
        RowSlice read_aux = SHA2_MAIN_SLICE_MEM(V, row, state_reads[i]);
        mem_helper.fill(
            read_aux.slice_from(COL_INDEX(MemoryReadAuxCols, base)),
            previous.timestamp,
            input.timestamp + SHA2_REGISTER_READS + V::BLOCK_READS + i
        );
    }
#pragma unroll
    for (size_t i = 0; i < V::STATE_WRITES; i++) {
        size_t event_index = input.write_start + i;
        ReplayPreviousValue previous;
        replay_previous_value(
            event_index, memory[event_index], predecessors[event_index], memory, seeds, previous
        );
        RowSlice write_aux = SHA2_MAIN_SLICE_MEM(V, row, write_aux[i]);
        Fp packed_prev[BLOCK_FE_WIDTH];
#pragma unroll
        for (size_t j = 0; j < BLOCK_FE_WIDTH; j++) {
            packed_prev[j] = Fp(previous.value[j]);
        }
        write_aux.write_array(
            COL_INDEX(MemoryWriteAuxCols, prev_data), BLOCK_FE_WIDTH, packed_prev
        );
        mem_helper.fill(
            write_aux.slice_from(COL_INDEX(MemoryWriteAuxCols, base)),
            previous.timestamp,
            input.timestamp + SHA2_REGISTER_READS + V::BLOCK_READS + V::STATE_READS + i
        );
    }
}

template <typename V>
static __device__ __noinline__ void sha2_main_replay_row_outlined(
    RowSlice row,
    uint32_t row_idx,
    Sha2ReplayInput const &input,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    uint32_t ptr_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    sha2_main_replay_row_body<V>(
        row,
        row_idx,
        input,
        memory,
        seeds,
        predecessors,
        ptr_max_bits,
        range_checker_ptr,
        range_checker_num_bins,
        timestamp_max_bits
    );
}

template <typename V>
__global__ void sha2_main_replay_tracegen(
    Fp *trace,
    size_t trace_height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t ptr_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    uint32_t *error
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= trace_height) return;
    RowSlice row(trace + row_idx, trace_height);
    row.fill_zero(0, Sha2MainLayout<V>::WIDTH);
    if (row_idx >= num_steps) return;
    if (step_start > steps.len() || row_idx >= steps.len() - step_start) {
        preflight_set_error(error, SHA2_REPLAY_ERROR);
        return;
    }
    Sha2ReplayInput input;
    if (!replay_sha2_instruction<V>(
            instructions,
            pc_base,
            program,
            memory,
            seeds,
            predecessors,
            steps[step_start + row_idx],
            expected_opcode,
            register_as,
            memory_as,
            ptr_max_bits,
            input
        )) {
        preflight_set_error(error, SHA2_REPLAY_ERROR);
        return;
    }
    if constexpr (V::WORD_BITS > 32) {
        sha2_main_replay_row_outlined<V>(
            row,
            row_idx,
            input,
            memory,
            seeds,
            predecessors,
            ptr_max_bits,
            range_checker_ptr,
            range_checker_num_bins,
            timestamp_max_bits
        );
    } else {
        sha2_main_replay_row_body<V>(
            row,
            row_idx,
            input,
            memory,
            seeds,
            predecessors,
            ptr_max_bits,
            range_checker_ptr,
            range_checker_num_bins,
            timestamp_max_bits
        );
    }
}

template <typename V>
int launch_sha2_main_replay_tracegen(
    Fp *d_trace,
    size_t trace_height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t ptr_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    auto [grid_size, block_size] = kernel_launch_params(trace_height, 256);
    sha2_main_replay_tracegen<V><<<grid_size, block_size, 0, stream>>>(
        d_trace,
        trace_height,
        instructions,
        pc_base,
        program,
        memory,
        seeds,
        predecessors,
        steps,
        step_start,
        num_steps,
        expected_opcode,
        register_as,
        memory_as,
        ptr_max_bits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits,
        error
    );
    return CHECK_KERNEL();
}

extern "C" {
int launch_sha256_main_replay_tracegen(
    Fp *d_trace,
    size_t trace_height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t ptr_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    return launch_sha2_main_replay_tracegen<Sha256Variant>(
        d_trace, trace_height, instructions, pc_base, program, memory, seeds, predecessors, steps,
        step_start, num_steps, expected_opcode, register_as, memory_as, ptr_max_bits,
        d_range_checker, range_checker_num_bins, timestamp_max_bits, error, stream
    );
}

int launch_sha512_main_replay_tracegen(
    Fp *d_trace,
    size_t trace_height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t ptr_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    return launch_sha2_main_replay_tracegen<Sha512Variant>(
        d_trace, trace_height, instructions, pc_base, program, memory, seeds, predecessors, steps,
        step_start, num_steps, expected_opcode, register_as, memory_as, ptr_max_bits,
        d_range_checker, range_checker_num_bins, timestamp_max_bits, error, stream
    );
}
}
