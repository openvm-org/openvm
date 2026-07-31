#include "launcher.cuh"
#include "riscv-adapters/vec_heap_replay.cuh"

__global__ void algebra_merge_range_counts(
    uint32_t *destination, uint32_t const *source, size_t len
) {
    size_t index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (index < len) destination[index] += source[index];
}

extern "C" int _algebra_merge_range_counts(
    uint32_t *destination, uint32_t const *source, size_t len, cudaStream_t stream
) {
    if (len == 0) return 0;
    auto [grid, block] = kernel_launch_params(len, 256);
    algebra_merge_range_counts<<<grid, block, 0, stream>>>(destination, source, len);
    return CHECK_KERNEL();
}

template <size_t NUM_READS, size_t BLOCKS>
static int launch_vec_heap_replay_gather(
    void *output,
    size_t output_len,
    size_t output_start,
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
    uint32_t local_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    static_assert(
        sizeof(VecHeapTraceInput<NUM_READS, BLOCKS>) ==
        VEC_HEAP_TRACE_INPUT_BYTES<NUM_READS, BLOCKS>
    );
    if (output_start > output_len || num_steps > output_len - output_start) return 1;
    if (num_steps == 0) return 0;
    auto [grid, block] = kernel_launch_params(num_steps, 256);
    vec_heap_replay_gather<NUM_READS, BLOCKS><<<grid, block, 0, stream>>>(
        static_cast<VecHeapTraceInput<NUM_READS, BLOCKS> *>(output),
        output_start,
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
        local_opcode,
        register_as,
        memory_as,
        pointer_max_bits,
        error
    );
    return CHECK_KERNEL();
}

extern "C" int _vec_heap_replay_gather(
    void *output,
    size_t output_len,
    size_t output_start,
    size_t num_reads,
    size_t blocks,
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
    uint32_t local_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    if (num_reads == 2 && blocks == 4) {
        return launch_vec_heap_replay_gather<2, 4>(
            output, output_len, output_start, instructions, pc_base, program, memory, seeds,
            predecessors, steps, step_start, num_steps, expected_opcode, local_opcode,
            register_as, memory_as, pointer_max_bits, error, stream
        );
    }
    if (num_reads == 2 && blocks == 6) {
        return launch_vec_heap_replay_gather<2, 6>(
            output, output_len, output_start, instructions, pc_base, program, memory, seeds,
            predecessors, steps, step_start, num_steps, expected_opcode, local_opcode,
            register_as, memory_as, pointer_max_bits, error, stream
        );
    }
    if (num_reads == 2 && blocks == 8) {
        return launch_vec_heap_replay_gather<2, 8>(
            output, output_len, output_start, instructions, pc_base, program, memory, seeds,
            predecessors, steps, step_start, num_steps, expected_opcode, local_opcode,
            register_as, memory_as, pointer_max_bits, error, stream
        );
    }
    if (num_reads == 2 && blocks == 12) {
        return launch_vec_heap_replay_gather<2, 12>(
            output, output_len, output_start, instructions, pc_base, program, memory, seeds,
            predecessors, steps, step_start, num_steps, expected_opcode, local_opcode,
            register_as, memory_as, pointer_max_bits, error, stream
        );
    }
    if (num_reads == 1 && blocks == 8) {
        return launch_vec_heap_replay_gather<1, 8>(
            output, output_len, output_start, instructions, pc_base, program, memory, seeds,
            predecessors, steps, step_start, num_steps, expected_opcode, local_opcode,
            register_as, memory_as, pointer_max_bits, error, stream
        );
    }
    if (num_reads == 1 && blocks == 12) {
        return launch_vec_heap_replay_gather<1, 12>(
            output, output_len, output_start, instructions, pc_base, program, memory, seeds,
            predecessors, steps, step_start, num_steps, expected_opcode, local_opcode,
            register_as, memory_as, pointer_max_bits, error, stream
        );
    }
    return 1;
}
