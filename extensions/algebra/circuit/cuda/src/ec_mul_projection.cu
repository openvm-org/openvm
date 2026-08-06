#include "launcher.cuh"
#include "riscv-adapters/ec_mul_replay.cuh"

// The `EC_MUL` gather lives alongside the other replay launchers because this crate is the one that
// compiles CUDA for the extension stack; the ECC circuit crate has no CUDA build of its own. The
// kernel itself is in `riscv-adapters`, next to the vec-heap gather whose event helper it reuses.

template <size_t BLOCKS>
static int launch_ec_mul_replay_gather(
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
    uint32_t is_setup,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    static_assert(sizeof(EcMulTraceInput<BLOCKS>) == EC_MUL_TRACE_INPUT_BYTES<BLOCKS>);
    if (output_start > output_len || num_steps > output_len - output_start) return 1;
    if (num_steps == 0) return 0;
    auto [grid, block] = kernel_launch_params(num_steps, 256);
    ec_mul_replay_gather<BLOCKS><<<grid, block, 0, stream>>>(
        static_cast<EcMulTraceInput<BLOCKS> *>(output),
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
        is_setup,
        register_as,
        memory_as,
        pointer_max_bits,
        error
    );
    return CHECK_KERNEL();
}

extern "C" int _ec_mul_replay_gather(
    void *output,
    size_t output_len,
    size_t output_start,
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
    uint32_t is_setup,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    if (blocks == 8) {
        return launch_ec_mul_replay_gather<8>(
            output, output_len, output_start, instructions, pc_base, program, memory, seeds,
            predecessors, steps, step_start, num_steps, expected_opcode, is_setup, register_as,
            memory_as, pointer_max_bits, error, stream
        );
    }
    if (blocks == 12) {
        return launch_ec_mul_replay_gather<12>(
            output, output_len, output_start, instructions, pc_base, program, memory, seeds,
            predecessors, steps, step_start, num_steps, expected_opcode, is_setup, register_as,
            memory_as, pointer_max_bits, error, stream
        );
    }
    return 1;
}
