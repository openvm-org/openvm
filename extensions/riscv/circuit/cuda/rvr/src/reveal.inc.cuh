__global__ void reveal_replay_tracegen(
    Fp *trace,
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
    uint32_t opcode,
    uint32_t register_as,
    uint32_t public_values_as,
    size_t pointer_max_bits,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= height) return;
    RowSlice row(trace + idx, height);
    row.fill_zero(0, sizeof(RevealCols<uint8_t>));
    if (idx >= num_steps) return;

    ReplayRevealInput input = {};
    if (!replay_reveal(
            instructions,
            pc_base,
            program,
            memory,
            seeds,
            predecessors,
            steps[step_start + idx],
            opcode,
            register_as,
            public_values_as,
            pointer_max_bits,
            input,
            error
        )) {
        return;
    }

    auto reveal = Reveal(
        pointer_max_bits,
        VariableRangeChecker(range_checker, range_checker_num_bins),
        timestamp_max_bits
    );
    reveal.fill_trace_row(row, input);
}

extern "C" int _reveal_replay_tracegen(
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
    uint32_t opcode,
    uint32_t register_as,
    uint32_t public_values_as,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(RevealCols<uint8_t>));
    assert(d_memory.len() == d_predecessors.len());
    assert(step_start <= d_steps.len());
    assert(num_steps <= d_steps.len() - step_start);
    assert(height >= num_steps);
    auto [grid, block] = kernel_launch_params(height, REPLAY_THREADS);
    reveal_replay_tracegen<<<grid, block, 0, stream>>>(
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
        opcode,
        register_as,
        public_values_as,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
