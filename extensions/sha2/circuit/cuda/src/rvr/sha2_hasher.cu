#include "../sha2_hasher.cu"
#include "rvr/replay.cuh"

template <typename V>
__device__ __forceinline__ typename V::Word replay_word_from_bytes_be(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    size_t first_event,
    size_t byte_offset
) {
    typename V::Word acc = 0;
#pragma unroll
    for (size_t i = 0; i < V::WORD_U8S; i++) {
        acc = (acc << 8) |
              static_cast<typename V::Word>(sha2_replay_byte(memory, first_event, byte_offset + i));
    }
    return acc;
}

template <typename V>
__device__ __forceinline__ typename V::Word replay_word_from_bytes_le(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    size_t first_event,
    size_t byte_offset
) {
    typename V::Word acc = 0;
#pragma unroll
    for (int i = static_cast<int>(V::WORD_U8S) - 1; i >= 0; i--) {
        acc = (acc << 8) |
              static_cast<typename V::Word>(sha2_replay_byte(memory, first_event, byte_offset + i));
    }
    return acc;
}

template <typename V>
__global__ void sha2_first_pass_phase1_replay(
    typename V::Word *__restrict__ d_scratch,
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
    uint32_t pointer_max_bits,
    typename V::Word *prev_hashes,
    uint32_t *error
) {
    uint32_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_steps) return;

    using SL = Sha2ScratchLayout<V>;
    typename V::Word *scratch = d_scratch + size_t(block_idx) * SL::WORDS_PER_BLOCK;
    typename V::Word *prev_hash = prev_hashes + block_idx * V::HASH_WORDS;
    if (step_start > steps.len() || block_idx >= steps.len() - step_start) {
        preflight_set_error(error, SHA2_REPLAY_ERROR);
        for (size_t i = 0; i < SL::WORDS_PER_BLOCK; i++) scratch[i] = 0;
#pragma unroll
        for (size_t i = 0; i < V::HASH_WORDS; i++) prev_hash[i] = 0;
        return;
    }

    Sha2ReplayInput input;
    if (!replay_sha2_instruction(
            instructions,
            pc_base,
            program,
            memory,
            seeds,
            predecessors,
            steps[step_start + block_idx],
            expected_opcode,
            register_as,
            memory_as,
            pointer_max_bits,
            input
        )) {
        preflight_set_error(error, SHA2_REPLAY_ERROR);
        for (size_t i = 0; i < SL::WORDS_PER_BLOCK; i++) scratch[i] = 0;
#pragma unroll
        for (size_t i = 0; i < V::HASH_WORDS; i++) prev_hash[i] = 0;
        return;
    }

#pragma unroll
    for (size_t i = 0; i < V::HASH_WORDS; i++) {
        prev_hash[i] =
            replay_word_from_bytes_le<V>(memory, input.state_start, i * V::WORD_U8S);
    }
    typename V::Word w_buf[V::BLOCK_WORDS];
#pragma unroll
    for (size_t i = 0; i < V::BLOCK_WORDS; i++) {
        w_buf[i] = replay_word_from_bytes_be<V>(memory, input.input_start, i * V::WORD_U8S);
    }

    typename V::Word a = prev_hash[0], b = prev_hash[1], c = prev_hash[2], d = prev_hash[3];
    typename V::Word e = prev_hash[4], f = prev_hash[5], g = prev_hash[6], h = prev_hash[7];
    for (uint32_t row_in_block = 0; row_in_block < V::ROWS_PER_BLOCK; row_in_block++) {
        typename V::Word *row_scratch = scratch + row_in_block * SL::WORDS_PER_ROW;
        row_scratch[0] = a;
        row_scratch[1] = b;
        row_scratch[2] = c;
        row_scratch[3] = d;
        row_scratch[4] = e;
        row_scratch[5] = f;
        row_scratch[6] = g;
        row_scratch[7] = h;
        for (uint32_t i = 0; i < V::BLOCK_WORDS; i++) {
            row_scratch[SHA2_SCRATCH_STATE + i] = w_buf[i];
        }

        if (row_in_block < V::ROUND_ROWS) {
            for (uint32_t j = 0; j < V::ROUNDS_PER_ROW; j++) {
                uint32_t t = row_in_block * V::ROUNDS_PER_ROW + j;
                typename V::Word w_val;
                if (t < V::BLOCK_WORDS) {
                    w_val = w_buf[t & (V::BLOCK_WORDS - 1)];
                } else {
                    w_val = sha2::small_sig1<V>(w_buf[(t - 2) & (V::BLOCK_WORDS - 1)]) +
                            w_buf[(t - 7) & (V::BLOCK_WORDS - 1)] +
                            sha2::small_sig0<V>(w_buf[(t - 15) & (V::BLOCK_WORDS - 1)]) +
                            w_buf[(t - 16) & (V::BLOCK_WORDS - 1)];
                    w_buf[t & (V::BLOCK_WORDS - 1)] = w_val;
                }
                typename V::Word t1 =
                    h + sha2::big_sig1<V>(e) + sha2::ch<V>(e, f, g) + V::K(t) + w_val;
                typename V::Word t2 = sha2::big_sig0<V>(a) + sha2::maj<V>(a, b, c);
                h = g;
                g = f;
                f = e;
                e = d + t1;
                d = c;
                c = b;
                b = a;
                a = t1 + t2;
            }
        }
    }

    // Deterministic outputs must be independently recomputed and compared with the
    // transcript: the logged post-write state has to equal prev_hash + final working
    // variables, or the device error flag rejects the segment before proving.
    typename V::Word const final_vars[8] = {a, b, c, d, e, f, g, h};
    bool output_matches = true;
#pragma unroll
    for (size_t i = 0; i < V::HASH_WORDS; i++) {
        typename V::Word logged =
            replay_word_from_bytes_le<V>(memory, input.write_start, i * V::WORD_U8S);
        if (logged != static_cast<typename V::Word>(prev_hash[i] + final_vars[i])) {
            output_matches = false;
        }
    }
    if (!output_matches) {
        preflight_set_error(error, SHA2_REPLAY_ERROR);
        for (size_t i = 0; i < SL::WORDS_PER_BLOCK; i++) scratch[i] = 0;
#pragma unroll
        for (size_t i = 0; i < V::HASH_WORDS; i++) prev_hash[i] = 0;
        return;
    }
}

template <typename V>
int launch_sha2_block_replay_tracegen(
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
    uint32_t pointer_max_bits,
    typename V::Word *d_prev_hashes,
    uint32_t *d_bitwise_lookup,
    typename V::Word *d_scratch,
    size_t scratch_words,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *error,
    cudaStream_t stream
) {
    using SL = Sha2ScratchLayout<V>;
    if (num_steps > UINT32_MAX || num_steps > SIZE_MAX / SL::WORDS_PER_BLOCK ||
        scratch_words < num_steps * SL::WORDS_PER_BLOCK) {
        return cudaErrorInvalidValue;
    }

    auto [block_grid, block_size] = kernel_launch_params(num_steps, 256);
    sha2_first_pass_phase1_replay<V><<<block_grid, block_size, 0, stream>>>(
        d_scratch,
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
        pointer_max_bits,
        d_prev_hashes,
        error
    );
    if (int result = CHECK_KERNEL()) return result;

    size_t rows_used = num_steps * V::ROWS_PER_BLOCK;
    auto [row_grid, row_block_size] = kernel_launch_params(rows_used, 256);
    sha2_first_pass_phase2<V><<<row_grid, row_block_size, 0, stream>>>(
        d_trace,
        trace_height,
        static_cast<uint32_t>(num_steps),
        num_steps,
        d_prev_hashes,
        d_scratch,
        d_bitwise_lookup,
        d_range_checker,
        range_checker_num_bins
    );
    if (int result = CHECK_KERNEL()) return result;

    sha2_first_pass_phase3<V><<<block_grid, block_size, 0, stream>>>(
        d_trace, trace_height, static_cast<uint32_t>(num_steps), num_steps
    );
    return CHECK_KERNEL();
}

extern "C" {
int launch_sha256_block_replay_tracegen(
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
    uint32_t pointer_max_bits,
    uint32_t *d_prev_hashes,
    uint32_t *d_bitwise_lookup,
    uint32_t *d_scratch,
    size_t scratch_words,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *error,
    cudaStream_t stream
) {
    return launch_sha2_block_replay_tracegen<Sha256Variant>(
        d_trace, trace_height, instructions, pc_base, program, memory, seeds, predecessors, steps,
        step_start, num_steps, expected_opcode, register_as, memory_as, pointer_max_bits,
        d_prev_hashes, d_bitwise_lookup, d_scratch, scratch_words, d_range_checker,
        range_checker_num_bins, error, stream
    );
}

int launch_sha512_block_replay_tracegen(
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
    uint32_t pointer_max_bits,
    uint64_t *d_prev_hashes,
    uint32_t *d_bitwise_lookup,
    uint64_t *d_scratch,
    size_t scratch_words,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *error,
    cudaStream_t stream
) {
    return launch_sha2_block_replay_tracegen<Sha512Variant>(
        d_trace, trace_height, instructions, pc_base, program, memory, seeds, predecessors, steps,
        step_start, num_steps, expected_opcode, register_as, memory_as, pointer_max_bits,
        d_prev_hashes, d_bitwise_lookup, d_scratch, scratch_words, d_range_checker,
        range_checker_num_bins, error, stream
    );
}
}
