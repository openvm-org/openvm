#include "../src/call.cu"

#include "arch/rvr/replay.cuh"

static constexpr uint32_t DEFERRAL_CALL_REPLAY_ERROR = 1201;
static constexpr uint32_t DEFERRAL_CALL_EVENTS = 19;

static __device__ __forceinline__ uint32_t deferral_call_field_reference(
    PreflightMemoryEvent const &event
) {
    return uint32_t(event.value[0]) | (uint32_t(event.value[1]) << 16);
}

static __device__ bool deferral_call_field_block(
    PreflightMemoryEvent const &event,
    DeviceBufferConstView<RvrFieldBlock> values,
    Fp (&out)[BLOCK_FE_WIDTH]
) {
    if (event.value[2] != 0 || event.value[3] != 0) return false;
    uint32_t reference = deferral_call_field_reference(event);
    if (reference >= values.len()) return false;
#pragma unroll
    for (size_t lane = 0; lane < BLOCK_FE_WIDTH; lane++) {
        if (values[reference].values[lane] >= Fp::P) return false;
        out[lane] = Fp::fromRaw(values[reference].values[lane]);
    }
    return true;
}

static __device__ bool deferral_call_previous_field(
    size_t event_index,
    PreflightMemoryEvent const &event,
    uint32_t predecessor,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<RvrFieldBlock> values,
    DeviceBufferConstView<RvrFieldBlock> seed_values,
    uint32_t &previous_timestamp,
    Fp (&previous_value)[BLOCK_FE_WIDTH]
) {
    if (predecessor == 0) {
        previous_timestamp = 0;
        return !preflight_is_write(event) &&
               deferral_call_field_block(event, values, previous_value);
    }
    if ((predecessor & MEMORY_PREDECESSOR_SEED_BIT) != 0) {
        uint32_t seed_index = predecessor & MEMORY_PREDECESSOR_INDEX_MASK;
        if (!preflight_is_write(event) || seed_index >= seeds.len()) return false;
        auto const &seed = seeds[seed_index];
        if (seed.address_space != preflight_address_space(event) ||
            seed.pointer != event.pointer || seed.initial_value[2] != 0 ||
            seed.initial_value[3] != 0) {
            return false;
        }
        uint32_t reference =
            uint32_t(seed.initial_value[0]) | (uint32_t(seed.initial_value[1]) << 16);
        if (reference >= seed_values.len()) return false;
        previous_timestamp = 0;
#pragma unroll
        for (size_t lane = 0; lane < BLOCK_FE_WIDTH; lane++) {
            if (seed_values[reference].values[lane] >= Fp::P) return false;
            previous_value[lane] = Fp::fromRaw(seed_values[reference].values[lane]);
        }
        return true;
    }

    size_t previous_index = predecessor - 1;
    if (previous_index >= event_index || previous_index >= memory.len()) return false;
    auto const &previous = memory[previous_index];
    if (preflight_address_space(previous) != preflight_address_space(event) ||
        previous.pointer != event.pointer || previous.timestamp >= event.timestamp ||
        !deferral_call_field_block(previous, values, previous_value)) {
        return false;
    }
    previous_timestamp = previous.timestamp;
    if (!preflight_is_write(event)) {
        Fp current[BLOCK_FE_WIDTH];
        if (!deferral_call_field_block(event, values, current)) return false;
#pragma unroll
        for (size_t lane = 0; lane < BLOCK_FE_WIDTH; lane++) {
            if (current[lane] != previous_value[lane]) return false;
        }
    }
    return true;
}

static __device__ bool deferral_call_u16_event(
    size_t event_index,
    uint32_t timestamp,
    uint32_t address_space,
    uint32_t pointer,
    bool is_write,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    ReplayPreviousValue &previous,
    uint32_t *error
) {
    if (event_index >= memory.len() || event_index >= predecessors.len()) {
        preflight_set_error(error, DEFERRAL_CALL_REPLAY_ERROR);
        return false;
    }
    auto const &event = memory[event_index];
    if (event.timestamp != timestamp || preflight_address_space(event) != address_space ||
        event.pointer != pointer || preflight_is_write(event) != is_write ||
        !replay_previous_value(
            event_index, event, predecessors[event_index], memory, seeds, previous
        )) {
        preflight_set_error(error, DEFERRAL_CALL_REPLAY_ERROR);
        return false;
    }
    return true;
}

static __device__ bool deferral_call_field_event(
    size_t event_index,
    uint32_t timestamp,
    uint32_t pointer,
    bool is_write,
    uint32_t deferral_as,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<RvrFieldBlock> values,
    DeviceBufferConstView<RvrFieldBlock> seed_values,
    DeviceBufferConstView<uint32_t> predecessors,
    uint32_t &previous_timestamp,
    Fp (&previous_value)[BLOCK_FE_WIDTH],
    Fp (&current_value)[BLOCK_FE_WIDTH],
    uint32_t *error
) {
    if (event_index >= memory.len() || event_index >= predecessors.len()) {
        preflight_set_error(error, DEFERRAL_CALL_REPLAY_ERROR);
        return false;
    }
    auto const &event = memory[event_index];
    if (event.timestamp != timestamp || preflight_address_space(event) != deferral_as ||
        event.pointer != pointer || preflight_is_write(event) != is_write ||
        !deferral_call_field_block(event, values, current_value) ||
        !deferral_call_previous_field(
            event_index,
            event,
            predecessors[event_index],
            memory,
            seeds,
            values,
            seed_values,
            previous_timestamp,
            previous_value
        )) {
        preflight_set_error(error, DEFERRAL_CALL_REPLAY_ERROR);
        return false;
    }
    return true;
}

static __device__ __forceinline__ void deferral_call_block_bytes(
    uint16_t const (&value)[BLOCK_FE_WIDTH], uint8_t *out
) {
#pragma unroll
    for (size_t lane = 0; lane < BLOCK_FE_WIDTH; lane++) {
        out[2 * lane] = uint8_t(value[lane]);
        out[2 * lane + 1] = uint8_t(value[lane] >> 8);
    }
}

__global__ void deferral_call_replay_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<RvrFieldBlock> field_values,
    DeviceBufferConstView<RvrFieldBlock> field_seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t deferral_as,
    uint32_t byte_pointer_bits,
    uint32_t *count_ptr,
    size_t num_def_circuits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    uint32_t *bitwise_ptr,
    FpArray<16> *poseidon2_records,
    DeferralPoseidon2Count *poseidon2_counts,
    uint32_t *poseidon2_idx,
    size_t poseidon2_capacity,
    size_t address_bits,
    uint32_t *error
) {
    size_t row_idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (row_idx >= height) return;
    RowSlice row(trace + row_idx, height);
    if (row_idx >= num_steps) {
        row.fill_zero(0, sizeof(DeferralCallCols<uint8_t>));
        return;
    }

    size_t step_idx = step_start + row_idx;
    if (step_idx >= steps.len()) {
        preflight_set_error(error, DEFERRAL_CALL_REPLAY_ERROR);
        return;
    }
    auto const step = steps[step_idx];
    ReplayProgramTransition transition;
    if (!replay_program_transition(
            instructions,
            pc_base,
            program,
            step.program_index,
            DEFERRAL_CALL_EVENTS,
            ReplayPcEffect::Sequential,
            transition,
            error,
            DEFERRAL_CALL_REPLAY_ERROR
        )) {
        return;
    }
    auto const &instruction = *transition.instruction;
    if (instruction.words[0] != expected_opcode || instruction.words[4] != register_as ||
        instruction.words[5] != memory_as || instruction.words[6] != 0 ||
        instruction.words[7] != 0 || instruction.words[1] >= 32u * 8u ||
        instruction.words[2] >= 32u * 8u || instruction.words[1] % 8u != 0 ||
        instruction.words[2] % 8u != 0 || instruction.words[3] >= num_def_circuits ||
        step.memory_start > memory.len() ||
        memory.len() - step.memory_start < DEFERRAL_CALL_EVENTS) {
        preflight_set_error(error, DEFERRAL_CALL_REPLAY_ERROR);
        return;
    }
    size_t event_start = step.memory_start;
    if (event_start + DEFERRAL_CALL_EVENTS < memory.len() &&
        memory[event_start + DEFERRAL_CALL_EVENTS].timestamp < transition.to->timestamp) {
        preflight_set_error(error, DEFERRAL_CALL_REPLAY_ERROR);
        return;
    }

    DeferralCallRecord<Fp> record{};
    record.adapter.from_pc = transition.from->pc;
    record.adapter.from_timestamp = transition.from->timestamp;
    record.adapter.rd_ptr = Fp(instruction.words[1]);
    record.adapter.rs_ptr = Fp(instruction.words[2]);
    record.core.deferral_idx = Fp(instruction.words[3]);

    ReplayPreviousValue rd_previous;
    ReplayPreviousValue rs_previous;
    if (!deferral_call_u16_event(
            event_start,
            transition.from->timestamp,
            register_as,
            instruction.words[1] / 2,
            false,
            memory,
            seeds,
            predecessors,
            rd_previous,
            error
        ) ||
        !deferral_call_u16_event(
            event_start + 1,
            transition.from->timestamp + 1,
            register_as,
            instruction.words[2] / 2,
            false,
            memory,
            seeds,
            predecessors,
            rs_previous,
            error
        )) {
        return;
    }
    record.adapter.rd_aux.prev_timestamp = rd_previous.timestamp;
    record.adapter.rs_aux.prev_timestamp = rs_previous.timestamp;
    deferral_call_block_bytes(memory[event_start].value, record.adapter.rd_val);
    deferral_call_block_bytes(memory[event_start + 1].value, record.adapter.rs_val);
    uint32_t output_ptr = uint32_t(memory[event_start].value[0]) |
                          (uint32_t(memory[event_start].value[1]) << 16);
    uint32_t input_ptr = uint32_t(memory[event_start + 1].value[0]) |
                         (uint32_t(memory[event_start + 1].value[1]) << 16);
    uint64_t domain_end = byte_pointer_bits < 32 ? uint64_t(1) << byte_pointer_bits
                                                : uint64_t(1) << 32;
    if (memory[event_start].value[2] != 0 || memory[event_start].value[3] != 0 ||
        memory[event_start + 1].value[2] != 0 ||
        memory[event_start + 1].value[3] != 0 ||
        (output_ptr & (MEMORY_BLOCK_BYTES - 1)) != 0 ||
        (input_ptr & (MEMORY_BLOCK_BYTES - 1)) != 0 ||
        uint64_t(output_ptr) + OUTPUT_TOTAL_BYTES > domain_end ||
        uint64_t(input_ptr) + COMMIT_NUM_BYTES > domain_end) {
        preflight_set_error(error, DEFERRAL_CALL_REPLAY_ERROR);
        return;
    }

#pragma unroll
    for (size_t block = 0; block < COMMIT_MEMORY_OPS; block++) {
        ReplayPreviousValue previous;
        size_t event_idx = event_start + 2 + block;
        if (!deferral_call_u16_event(
                event_idx,
                transition.from->timestamp + 2 + block,
                memory_as,
                input_ptr / 2 + block * BLOCK_FE_WIDTH,
                false,
                memory,
                seeds,
                predecessors,
                previous,
                error
            )) {
            return;
        }
        record.adapter.input_commit_aux[block].prev_timestamp = previous.timestamp;
        deferral_call_block_bytes(
            memory[event_idx].value, record.core.reads.input_commit + block * MEMORY_BLOCK_BYTES
        );
    }

    uint64_t input_acc_ptr_wide =
        uint64_t(instruction.words[3]) * 2u * DIGEST_SIZE;
    uint64_t output_acc_ptr_wide = input_acc_ptr_wide + DIGEST_SIZE;
    if (output_acc_ptr_wide + DIGEST_SIZE > (uint64_t(1) << 32)) {
        preflight_set_error(error, DEFERRAL_CALL_REPLAY_ERROR);
        return;
    }
    uint32_t input_acc_ptr = uint32_t(input_acc_ptr_wide);
    uint32_t output_acc_ptr = uint32_t(output_acc_ptr_wide);
    size_t field_read_start = event_start + 2 + COMMIT_MEMORY_OPS;
#pragma unroll
    for (size_t block = 0; block < DIGEST_F_MEMORY_OPS; block++) {
        uint32_t previous_timestamp;
        Fp previous[BLOCK_FE_WIDTH];
        Fp current[BLOCK_FE_WIDTH];
        if (!deferral_call_field_event(
                field_read_start + block,
                transition.from->timestamp + 2 + COMMIT_MEMORY_OPS + block,
                input_acc_ptr + block * BLOCK_FE_WIDTH,
                false,
                deferral_as,
                memory,
                seeds,
                field_values,
                field_seeds,
                predecessors,
                previous_timestamp,
                previous,
                current,
                error
            )) {
            return;
        }
        record.adapter.old_input_acc_aux[block].prev_timestamp = previous_timestamp;
#pragma unroll
        for (size_t lane = 0; lane < BLOCK_FE_WIDTH; lane++) {
            record.core.reads.old_input_acc[block * BLOCK_FE_WIDTH + lane] = current[lane];
        }
    }
#pragma unroll
    for (size_t block = 0; block < DIGEST_F_MEMORY_OPS; block++) {
        uint32_t previous_timestamp;
        Fp previous[BLOCK_FE_WIDTH];
        Fp current[BLOCK_FE_WIDTH];
        size_t event_idx = field_read_start + DIGEST_F_MEMORY_OPS + block;
        if (!deferral_call_field_event(
                event_idx,
                transition.from->timestamp + 2 + COMMIT_MEMORY_OPS +
                    DIGEST_F_MEMORY_OPS + block,
                output_acc_ptr + block * BLOCK_FE_WIDTH,
                false,
                deferral_as,
                memory,
                seeds,
                field_values,
                field_seeds,
                predecessors,
                previous_timestamp,
                previous,
                current,
                error
            )) {
            return;
        }
        record.adapter.old_output_acc_aux[block].prev_timestamp = previous_timestamp;
#pragma unroll
        for (size_t lane = 0; lane < BLOCK_FE_WIDTH; lane++) {
            record.core.reads.old_output_acc[block * BLOCK_FE_WIDTH + lane] = current[lane];
        }
    }

    size_t output_start =
        field_read_start + 2 * DIGEST_F_MEMORY_OPS;
    uint8_t output_key[OUTPUT_TOTAL_BYTES];
#pragma unroll
    for (size_t block = 0; block < OUTPUT_TOTAL_MEMORY_OPS; block++) {
        ReplayPreviousValue previous;
        size_t event_idx = output_start + block;
        if (!deferral_call_u16_event(
                event_idx,
                transition.from->timestamp + event_idx - event_start,
                memory_as,
                output_ptr / 2 + block * BLOCK_FE_WIDTH,
                true,
                memory,
                seeds,
                predecessors,
                previous,
                error
            )) {
            return;
        }
        auto &aux = record.adapter.output_commit_and_len_aux[block];
        aux.prev_timestamp = previous.timestamp;
        deferral_call_block_bytes(previous.value, aux.prev_data);
        deferral_call_block_bytes(
            memory[event_idx].value, output_key + block * MEMORY_BLOCK_BYTES
        );
    }
#pragma unroll
    for (size_t byte = 0; byte < COMMIT_NUM_BYTES; byte++) {
        record.core.writes.output_commit[byte] = output_key[byte];
    }
#pragma unroll
    for (size_t byte = 0; byte < F_NUM_BYTES; byte++) {
        record.core.writes.output_len[byte] = output_key[COMMIT_NUM_BYTES + byte];
    }
    if (output_key[COMMIT_NUM_BYTES + 4] != 0 || output_key[COMMIT_NUM_BYTES + 5] != 0 ||
        output_key[COMMIT_NUM_BYTES + 6] != 0 || output_key[COMMIT_NUM_BYTES + 7] != 0) {
        preflight_set_error(error, DEFERRAL_CALL_REPLAY_ERROR);
        return;
    }
    uint32_t output_len = uint32_t(output_key[COMMIT_NUM_BYTES]) |
                          (uint32_t(output_key[COMMIT_NUM_BYTES + 1]) << 8) |
                          (uint32_t(output_key[COMMIT_NUM_BYTES + 2]) << 16) |
                          (uint32_t(output_key[COMMIT_NUM_BYTES + 3]) << 24);
    if (uint64_t(output_len) >= domain_end) {
        preflight_set_error(error, DEFERRAL_CALL_REPLAY_ERROR);
        return;
    }

    size_t field_write_start = output_start + OUTPUT_TOTAL_MEMORY_OPS;
#pragma unroll
    for (size_t group = 0; group < 2; group++) {
        uint32_t pointer = group == 0 ? input_acc_ptr : output_acc_ptr;
#pragma unroll
        for (size_t block = 0; block < DIGEST_F_MEMORY_OPS; block++) {
            uint32_t previous_timestamp;
            Fp previous[BLOCK_FE_WIDTH];
            Fp current[BLOCK_FE_WIDTH];
            size_t event_idx =
                field_write_start + group * DIGEST_F_MEMORY_OPS + block;
            if (!deferral_call_field_event(
                    event_idx,
                    transition.from->timestamp + event_idx - event_start,
                    pointer + block * BLOCK_FE_WIDTH,
                    true,
                    deferral_as,
                    memory,
                    seeds,
                    field_values,
                    field_seeds,
                    predecessors,
                    previous_timestamp,
                    previous,
                    current,
                    error
                )) {
                return;
            }
            auto &aux = group == 0 ? record.adapter.new_input_acc_aux[block]
                                   : record.adapter.new_output_acc_aux[block];
            aux.prev_timestamp = previous_timestamp;
#pragma unroll
            for (size_t lane = 0; lane < BLOCK_FE_WIDTH; lane++) {
                aux.prev_data[lane] = previous[lane];
                if (group == 0) {
                    record.core.writes.new_input_acc[block * BLOCK_FE_WIDTH + lane] =
                        current[lane];
                } else {
                    record.core.writes.new_output_acc[block * BLOCK_FE_WIDTH + lane] =
                        current[lane];
                }
            }
        }
    }

    Histogram count_buffer(count_ptr, num_def_circuits);
    MemoryAuxColsFactory mem_helper(
        VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
    );
    BitwiseOperationLookup bitwise_buffer(bitwise_ptr);
    DeferralPoseidon2Buffer poseidon2_buffer(
        poseidon2_records, poseidon2_counts, poseidon2_idx, poseidon2_capacity
    );
    deferral_call_adapter_tracegen(row, record.adapter, bitwise_buffer, mem_helper, address_bits);
    deferral_call_core_tracegen(
        row.slice_from(COL_INDEX(DeferralCallCols, core)),
        record.core,
        count_buffer,
        bitwise_buffer,
        poseidon2_buffer,
        address_bits
    );
}

extern "C" int _deferral_call_replay_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<RvrFieldBlock> field_values,
    DeviceBufferConstView<RvrFieldBlock> field_seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t deferral_as,
    uint32_t byte_pointer_bits,
    uint32_t *count,
    size_t num_def_circuits,
    uint32_t *range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    uint32_t *bitwise,
    FpArray<16> *poseidon2_records,
    DeferralPoseidon2Count *poseidon2_counts,
    uint32_t *poseidon2_idx,
    size_t poseidon2_capacity,
    size_t address_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    if (width != sizeof(DeferralCallCols<uint8_t>) || num_steps > height) {
        return int(cudaErrorInvalidValue);
    }
    auto [grid, block] = kernel_launch_params(height, 256);
    deferral_call_replay_tracegen<<<grid, block, 0, stream>>>(
        trace,
        height,
        instructions,
        pc_base,
        program,
        memory,
        seeds,
        field_values,
        field_seeds,
        predecessors,
        steps,
        step_start,
        num_steps,
        expected_opcode,
        register_as,
        memory_as,
        deferral_as,
        byte_pointer_bits,
        count,
        num_def_circuits,
        range_checker,
        range_checker_num_bins,
        timestamp_max_bits,
        bitwise,
        poseidon2_records,
        poseidon2_counts,
        poseidon2_idx,
        poseidon2_capacity,
        address_bits,
        error
    );
    return CHECK_KERNEL();
}
