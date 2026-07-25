#include "arch/rvr/replay.cuh"

constexpr uint32_t HINTSTORE_REPLAY_ERROR = 701;

struct ReplayHintStoreInput {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t num_words;
    uint32_t mem_ptr_ptr;
    uint32_t num_words_ptr;
    uint32_t mem_ptr;
    bool is_single;
};

static __device__ __forceinline__ bool canonical_register_pointer(uint32_t pointer) {
    return pointer < 32 * RV64_REGISTER_NUM_LIMBS &&
           pointer % RV64_REGISTER_NUM_LIMBS == 0;
}

static __device__ __forceinline__ uint64_t replay_hint_u64(
    uint16_t const (&value)[BLOCK_FE_WIDTH]
) {
    return uint64_t(value[0]) | (uint64_t(value[1]) << 16) |
           (uint64_t(value[2]) << 32) | (uint64_t(value[3]) << 48);
}

static __device__ bool replay_hintstore_instruction(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    RvrReplayStep const &step,
    uint32_t hint_stored_opcode,
    uint32_t hint_buffer_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    ReplayHintStoreInput &out
) {
    size_t program_index = step.program_index;
    if (program_index + 1 >= program.len()) return false;
    auto const &from = program[program_index];
    auto const &to = program[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % DEFAULT_PC_STEP != 0 ||
        from.pc > UINT32_MAX - DEFAULT_PC_STEP || to.pc != from.pc + DEFAULT_PC_STEP) {
        return false;
    }
    size_t instruction_index = (from.pc - pc_base) / DEFAULT_PC_STEP;
    if (instruction_index >= instructions.len()) return false;
    auto const &instruction = instructions[instruction_index];
    bool is_single = instruction.words[0] == hint_stored_opcode;
    if ((!is_single && instruction.words[0] != hint_buffer_opcode) ||
        instruction.words[3] != 0 || instruction.words[4] != register_as ||
        instruction.words[5] != memory_as || instruction.words[6] != 0 ||
        instruction.words[7] != 0 || !canonical_register_pointer(instruction.words[2]) ||
        (is_single ? instruction.words[1] != 0
                   : !canonical_register_pointer(instruction.words[1]))) {
        return false;
    }

    size_t mem_ptr_index = step.memory_start;
    if (mem_ptr_index >= memory.len() || mem_ptr_index >= predecessors.len()) return false;
    auto const &mem_ptr_read = memory[mem_ptr_index];
    if (mem_ptr_read.timestamp != from.timestamp || preflight_is_write(mem_ptr_read) ||
        preflight_address_space(mem_ptr_read) != register_as ||
        mem_ptr_read.pointer != instruction.words[2] / U16_CELL_SIZE) {
        return false;
    }
    ReplayPreviousValue mem_ptr_previous;
    if (!replay_previous_value(
            mem_ptr_index,
            mem_ptr_read,
            predecessors[mem_ptr_index],
            memory,
            seeds,
            mem_ptr_previous
        )) {
        return false;
    }
    uint64_t mem_ptr_u64 = replay_hint_u64(mem_ptr_read.value);
    if (mem_ptr_u64 > UINT32_MAX || mem_ptr_u64 % MEMORY_BLOCK_BYTES != 0) return false;
    uint32_t mem_ptr = uint32_t(mem_ptr_u64);

    uint32_t num_words = 1;
    size_t write_start = mem_ptr_index + 1;
    if (!is_single) {
        size_t num_words_index = write_start++;
        if (num_words_index >= memory.len() || num_words_index >= predecessors.len()) return false;
        auto const &num_words_read = memory[num_words_index];
        if (num_words_read.timestamp != from.timestamp + 1 ||
            preflight_is_write(num_words_read) ||
            preflight_address_space(num_words_read) != register_as ||
            num_words_read.pointer != instruction.words[1] / U16_CELL_SIZE) {
            return false;
        }
        ReplayPreviousValue num_words_previous;
        if (!replay_previous_value(
                num_words_index,
                num_words_read,
                predecessors[num_words_index],
                memory,
                seeds,
                num_words_previous
            )) {
            return false;
        }
        uint64_t count = replay_hint_u64(num_words_read.value);
        if (count == 0 || count > MAX_HINT_BUFFER_DWORDS) return false;
        num_words = uint32_t(count);
    }

    if (num_words > (UINT32_MAX - from.timestamp) / 3 ||
        to.timestamp != from.timestamp + 3 * num_words) {
        return false;
    }
    uint64_t access_end = uint64_t(mem_ptr) + uint64_t(num_words) * MEMORY_BLOCK_BYTES;
    uint64_t pointer_limit = pointer_max_bits < 32 ? uint64_t(1) << pointer_max_bits
                                                  : uint64_t(1) << 32;
    if (access_end > pointer_limit || write_start > memory.len() ||
        size_t(num_words) > memory.len() - write_start ||
        write_start > predecessors.len() ||
        size_t(num_words) > predecessors.len() - write_start) {
        return false;
    }
    for (uint32_t word = 0; word < num_words; word++) {
        size_t write_index = write_start + word;
        auto const &write = memory[write_index];
        if (write.timestamp != from.timestamp + 2 + 3 * word || !preflight_is_write(write) ||
            preflight_address_space(write) != memory_as ||
            write.pointer != (mem_ptr + word * MEMORY_BLOCK_BYTES) / U16_CELL_SIZE) {
            return false;
        }
        ReplayPreviousValue write_previous;
        if (!replay_previous_value(
                write_index, write, predecessors[write_index], memory, seeds, write_previous
            )) {
            return false;
        }
    }
    size_t next_index = write_start + num_words;
    if (next_index < memory.len() && memory[next_index].timestamp < to.timestamp) return false;

    out.from_pc = from.pc;
    out.from_timestamp = from.timestamp;
    out.num_words = num_words;
    out.mem_ptr_ptr = instruction.words[2];
    out.num_words_ptr = is_single ? UINT32_MAX : instruction.words[1];
    out.mem_ptr = mem_ptr;
    out.is_single = is_single;
    return true;
}

__global__ void hintstore_replay_count(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    DeviceBufferConstView<uint32_t> predecessors,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t hint_stored_opcode,
    uint32_t hint_buffer_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    uint32_t *counts,
    uint32_t *error
) {
    size_t index = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (index >= num_steps) return;
    ReplayHintStoreInput input{};
    if (!replay_hintstore_instruction(
            instructions,
            pc_base,
            program,
            memory,
            seeds,
            predecessors,
            steps[step_start + index],
            hint_stored_opcode,
            hint_buffer_opcode,
            register_as,
            memory_as,
            pointer_max_bits,
            input
        )) {
        preflight_set_error(error, HINTSTORE_REPLAY_ERROR);
        counts[index] = 0;
        return;
    }
    counts[index] = input.num_words;
}

__global__ void hintstore_replay_tracegen(
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
    DeviceBufferConstView<uint32_t> row_offsets,
    uint32_t hint_stored_opcode,
    uint32_t hint_buffer_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    uint32_t *error
) {
    size_t instruction_index = blockIdx.x;
    if (instruction_index >= num_steps || *error != 0) return;

    __shared__ ReplayHintStoreInput input;
    __shared__ bool valid;
    if (threadIdx.x == 0) {
        valid = replay_hintstore_instruction(
            instructions,
            pc_base,
            program,
            memory,
            seeds,
            predecessors,
            steps[step_start + instruction_index],
            hint_stored_opcode,
            hint_buffer_opcode,
            register_as,
            memory_as,
            pointer_max_bits,
            input
        );
        if (!valid || row_offsets[instruction_index + 1] - row_offsets[instruction_index] !=
                          input.num_words) {
            preflight_set_error(error, HINTSTORE_REPLAY_ERROR);
            valid = false;
        }
    }
    __syncthreads();
    if (!valid) return;

    size_t write_start = steps[step_start + instruction_index].memory_start +
                         (input.is_single ? 1 : 2);
    for (uint32_t local_idx = threadIdx.x; local_idx < input.num_words;
         local_idx += blockDim.x) {
        size_t row_index = row_offsets[instruction_index] + local_idx;
        if (row_index >= height) {
            preflight_set_error(error, HINTSTORE_REPLAY_ERROR);
            return;
        }
        size_t write_index = write_start + local_idx;
        auto const &write = memory[write_index];
        ReplayPreviousValue mem_ptr_previous;
        ReplayPreviousValue num_words_previous{};
        ReplayPreviousValue write_previous;
        size_t mem_ptr_index = steps[step_start + instruction_index].memory_start;
        bool replayed = replay_previous_value(
            mem_ptr_index,
            memory[mem_ptr_index],
            predecessors[mem_ptr_index],
            memory,
            seeds,
            mem_ptr_previous
        );
        if (!input.is_single) {
            size_t num_words_index = mem_ptr_index + 1;
            replayed &= replay_previous_value(
                num_words_index,
                memory[num_words_index],
                predecessors[num_words_index],
                memory,
                seeds,
                num_words_previous
            );
        }
        replayed &= replay_previous_value(
            write_index, write, predecessors[write_index], memory, seeds, write_previous
        );
        if (!replayed) {
            preflight_set_error(error, HINTSTORE_REPLAY_ERROR);
            return;
        }

        Rv64HintStoreRecordHeader record{};
        record.num_words = input.num_words;
        record.from_pc = input.from_pc;
        record.timestamp = input.from_timestamp;
        record.mem_ptr_ptr = input.mem_ptr_ptr;
        record.mem_ptr = input.mem_ptr;
        record.mem_ptr_aux_record.prev_timestamp = mem_ptr_previous.timestamp;
        record.num_words_ptr = input.num_words_ptr;
        record.num_words_read.prev_timestamp = num_words_previous.timestamp;

        Rv64HintStoreVars vars{};
        vars.write_aux.prev_timestamp = write_previous.timestamp;
#pragma unroll
        for (uint32_t cell = 0; cell < BLOCK_FE_WIDTH; cell++) {
            vars.write_aux.prev_data[2 * cell] = uint8_t(write_previous.value[cell]);
            vars.write_aux.prev_data[2 * cell + 1] = uint8_t(write_previous.value[cell] >> 8);
            vars.data[2 * cell] = uint8_t(write.value[cell]);
            vars.data[2 * cell + 1] = uint8_t(write.value[cell] >> 8);
        }
        RowSlice row(trace + row_index, height);
        auto filler = Rv64HintStore(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        filler.fill_trace_row(row, record, vars, local_idx);
    }
}


extern "C" int _hintstore_replay_count(
    DeviceBufferConstView<RvrReplayInstruction> d_instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> d_program,
    DeviceBufferConstView<PreflightMemoryEvent> d_memory,
    DeviceBufferConstView<PreflightInitialWrite> d_seeds,
    DeviceBufferConstView<uint32_t> d_predecessors,
    DeviceBufferConstView<RvrReplayStep> d_steps,
    size_t step_start,
    size_t num_steps,
    uint32_t hint_stored_opcode,
    uint32_t hint_buffer_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    uint32_t *d_counts,
    uint32_t *d_error,
    cudaStream_t stream
) {
    assert(d_memory.len() == d_predecessors.len());
    assert(step_start <= d_steps.len());
    assert(num_steps <= d_steps.len() - step_start);
    if (num_steps == 0) return 0;
    auto [grid, block] = kernel_launch_params(num_steps);
    hintstore_replay_count<<<grid, block, 0, stream>>>(
        d_instructions,
        pc_base,
        d_program,
        d_memory,
        d_seeds,
        d_predecessors,
        d_steps,
        step_start,
        num_steps,
        hint_stored_opcode,
        hint_buffer_opcode,
        register_as,
        memory_as,
        pointer_max_bits,
        d_counts,
        d_error
    );
    return CHECK_KERNEL();
}



extern "C" int _hintstore_replay_tracegen(
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
    DeviceBufferConstView<uint32_t> d_row_offsets,
    uint32_t hint_stored_opcode,
    uint32_t hint_buffer_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    uint32_t *d_error,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64HintStoreCols<uint8_t>));
    assert(d_memory.len() == d_predecessors.len());
    assert(step_start <= d_steps.len());
    assert(num_steps <= d_steps.len() - step_start);
    assert(d_row_offsets.len() == num_steps + 1);
    if (cudaError_t err = cudaMemsetAsync(d_trace, 0, height * width * sizeof(Fp), stream);
        err != cudaSuccess) {
        return err;
    }
    if (num_steps == 0) return 0;
    hintstore_replay_tracegen<<<num_steps, 128, 0, stream>>>(
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
        d_row_offsets,
        hint_stored_opcode,
        hint_buffer_opcode,
        register_as,
        memory_as,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits,
        d_error
    );
    return CHECK_KERNEL();
}
