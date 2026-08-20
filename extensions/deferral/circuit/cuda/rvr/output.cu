#include "../src/output.cu"

#include "poseidon2-air/params.cuh"
#include "poseidon2-air/tracegen.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "arch/rvr/replay.cuh"

using namespace riscv;

static constexpr uint32_t DEFERRAL_OUTPUT_REPLAY_ERROR = 1101;

struct DeferralOutputReplayCall {
    uint32_t row_start;
    uint32_t num_rows;
};
static_assert(sizeof(DeferralOutputReplayCall) == 8);

static __device__ bool deferral_output_replay_event(
    size_t event_idx,
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
    if (event_idx >= memory.len() || event_idx >= predecessors.len()) {
        preflight_set_error(error, DEFERRAL_OUTPUT_REPLAY_ERROR);
        return false;
    }
    auto const &event = memory[event_idx];
    if (event.timestamp != timestamp || preflight_address_space(event) != address_space ||
        event.pointer != pointer || preflight_is_write(event) != is_write ||
        !replay_previous_value(
            event_idx, event, predecessors[event_idx], memory, seeds, previous
        )) {
        preflight_set_error(error, DEFERRAL_OUTPUT_REPLAY_ERROR);
        return false;
    }
    return true;
}

static __device__ __forceinline__ uint32_t deferral_output_replay_u32(
    uint16_t const (&value)[BLOCK_FE_WIDTH]
) {
    return static_cast<uint32_t>(value[0]) | (static_cast<uint32_t>(value[1]) << 16);
}

static __device__ __forceinline__ void deferral_output_replay_bytes(
    uint16_t const (&value)[BLOCK_FE_WIDTH], uint8_t out[MEMORY_BLOCK_BYTES]
) {
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        out[2 * i] = static_cast<uint8_t>(value[i]);
        out[2 * i + 1] = static_cast<uint8_t>(value[i] >> 8);
    }
}

static __device__ bool deferral_output_resolve_step(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_idx,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t num_def_circuits,
    uint32_t *instruction_idx,
    uint32_t *memory_start,
    uint32_t *output_len,
    uint32_t *error
) {
    if (step_idx >= steps.len()) {
        preflight_set_error(error, DEFERRAL_OUTPUT_REPLAY_ERROR);
        return false;
    }
    auto const step = steps[step_idx];
    size_t program_index = step.program_index;
    if (program_index >= program.len() || program.len() - program_index <= 1) {
        preflight_set_error(error, DEFERRAL_OUTPUT_REPLAY_ERROR);
        return false;
    }
    auto const from = program[program_index];
    auto const to = program[program_index + 1];
    if (from.pc < pc_base ||
        (from.pc - pc_base) % ::program::DEFAULT_PC_STEP != 0) {
        preflight_set_error(error, DEFERRAL_OUTPUT_REPLAY_ERROR);
        return false;
    }
    size_t resolved_idx = (from.pc - pc_base) / ::program::DEFAULT_PC_STEP;
    if (resolved_idx >= instructions.len()) {
        preflight_set_error(error, DEFERRAL_OUTPUT_REPLAY_ERROR);
        return false;
    }
    auto const &instruction = instructions[resolved_idx];
    if (instruction.words[0] != expected_opcode || instruction.words[4] != register_as ||
        instruction.words[5] != memory_as || instruction.words[6] != 0 ||
        instruction.words[7] != 0 || instruction.words[3] >= num_def_circuits ||
        instruction.words[1] >= 32u * 8u || instruction.words[2] >= 32u * 8u ||
        instruction.words[1] % 8u != 0 || instruction.words[2] % 8u != 0 ||
        step.memory_start > memory.len() || memory.len() - step.memory_start < 7) {
        preflight_set_error(error, DEFERRAL_OUTPUT_REPLAY_ERROR);
        return false;
    }
    auto const &len_event = memory[step.memory_start + 6];
    if (len_event.timestamp != from.timestamp + 6 ||
        preflight_address_space(len_event) != memory_as || preflight_is_write(len_event) ||
        len_event.value[2] != 0 || len_event.value[3] != 0) {
        preflight_set_error(error, DEFERRAL_OUTPUT_REPLAY_ERROR);
        return false;
    }
    uint32_t len = deferral_output_replay_u32(len_event.value);
    uint32_t words = len / MEMORY_BLOCK_BYTES;
    if (len % DIGEST_SIZE != 0 || from.timestamp > UINT32_MAX - 7u - words ||
        from.pc > UINT32_MAX - ::program::DEFAULT_PC_STEP ||
        to.timestamp != from.timestamp + 7u + words ||
        to.pc != from.pc + ::program::DEFAULT_PC_STEP ||
        memory.len() - step.memory_start < 7u + words) {
        preflight_set_error(error, DEFERRAL_OUTPUT_REPLAY_ERROR);
        return false;
    }
    size_t next_event = static_cast<size_t>(step.memory_start) + 7u + words;
    if (next_event < memory.len() && memory[next_event].timestamp < to.timestamp) {
        preflight_set_error(error, DEFERRAL_OUTPUT_REPLAY_ERROR);
        return false;
    }
    *instruction_idx = static_cast<uint32_t>(resolved_idx);
    *memory_start = step.memory_start;
    *output_len = len;
    return true;
}

__global__ void deferral_output_replay_count_rows(
    uint32_t *counts,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t num_def_circuits,
    uint32_t *error
) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= num_steps) return;
    uint32_t instruction_idx;
    uint32_t memory_start;
    uint32_t output_len;
    if (!deferral_output_resolve_step(
            instructions,
            pc_base,
            program,
            memory,
            steps,
            step_start + idx,
            expected_opcode,
            register_as,
            memory_as,
            num_def_circuits,
            &instruction_idx,
            &memory_start,
            &output_len,
            error
        )) {
        counts[idx] = 0;
        return;
    }
    counts[idx] = output_len / DIGEST_SIZE + 1;
}

using DeferralPoseidonParams = Poseidon2ParamsS1;
using DeferralPoseidonRow = poseidon2::Poseidon2Row<
    16,
    DeferralPoseidonParams::SBOX_DEGREE,
    DeferralPoseidonParams::SBOX_REGS,
    DeferralPoseidonParams::HALF_FULL_ROUNDS,
    DeferralPoseidonParams::PARTIAL_ROUNDS>;

__global__ void deferral_output_replay_poseidon(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    size_t num_steps,
    const DeferralOutputReplayCall *calls,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t num_def_circuits,
    uint32_t *error
) {
    size_t call_idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (call_idx >= num_steps) return;
    uint32_t instruction_idx;
    uint32_t memory_start;
    uint32_t output_len;
    if (!deferral_output_resolve_step(
            instructions,
            pc_base,
            program,
            memory,
            steps,
            step_start + call_idx,
            expected_opcode,
            register_as,
            memory_as,
            num_def_circuits,
            &instruction_idx,
            &memory_start,
            &output_len,
            error
        )) return;
    auto const &call = calls[call_idx];
    if (call.num_rows != output_len / DIGEST_SIZE + 1) {
        preflight_set_error(error, DEFERRAL_OUTPUT_REPLAY_ERROR);
        return;
    }
    uint32_t deferral_idx = instructions[instruction_idx].words[3];
    Fp capacity[DIGEST_SIZE] = {};
    for (uint32_t section = 0; section < call.num_rows; section++) {
        Fp state[2 * DIGEST_SIZE] = {};
        if (section == 0) {
            state[0] = Fp(deferral_idx);
            state[1] = Fp(output_len);
        } else {
            auto const &event = memory[memory_start + 7 + section - 1];
            if (!preflight_is_write(event)) {
                preflight_set_error(error, DEFERRAL_OUTPUT_REPLAY_ERROR);
                return;
            }
            uint8_t bytes[MEMORY_BLOCK_BYTES];
            deferral_output_replay_bytes(event.value, bytes);
#pragma unroll
            for (size_t i = 0; i < DIGEST_SIZE; i++) state[i] = Fp(bytes[i]);
        }
#pragma unroll
        for (size_t i = 0; i < DIGEST_SIZE; i++) state[DIGEST_SIZE + i] = capacity[i];
        poseidon2::generate_trace_row_for_perm(
            DeferralPoseidonRow::null(), RowSlice(state, 1)
        );
        bool is_last = section + 1 == call.num_rows;
        size_t result_offset = is_last ? 0 : DIGEST_SIZE;
#pragma unroll
        for (size_t i = 0; i < DIGEST_SIZE; i++) {
            Fp value = state[result_offset + i];
            capacity[i] = value;
        }
        RowSlice row(trace + static_cast<size_t>(call.row_start) + section, height);
        COL_WRITE_ARRAY(row, DeferralOutputCols, poseidon2_res, capacity);
    }
}

__global__ void deferral_output_replay_tracegen(
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
    const DeferralOutputReplayCall *calls,
    size_t rows_used,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t byte_pointer_bits,
    uint32_t *count_ptr,
    size_t num_def_circuits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    uint32_t *bitwise_ptr,
    size_t address_bits,
    FpArray<16> *poseidon2_records,
    DeferralPoseidon2Count *poseidon2_counts,
    uint32_t *poseidon2_idx,
    size_t poseidon2_capacity,
    uint32_t *error
) {
    size_t row_idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (row_idx >= height) return;
    if (row_idx >= rows_used) return;

    size_t low = 0;
    size_t high = num_steps;
    while (low + 1 < high) {
        size_t mid = low + (high - low) / 2;
        if (calls[mid].row_start <= row_idx) low = mid;
        else high = mid;
    }
    size_t call_idx = low;
    auto const call = calls[call_idx];
    uint32_t section_idx = static_cast<uint32_t>(row_idx - call.row_start);
    if (section_idx >= call.num_rows) {
        preflight_set_error(error, DEFERRAL_OUTPUT_REPLAY_ERROR);
        return;
    }
    RowSlice row(trace + row_idx, height);
    uint32_t instruction_idx;
    uint32_t event_start;
    uint32_t output_len;
    if (!deferral_output_resolve_step(
            instructions,
            pc_base,
            program,
            memory,
            steps,
            step_start + call_idx,
            expected_opcode,
            register_as,
            memory_as,
            num_def_circuits,
            &instruction_idx,
            &event_start,
            &output_len,
            error
        )) return;
    auto const &instruction = instructions[instruction_idx];
    auto const &step = steps[step_start + call_idx];
    auto const from = program[step.program_index];
    uint32_t rd_ptr = instruction.words[1];
    uint32_t rs_ptr = instruction.words[2];
    uint32_t deferral_idx = instruction.words[3];
    uint64_t domain_end = byte_pointer_bits < 32 ? (uint64_t(1) << byte_pointer_bits)
                                                : (uint64_t(1) << 32);

    ReplayPreviousValue rd_previous;
    ReplayPreviousValue rs_previous;
    if (!deferral_output_replay_event(
            event_start,
            from.timestamp,
            register_as,
            rd_ptr / 2,
            false,
            memory,
            seeds,
            predecessors,
            rd_previous,
            error
        ) ||
        !deferral_output_replay_event(
            event_start + 1,
            from.timestamp + 1,
            register_as,
            rs_ptr / 2,
            false,
            memory,
            seeds,
            predecessors,
            rs_previous,
            error
        )) return;
    uint32_t output_ptr = deferral_output_replay_u32(memory[event_start].value);
    uint32_t input_ptr = deferral_output_replay_u32(memory[event_start + 1].value);
    uint8_t rd_bytes[MEMORY_BLOCK_BYTES];
    uint8_t rs_bytes[MEMORY_BLOCK_BYTES];
    deferral_output_replay_bytes(memory[event_start].value, rd_bytes);
    deferral_output_replay_bytes(memory[event_start + 1].value, rs_bytes);
    if (memory[event_start].value[2] != 0 || memory[event_start].value[3] != 0 ||
        memory[event_start + 1].value[2] != 0 ||
        memory[event_start + 1].value[3] != 0 ||
        (output_ptr & (MEMORY_BLOCK_BYTES - 1)) != 0 ||
        (input_ptr & (MEMORY_BLOCK_BYTES - 1)) != 0 || output_ptr >= domain_end ||
        input_ptr >= domain_end || static_cast<uint64_t>(output_ptr) + output_len > domain_end ||
        static_cast<uint64_t>(input_ptr) + OUTPUT_TOTAL_BYTES > domain_end) {
        preflight_set_error(error, DEFERRAL_OUTPUT_REPLAY_ERROR);
        return;
    }

    uint32_t key_previous_timestamps[OUTPUT_TOTAL_MEMORY_OPS];
    uint8_t output_key[OUTPUT_TOTAL_BYTES];
#pragma unroll
    for (size_t i = 0; i < OUTPUT_TOTAL_MEMORY_OPS; i++) {
        ReplayPreviousValue previous;
        if (!deferral_output_replay_event(
                event_start + 2 + i,
                from.timestamp + 2 + i,
                memory_as,
                input_ptr / 2 + i * BLOCK_FE_WIDTH,
                false,
                memory,
                seeds,
                predecessors,
                previous,
                error
            )) return;
        key_previous_timestamps[i] = previous.timestamp;
        deferral_output_replay_bytes(
            memory[event_start + 2 + i].value, output_key + i * MEMORY_BLOCK_BYTES
        );
    }
    uint32_t encoded_len = static_cast<uint32_t>(output_key[COMMIT_NUM_BYTES]) |
                           (static_cast<uint32_t>(output_key[COMMIT_NUM_BYTES + 1]) << 8) |
                           (static_cast<uint32_t>(output_key[COMMIT_NUM_BYTES + 2]) << 16) |
                           (static_cast<uint32_t>(output_key[COMMIT_NUM_BYTES + 3]) << 24);
    if (encoded_len != output_len) {
        preflight_set_error(error, DEFERRAL_OUTPUT_REPLAY_ERROR);
        return;
    }

    bool is_first = section_idx == 0;
    bool is_last = section_idx + 1 == call.num_rows;
    Histogram count_buffer(count_ptr, num_def_circuits);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);
    MemoryAuxColsFactory mem_helper(
        VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
    );
    BitwiseOperationLookup bitwise_buffer(bitwise_ptr);
    DeferralPoseidon2Buffer poseidon2_buffer(
        poseidon2_records, poseidon2_counts, poseidon2_idx, poseidon2_capacity
    );

    COL_WRITE_VALUE(row, DeferralOutputCols, is_valid, Fp::one());
    COL_WRITE_VALUE(row, DeferralOutputCols, is_first, is_first);
    COL_WRITE_VALUE(row, DeferralOutputCols, is_last, is_last);
    COL_WRITE_VALUE(row, DeferralOutputCols, section_idx, section_idx);
    COL_WRITE_VALUE(row, DeferralOutputCols, from_state.pc, from.pc);
    COL_WRITE_VALUE(row, DeferralOutputCols, from_state.timestamp, from.timestamp);
    COL_WRITE_VALUE(row, DeferralOutputCols, rd_ptr, rd_ptr);
    COL_WRITE_VALUE(row, DeferralOutputCols, rs_ptr, rs_ptr);
    COL_WRITE_VALUE(row, DeferralOutputCols, deferral_idx, deferral_idx);
    COL_WRITE_ARRAY(row, DeferralOutputCols, rd_val, rd_bytes);
    COL_WRITE_ARRAY(row, DeferralOutputCols, rs_val, rs_bytes);
    COL_WRITE_ARRAY(row, DeferralOutputCols, output_commit, output_key);
    COL_WRITE_ARRAY(
        row, DeferralOutputCols, output_len, output_key + COMMIT_NUM_BYTES
    );

    if (is_first) {
        count_buffer.add_count(deferral_idx);
        uint32_t limb_shift_bits = BYTE_BITS * WORD_NUM_LIMBS - address_bits;
        bitwise_buffer.add_range(static_cast<uint32_t>(rd_bytes[3]) << limb_shift_bits,
                                 static_cast<uint32_t>(rs_bytes[3]) << limb_shift_bits);
#pragma unroll
        for (size_t i = 0; i < WORD_NUM_LIMBS; i += 2) {
            bitwise_buffer.add_range(rd_bytes[i], rd_bytes[i + 1]);
            bitwise_buffer.add_range(rs_bytes[i], rs_bytes[i + 1]);
        }
#pragma unroll
        for (size_t i = 0; i < COMMIT_NUM_BYTES; i += 2)
            bitwise_buffer.add_range(output_key[i], output_key[i + 1]);
#pragma unroll
        for (size_t i = 0; i < F_NUM_BYTES; i += 2)
            bitwise_buffer.add_range(output_key[COMMIT_NUM_BYTES + i],
                                     output_key[COMMIT_NUM_BYTES + i + 1]);
        bitwise_buffer.add_range(output_key[COMMIT_NUM_BYTES + F_NUM_BYTES - 1]
                                     << limb_shift_bits, 0);
        mem_helper.fill(row.slice_from(COL_INDEX(DeferralOutputCols, rd_aux)),
                        rd_previous.timestamp, from.timestamp);
        mem_helper.fill(row.slice_from(COL_INDEX(DeferralOutputCols, rs_aux)),
                        rs_previous.timestamp, from.timestamp + 1);
        constexpr size_t read_stride = sizeof(MemoryReadAuxCols<uint8_t>);
#pragma unroll
        for (size_t i = 0; i < OUTPUT_TOTAL_MEMORY_OPS; i++)
            mem_helper.fill(row.slice_from(COL_INDEX(DeferralOutputCols,
                                                     output_commit_and_len_aux) + i * read_stride),
                            key_previous_timestamps[i], from.timestamp + 2 + i);
        uint32_t output_commit_rcs[DIGEST_SIZE];
#pragma unroll
        for (size_t i = 0; i < DIGEST_SIZE; i++) {
            CanonicityAuxCols<Fp> aux;
            Fp x_le[F_NUM_BYTES];
#pragma unroll
            for (size_t j = 0; j < F_NUM_BYTES; j++)
                x_le[j] = Fp(output_key[i * F_NUM_BYTES + j]);
            output_commit_rcs[i] = generate_subrow(x_le, aux);
            write_canonicity_aux(row, COL_INDEX(DeferralOutputCols, output_commit_lt_aux), i, aux);
        }
#pragma unroll
        for (size_t i = 0; i < DIGEST_SIZE; i += 2)
            bitwise_buffer.add_range(output_commit_rcs[i], output_commit_rcs[i + 1]);
        {
            CanonicityAuxCols<Fp> aux;
            Fp x_le[F_NUM_BYTES];
#pragma unroll
            for (size_t j = 0; j < F_NUM_BYTES; j++)
                x_le[j] = Fp(output_key[COMMIT_NUM_BYTES + j]);
            uint32_t output_len_rc = generate_subrow(x_le, aux);
            write_canonicity_aux(row, COL_INDEX(DeferralOutputCols, output_len_lt_aux), 0, aux);
            bitwise_buffer.add_range(output_len_rc, 0);
        }
        Fp sponge_inputs[DIGEST_SIZE] = {};
        sponge_inputs[0] = Fp(deferral_idx);
        sponge_inputs[1] = Fp(output_len);
        COL_WRITE_ARRAY(row, DeferralOutputCols, sponge_inputs, sponge_inputs);

        // Block-index range-check counts for the heap `input` (rs_val) and `output` (rd_val)
        // base byte pointers. Mirrors the first-row branch of the host `DeferralOutputFiller`.
        add_block_index_range_checks(range_checker, input_ptr, address_bits);
        add_block_index_range_checks(range_checker, output_ptr, address_bits);

        // The write block index is unconstrained on the first row (its constraints are gated by
        // `is_write_row`); match the host trace, which leaves it zero.
        COL_WRITE_VALUE(row, DeferralOutputCols, write_block_index, Fp::zero());
    } else {
        COL_FILL_ZERO(row, DeferralOutputCols, rd_aux);
        COL_FILL_ZERO(row, DeferralOutputCols, rs_aux);
        COL_FILL_ZERO(row, DeferralOutputCols, output_commit_and_len_aux);
        COL_FILL_ZERO(row, DeferralOutputCols, output_commit_lt_aux);
        COL_FILL_ZERO(row, DeferralOutputCols, output_len_lt_aux);
        size_t write_idx = event_start + 7 + section_idx - 1;
        ReplayPreviousValue write_previous;
        if (!deferral_output_replay_event(
                write_idx,
                from.timestamp + 7 + section_idx - 1,
                memory_as,
                output_ptr / 2 + (section_idx - 1) * BLOCK_FE_WIDTH,
                true,
                memory,
                seeds,
                predecessors,
                write_previous,
                error
            )) return;
        uint8_t write_bytes[MEMORY_BLOCK_BYTES];
        deferral_output_replay_bytes(memory[write_idx].value, write_bytes);
        COL_WRITE_ARRAY(row, DeferralOutputCols, sponge_inputs, write_bytes);
        for (size_t i = 0; i < DIGEST_SIZE; i += 2)
            bitwise_buffer.add_range(write_bytes[i], write_bytes[i + 1]);
        RowSlice aux = row.slice_from(COL_INDEX(DeferralOutputCols, write_bytes_aux));
        Fp packed_previous[BLOCK_FE_WIDTH];
        uint8_t previous_bytes[MEMORY_BLOCK_BYTES];
        deferral_output_replay_bytes(write_previous.value, previous_bytes);
        pack_u8_block_bytes(packed_previous, previous_bytes);
        COL_WRITE_ARRAY(aux, MemoryWriteAuxColsDef, prev_data, packed_previous);
        mem_helper.fill(aux, write_previous.timestamp,
                        from.timestamp + 7 + section_idx - 1);

        // Memory-bus block index of this row's output write. Mirrors the write-row branch of the
        // host `DeferralOutputFiller`.
        const uint32_t write_byte_ptr = output_ptr + (section_idx - 1) * DIGEST_SIZE;
        COL_WRITE_VALUE(
            row, DeferralOutputCols, write_block_index, Fp(write_byte_ptr / MEMORY_BLOCK_BYTES)
        );
    }

    Fp prev_capacity[DIGEST_SIZE] = {};
    if (!is_first) {
#pragma unroll
        for (size_t i = 0; i < DIGEST_SIZE; i++)
            prev_capacity[i] = trace[(row_idx - 1) +
                                     (COL_INDEX(DeferralOutputCols, poseidon2_res) + i) * height];
    }
    poseidon2_buffer.record(row.slice_from(COL_INDEX(DeferralOutputCols, sponge_inputs)),
                            RowSlice(prev_capacity, 1), is_last);
}

extern "C" int _deferral_output_replay_count_rows(
    uint32_t *d_counts,
    DeviceBufferConstView<RvrReplayInstruction> d_instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> d_program,
    DeviceBufferConstView<PreflightMemoryEvent> d_memory,
    DeviceBufferConstView<RvrReplayStep> d_steps,
    size_t step_start,
    size_t num_steps,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t num_def_circuits,
    uint32_t *d_error,
    cudaStream_t stream
) {
    auto [grid, block] = kernel_launch_params(num_steps, 128);
    deferral_output_replay_count_rows<<<grid, block, 0, stream>>>(
        d_counts, d_instructions, pc_base, d_program, d_memory, d_steps,
        step_start, num_steps, expected_opcode, register_as, memory_as,
        num_def_circuits, d_error);
    return CHECK_KERNEL();
}

extern "C" int _deferral_output_replay_tracegen(
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
    const DeferralOutputReplayCall *d_calls,
    size_t rows_used,
    uint32_t expected_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t byte_pointer_bits,
    uint32_t *d_count,
    size_t num_def_circuits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    uint32_t *d_bitwise,
    size_t address_bits,
    Fp *d_poseidon2_records,
    DeferralPoseidon2Count *d_poseidon2_counts,
    uint32_t *d_poseidon2_idx,
    size_t poseidon2_capacity,
    uint32_t *d_error,
    cudaStream_t stream
) {
    assert(width == sizeof(DeferralOutputCols<uint8_t>));
    assert(d_memory.len() == d_predecessors.len());
    auto [poseidon_grid, poseidon_block] = kernel_launch_params(num_steps, 64);
    deferral_output_replay_poseidon<<<poseidon_grid, poseidon_block, 0, stream>>>(
        d_trace, height, d_instructions, pc_base, d_program, d_memory, d_steps,
        step_start, num_steps, d_calls, expected_opcode, register_as, memory_as,
        num_def_circuits, d_error);
    int result = CHECK_KERNEL();
    if (result != 0) return result;
    auto [grid, block] = kernel_launch_params(height, 128);
    deferral_output_replay_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_instructions, pc_base, d_program, d_memory, d_seeds,
        d_predecessors, d_steps, step_start, num_steps, d_calls, rows_used,
        expected_opcode, register_as, memory_as, byte_pointer_bits,
        d_count, num_def_circuits, d_range_checker, range_checker_num_bins,
        timestamp_max_bits, d_bitwise, address_bits,
        reinterpret_cast<FpArray<16> *>(d_poseidon2_records), d_poseidon2_counts,
        d_poseidon2_idx, poseidon2_capacity / 16, d_error);
    return CHECK_KERNEL();
}
