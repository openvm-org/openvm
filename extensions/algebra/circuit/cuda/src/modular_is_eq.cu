#include "launcher.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv-adapters/vec_heap.cuh"
#include "riscv/replay.cuh"
#include "system/memory/params.cuh"

#include <cstddef>
#include <cstdint>

template <typename T, size_t NUM_READS, size_t BLOCKS_PER_READ>
struct Rv64IsEqualModU16AdapterCols {
    ExecutionState<T> from_state;
    T rs_ptr[NUM_READS];
    T rs_val[NUM_READS][RV64_PTR_U16_LIMBS];
    MemoryReadAuxCols<T> rs_read_aux[NUM_READS];
    MemoryReadAuxCols<T> heap_read_aux[NUM_READS][BLOCKS_PER_READ];
    T rd_ptr;
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> writes_aux;
};

template <typename T, size_t READ_LIMBS> struct ModularIsEqualCoreCols {
    T is_valid;
    T is_setup;
    T b[READ_LIMBS];
    T c[READ_LIMBS];
    T cmp_result;
    T eq_marker[READ_LIMBS];
    T lt_marker[READ_LIMBS];
    T b_lt_diff;
    T c_lt_diff;
    T c_lt_mark;
};

static constexpr uint32_t MODULAR_IS_EQ_LOCAL_OPCODE = 6;
static constexpr uint32_t MODULAR_SETUP_IS_EQ_LOCAL_OPCODE = 7;
static constexpr uint32_t MODULAR_IS_EQ_REPLAY_ERROR = 0x4d010001;

static __device__ bool modular_is_eq_canonical_register(uint32_t pointer) {
    return pointer < 32u * 8u && (pointer & 7u) == 0;
}

static __device__ bool modular_is_eq_event(
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
        preflight_set_error(error, MODULAR_IS_EQ_REPLAY_ERROR);
        return false;
    }
    auto const &event = memory[event_index];
    if (event.timestamp != timestamp || preflight_address_space(event) != address_space ||
        event.pointer != pointer || preflight_is_write(event) != is_write ||
        !replay_previous_value(
            event_index, event, predecessors[event_index], memory, seeds, previous
        ) || previous.timestamp >= timestamp) {
        preflight_set_error(error, MODULAR_IS_EQ_REPLAY_ERROR);
        return false;
    }
    return true;
}

template <size_t BLOCKS>
static __device__ uint16_t modular_is_eq_limb(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    size_t heap_event_start,
    size_t read,
    size_t limb
) {
    size_t event_index = heap_event_start + read * BLOCKS + limb / BLOCK_FE_WIDTH;
    return memory[event_index].value[limb % BLOCK_FE_WIDTH];
}

template <size_t BLOCKS, size_t LIMBS>
static __device__ size_t unsigned_less_than(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    size_t heap_event_start,
    size_t read,
    uint16_t const *modulus,
    bool &less
) {
    for (int i = static_cast<int>(LIMBS) - 1; i >= 0; i--) {
        uint16_t value = modular_is_eq_limb<BLOCKS>(memory, heap_event_start, read, i);
        if (value != modulus[i]) {
            less = value < modulus[i];
            return static_cast<size_t>(i);
        }
    }
    less = false;
    return LIMBS;
}

template <size_t BLOCKS, size_t LIMBS>
__global__ void modular_is_eq_replay_tracegen(
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
    uint32_t opcode_base,
    uint32_t register_as,
    uint32_t memory_as,
    uint16_t const *modulus,
    uint32_t *range_checker_counts,
    size_t range_checker_bins,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    using AdapterCols = Rv64IsEqualModU16AdapterCols<uint8_t, 2, BLOCKS>;
    using CoreCols = ModularIsEqualCoreCols<uint8_t, LIMBS>;
    using WriteAuxCols = MemoryWriteAuxCols<uint8_t, BLOCK_FE_WIDTH>;
    constexpr size_t ADAPTER_WIDTH = sizeof(AdapterCols);
    constexpr size_t WIDTH = ADAPTER_WIDTH + sizeof(CoreCols);
    constexpr uint32_t EVENT_COUNT = 2 + 2 * BLOCKS + 1;

    __shared__ uint16_t shared_modulus[LIMBS];
    for (size_t limb = threadIdx.x; limb < LIMBS; limb += blockDim.x) {
        shared_modulus[limb] = modulus[limb];
    }
    __syncthreads();

    size_t row_index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (row_index >= height) return;
    RowSlice row(d_trace + row_index, height);
    row.fill_zero(0, WIDTH);
    if (row_index >= num_steps) return;

    if (step_start > steps.len() || row_index >= steps.len() - step_start) {
        preflight_set_error(error, MODULAR_IS_EQ_REPLAY_ERROR);
        return;
    }
    auto const &step = steps[step_start + row_index];
    size_t program_index = step.program_index;
    if (program_index + 1 >= program.len() || predecessors.len() != memory.len()) {
        preflight_set_error(error, MODULAR_IS_EQ_REPLAY_ERROR);
        return;
    }
    auto const &from = program[program_index];
    auto const &to = program[program_index + 1];
    if (from.pc < pc_base || (from.pc - pc_base) % DEFAULT_PC_STEP != 0 ||
        from.pc > UINT32_MAX - DEFAULT_PC_STEP || from.timestamp > UINT32_MAX - EVENT_COUNT) {
        preflight_set_error(error, MODULAR_IS_EQ_REPLAY_ERROR);
        return;
    }
    size_t instruction_index = (from.pc - pc_base) / DEFAULT_PC_STEP;
    if (instruction_index >= instructions.len()) {
        preflight_set_error(error, MODULAR_IS_EQ_REPLAY_ERROR);
        return;
    }
    auto const &instruction = instructions[instruction_index];
    bool is_setup = instruction.words[0] == opcode_base + MODULAR_SETUP_IS_EQ_LOCAL_OPCODE;
    if ((!is_setup && instruction.words[0] != opcode_base + MODULAR_IS_EQ_LOCAL_OPCODE) ||
        instruction.words[4] != register_as || instruction.words[5] != memory_as ||
        instruction.words[6] != 0 || instruction.words[7] != 0 ||
        !modular_is_eq_canonical_register(instruction.words[1]) ||
        !modular_is_eq_canonical_register(instruction.words[2]) ||
        !modular_is_eq_canonical_register(instruction.words[3]) || pointer_max_bits > 32 ||
        to.pc != from.pc + DEFAULT_PC_STEP ||
        to.timestamp != from.timestamp + EVENT_COUNT) {
        preflight_set_error(error, MODULAR_IS_EQ_REPLAY_ERROR);
        return;
    }

    uint32_t rs_ptr[2] = {instruction.words[2], instruction.words[3]};
    uint32_t rd_ptr = instruction.words[1];
    uint32_t rs_val[2];
    uint32_t rs_prev_timestamp[2];
    uint32_t heap_prev_timestamp[2][BLOCKS];
    size_t cursor = step.memory_start;
    ReplayPreviousValue previous;

    for (size_t read = 0; read < 2; read++, cursor++) {
        if (!modular_is_eq_event(
                cursor,
                from.timestamp + static_cast<uint32_t>(read),
                register_as,
                rs_ptr[read] / 2,
                false,
                memory,
                seeds,
                predecessors,
                previous,
                error
            )) {
            return;
        }
        auto const &event = memory[cursor];
        if (event.value[2] != 0 || event.value[3] != 0) {
            preflight_set_error(error, MODULAR_IS_EQ_REPLAY_ERROR);
            return;
        }
        rs_val[read] = static_cast<uint32_t>(event.value[0]) |
                       (static_cast<uint32_t>(event.value[1]) << U16_BITS);
        rs_prev_timestamp[read] = previous.timestamp;
    }

    uint64_t pointer_limit = pointer_max_bits < 32 ? uint64_t(1) << pointer_max_bits
                                                   : uint64_t(1) << 32;
    for (size_t read = 0; read < 2; read++) {
        if ((rs_val[read] & 1) != 0 ||
            static_cast<uint64_t>(rs_val[read]) + BLOCKS * MEMORY_BLOCK_BYTES > pointer_limit) {
            preflight_set_error(error, MODULAR_IS_EQ_REPLAY_ERROR);
            return;
        }
        for (size_t block = 0; block < BLOCKS; block++, cursor++) {
            uint32_t timestamp = from.timestamp + 2 + read * BLOCKS + block;
            uint32_t pointer = (rs_val[read] + block * MEMORY_BLOCK_BYTES) / U16_CELL_SIZE;
            if (!modular_is_eq_event(
                    cursor,
                    timestamp,
                    memory_as,
                    pointer,
                    false,
                    memory,
                    seeds,
                    predecessors,
                    previous,
                    error
                )) {
                return;
            }
            heap_prev_timestamp[read][block] = previous.timestamp;
        }
    }

    if (!modular_is_eq_event(
            cursor,
            from.timestamp + EVENT_COUNT - 1,
            register_as,
            rd_ptr / 2,
            true,
            memory,
            seeds,
            predecessors,
            previous,
            error
        )) {
        return;
    }
    auto const &write = memory[cursor];
    size_t heap_event_start = step.memory_start + 2;
    bool equal = true;
    for (size_t limb = 0; limb < LIMBS; limb++) {
        equal &= modular_is_eq_limb<BLOCKS>(memory, heap_event_start, 0, limb) ==
                 modular_is_eq_limb<BLOCKS>(memory, heap_event_start, 1, limb);
    }
    if (write.value[0] != static_cast<uint16_t>(equal) || write.value[1] != 0 ||
        write.value[2] != 0 || write.value[3] != 0) {
        preflight_set_error(error, MODULAR_IS_EQ_REPLAY_ERROR);
        return;
    }
    cursor++;
    if (cursor < memory.len() && memory[cursor].timestamp < to.timestamp) {
        preflight_set_error(error, MODULAR_IS_EQ_REPLAY_ERROR);
        return;
    }

    bool b_less;
    bool c_less;
    size_t b_diff = unsigned_less_than<BLOCKS, LIMBS>(
        memory, heap_event_start, 0, shared_modulus, b_less
    );
    size_t c_diff = unsigned_less_than<BLOCKS, LIMBS>(
        memory, heap_event_start, 1, shared_modulus, c_less
    );
    bool setup_b_is_modulus = is_setup && b_diff == LIMBS;
    if ((!is_setup && (!b_less || b_diff == LIMBS)) || !c_less || c_diff == LIMBS ||
        (is_setup && !setup_b_is_modulus)) {
        preflight_set_error(error, MODULAR_IS_EQ_REPLAY_ERROR);
        return;
    }

    VariableRangeChecker range_checker(range_checker_counts, range_checker_bins);
    MemoryAuxColsFactory memory_aux(range_checker, timestamp_max_bits);

    row[offsetof(AdapterCols, from_state) + offsetof(ExecutionState<uint8_t>, pc)] = Fp(from.pc);
    row[offsetof(AdapterCols, from_state) + offsetof(ExecutionState<uint8_t>, timestamp)] =
        Fp(from.timestamp);
    for (size_t read = 0; read < 2; read++) {
        row[offsetof(AdapterCols, rs_ptr) + read] = Fp(rs_ptr[read]);
        row[offsetof(AdapterCols, rs_val) + read * RV64_PTR_U16_LIMBS] =
            Fp(static_cast<uint16_t>(rs_val[read]));
        row[offsetof(AdapterCols, rs_val) + read * RV64_PTR_U16_LIMBS + 1] =
            Fp(static_cast<uint16_t>(rs_val[read] >> U16_BITS));
        memory_aux.fill(
            row.slice_from(offsetof(AdapterCols, rs_read_aux) +
                           read * sizeof(MemoryReadAuxCols<uint8_t>)),
            rs_prev_timestamp[read],
            from.timestamp + static_cast<uint32_t>(read)
        );
        range_checker.add_count(
            ptr_bound_from_high_u16(
                static_cast<uint16_t>(rs_val[read] >> U16_BITS), pointer_max_bits
            ),
            U16_BITS
        );
        for (size_t block = 0; block < BLOCKS; block++) {
            size_t aux_index = read * BLOCKS + block;
            memory_aux.fill(
                row.slice_from(offsetof(AdapterCols, heap_read_aux) +
                               aux_index * sizeof(MemoryReadAuxCols<uint8_t>)),
                heap_prev_timestamp[read][block],
                from.timestamp + 2 + read * BLOCKS + block
            );
        }
    }
    row[offsetof(AdapterCols, rd_ptr)] = Fp(rd_ptr);
    for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
        row[offsetof(AdapterCols, writes_aux) +
            offsetof(WriteAuxCols, prev_data) + limb] =
            Fp(previous.value[limb]);
    }
    memory_aux.fill(
        row.slice_from(offsetof(AdapterCols, writes_aux)),
        previous.timestamp,
        from.timestamp + EVENT_COUNT - 1
    );

    RowSlice core = row.slice_from(ADAPTER_WIDTH);
    core[offsetof(CoreCols, is_valid)] = Fp::one();
    core[offsetof(CoreCols, is_setup)] = Fp(static_cast<uint32_t>(is_setup));
    core[offsetof(CoreCols, cmp_result)] = Fp(static_cast<uint32_t>(equal));
    uint32_t c_mark = b_diff == c_diff ? 1 : 2;
    core[offsetof(CoreCols, c_lt_mark)] = Fp(c_mark);
    uint16_t c_diff_value =
        modular_is_eq_limb<BLOCKS>(memory, heap_event_start, 1, c_diff);
    core[offsetof(CoreCols, c_lt_diff)] = Fp(shared_modulus[c_diff] - c_diff_value);
    if (!is_setup) {
        uint16_t b_diff_value =
            modular_is_eq_limb<BLOCKS>(memory, heap_event_start, 0, b_diff);
        core[offsetof(CoreCols, b_lt_diff)] = Fp(shared_modulus[b_diff] - b_diff_value);
        range_checker.add_count(shared_modulus[b_diff] - b_diff_value - 1, U16_BITS);
        range_checker.add_count(shared_modulus[c_diff] - c_diff_value - 1, U16_BITS);
    }
    for (size_t limb = 0; limb < LIMBS; limb++) {
        uint16_t b = modular_is_eq_limb<BLOCKS>(memory, heap_event_start, 0, limb);
        uint16_t c = modular_is_eq_limb<BLOCKS>(memory, heap_event_start, 1, limb);
        core[offsetof(CoreCols, b) + limb] = Fp(b);
        core[offsetof(CoreCols, c) + limb] = Fp(c);
        if (limb == b_diff) {
            core[offsetof(CoreCols, lt_marker) + limb] = Fp::one();
        } else if (limb == c_diff) {
            core[offsetof(CoreCols, lt_marker) + limb] = Fp(c_mark);
        }
    }
    if (!equal) {
        for (size_t limb = 0; limb < LIMBS; limb++) {
            uint16_t b = modular_is_eq_limb<BLOCKS>(memory, heap_event_start, 0, limb);
            uint16_t c = modular_is_eq_limb<BLOCKS>(memory, heap_event_start, 1, limb);
            if (b != c) {
                core[offsetof(CoreCols, eq_marker) + limb] =
                    inv(Fp(b) - Fp(c));
                break;
            }
        }
    }
}

template <size_t BLOCKS, size_t LIMBS>
static int launch_modular_is_eq_replay(
    Fp *d_trace,
    size_t height,
    size_t width,
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
    uint32_t opcode_base,
    uint32_t register_as,
    uint32_t memory_as,
    uint16_t const *modulus,
    uint32_t *range_checker_counts,
    size_t range_checker_bins,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    using AdapterCols = Rv64IsEqualModU16AdapterCols<uint8_t, 2, BLOCKS>;
    using CoreCols = ModularIsEqualCoreCols<uint8_t, LIMBS>;
    if (width != sizeof(AdapterCols) + sizeof(CoreCols)) return 1;
    auto [grid, block] = kernel_launch_params(height, 256);
    modular_is_eq_replay_tracegen<BLOCKS, LIMBS><<<grid, block, 0, stream>>>(
        d_trace,
        height,
        instructions,
        pc_base,
        program,
        memory,
        seeds,
        predecessors,
        steps,
        step_start,
        num_steps,
        error,
        opcode_base,
        register_as,
        memory_as,
        modulus,
        range_checker_counts,
        range_checker_bins,
        pointer_max_bits,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

extern "C" int _modular_is_eq_replay_tracegen_l4(
    Fp *d_trace,
    size_t height,
    size_t width,
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
    uint32_t opcode_base,
    uint32_t register_as,
    uint32_t memory_as,
    uint16_t const *modulus,
    uint32_t *range_checker_counts,
    size_t range_checker_bins,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    return launch_modular_is_eq_replay<4, 16>(
        d_trace, height, width, instructions, pc_base, program, memory, seeds, predecessors, steps,
        step_start, num_steps, error, opcode_base, register_as, memory_as, modulus,
        range_checker_counts, range_checker_bins, pointer_max_bits, timestamp_max_bits, stream
    );
}

extern "C" int _modular_is_eq_replay_tracegen_l6(
    Fp *d_trace,
    size_t height,
    size_t width,
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
    uint32_t opcode_base,
    uint32_t register_as,
    uint32_t memory_as,
    uint16_t const *modulus,
    uint32_t *range_checker_counts,
    size_t range_checker_bins,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    return launch_modular_is_eq_replay<6, 24>(
        d_trace, height, width, instructions, pc_base, program, memory, seeds, predecessors, steps,
        step_start, num_steps, error, opcode_base, register_as, memory_as, modulus,
        range_checker_counts, range_checker_bins, pointer_max_bits, timestamp_max_bits, stream
    );
}
