#include "arch/rvr/preflight.cuh"
#include "launcher.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cstddef>
#include <cstdint>

namespace {

static constexpr uint32_t ERROR_MEMORY_TIMESTAMPS = 101;
static constexpr uint32_t ERROR_SORT_ORDER = 103;
static constexpr uint32_t ERROR_MISSING_FIRST_WRITE_SEED = 104;
static constexpr uint32_t ERROR_DUPLICATE_SEED = 105;
static constexpr uint32_t ERROR_UNUSED_SEED = 106;
static constexpr uint32_t ERROR_MEMORY_ADDRESS = 107;

__device__ bool compact_block_key(
    uint32_t address_space,
    uint32_t pointer,
    uint32_t address_space_offset,
    uint32_t address_space_height,
    uint32_t pointer_max_bits,
    uint32_t &out
) {
    uint64_t address_space_limit =
        static_cast<uint64_t>(address_space_offset) + (uint64_t{1} << address_space_height);
    uint64_t pointer_limit = uint64_t{1} << pointer_max_bits;
    if (address_space < address_space_offset || address_space >= address_space_limit ||
        pointer >= pointer_limit || pointer % 4 != 0) {
        return false;
    }
    uint32_t block_pointer_bits = pointer_max_bits - 2;
    out = ((address_space - address_space_offset) << block_pointer_bits) | (pointer >> 2);
    return true;
}

// The low 32 bits are the source ordinal. Seeds occupy the ordinal prefix, so
// for one block they sort before all chronological memory events. Including
// the ordinal also makes every key unique; sort stability is not required.
__global__ void prepare_entries(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    uint32_t address_space_offset,
    uint32_t address_space_height,
    uint32_t pointer_max_bits,
    uint64_t *keys,
    uint32_t *error
) {
    size_t ordinal = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (ordinal < seeds.len()) {
        auto const &seed = seeds[ordinal];
        uint32_t compact_key;
        if ((seed.address_space & PREFLIGHT_WRITE_BIT) != 0 ||
            !compact_block_key(
                seed.address_space,
                seed.pointer,
                address_space_offset,
                address_space_height,
                pointer_max_bits,
                compact_key
            )) {
            preflight_set_error(error, ERROR_MEMORY_ADDRESS);
            compact_key = 0;
        }
        keys[ordinal] = (static_cast<uint64_t>(compact_key) << 32) | ordinal;
        return;
    }
    size_t event_index = ordinal - seeds.len();
    if (event_index >= memory.len()) return;
    auto const &event = memory[event_index];
    uint32_t compact_key;
    if (!compact_block_key(
            preflight_address_space(event),
            event.pointer,
            address_space_offset,
            address_space_height,
            pointer_max_bits,
            compact_key
        )) {
        preflight_set_error(error, ERROR_MEMORY_ADDRESS);
        compact_key = 0;
    }
    keys[ordinal] = (static_cast<uint64_t>(compact_key) << 32) | ordinal;
    if (event_index != 0 && memory[event_index - 1].timestamp >= event.timestamp) {
        preflight_set_error(error, ERROR_MEMORY_TIMESTAMPS);
    }
}

__global__ void scatter_predecessors(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    size_t num_seeds,
    uint64_t const *sorted_keys,
    size_t num_entries,
    uint32_t *predecessors,
    uint32_t *error
) {
    size_t sorted_pos = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (sorted_pos >= num_entries) return;
    uint32_t ordinal = static_cast<uint32_t>(sorted_keys[sorted_pos]);
    uint32_t block_key = static_cast<uint32_t>(sorted_keys[sorted_pos] >> 32);

    if (ordinal < num_seeds) {
        bool duplicate = sorted_pos + 1 < num_entries &&
                         block_key == static_cast<uint32_t>(sorted_keys[sorted_pos + 1] >> 32) &&
                         static_cast<uint32_t>(sorted_keys[sorted_pos + 1]) < num_seeds;
        if (duplicate) {
            preflight_set_error(error, ERROR_DUPLICATE_SEED);
            return;
        }
        bool followed_by_event = sorted_pos + 1 < num_entries &&
                                 block_key == static_cast<uint32_t>(sorted_keys[sorted_pos + 1] >> 32) &&
                                 static_cast<uint32_t>(sorted_keys[sorted_pos + 1]) >= num_seeds;
        if (!followed_by_event) {
            preflight_set_error(error, ERROR_UNUSED_SEED);
            return;
        }
        size_t first_event_index = static_cast<uint32_t>(sorted_keys[sorted_pos + 1]) - num_seeds;
        if (!preflight_is_write(memory[first_event_index])) {
            preflight_set_error(error, ERROR_UNUSED_SEED);
        }
        return;
    }

    size_t event_index = ordinal - num_seeds;
    auto const &event = memory[event_index];
    bool same_as_previous = sorted_pos != 0 &&
                            static_cast<uint32_t>(sorted_keys[sorted_pos - 1] >> 32) == block_key;
    if (!same_as_previous) {
        if (preflight_is_write(event)) {
            preflight_set_error(error, ERROR_MISSING_FIRST_WRITE_SEED);
            return;
        }
        predecessors[event_index] = 0;
        return;
    }

    uint32_t previous_ordinal = static_cast<uint32_t>(sorted_keys[sorted_pos - 1]);
    if (previous_ordinal < num_seeds) {
        if (!preflight_is_write(event)) {
            predecessors[event_index] = 0;
            return;
        }
        predecessors[event_index] = MEMORY_PREDECESSOR_SEED_BIT | previous_ordinal;
        return;
    }

    size_t previous_event_index = previous_ordinal - num_seeds;
    if (previous_event_index >= event_index) {
        preflight_set_error(error, ERROR_SORT_ORDER);
        return;
    }
    predecessors[event_index] = static_cast<uint32_t>(previous_event_index + 1);
}

} // namespace

extern "C" int _rvr_memory_index_get_temp_bytes(
    size_t num_entries,
    size_t *h_temp_bytes_out,
    cudaStream_t stream
) {
    size_t temp_bytes = 0;
    if (num_entries != 0) {
        cub::DeviceRadixSort::SortKeys(
            nullptr,
            temp_bytes,
            static_cast<uint64_t *>(nullptr),
            static_cast<uint64_t *>(nullptr),
            num_entries,
            0,
            64,
            stream
        );
    }
    *h_temp_bytes_out = temp_bytes;
    return CHECK_KERNEL();
}

extern "C" int _rvr_memory_index(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<PreflightInitialWrite> seeds,
    uint32_t address_space_offset,
    uint32_t address_space_height,
    uint32_t pointer_max_bits,
    uint64_t *keys_in,
    uint64_t *keys_out,
    uint32_t *predecessors,
    void *temp_storage,
    size_t temp_storage_bytes,
    uint32_t *error,
    cudaStream_t stream
) {
    size_t num_entries = memory.len() + seeds.len();
    if (num_entries == 0) return 0;

    auto [grid, block] = kernel_launch_params(num_entries);
    prepare_entries<<<grid, block, 0, stream>>>(
        memory,
        seeds,
        address_space_offset,
        address_space_height,
        pointer_max_bits,
        keys_in,
        error
    );
    if (int err = CHECK_KERNEL(); err) return err;
    if (cudaError_t err = cub::DeviceRadixSort::SortKeys(
            temp_storage,
            temp_storage_bytes,
            keys_in,
            keys_out,
            num_entries,
            0,
            64,
            stream
        );
        err != cudaSuccess) {
        return err;
    }
    scatter_predecessors<<<grid, block, 0, stream>>>(
        memory,
        seeds.len(),
        keys_out,
        num_entries,
        predecessors,
        error
    );
    return CHECK_KERNEL();
}

namespace {

static constexpr uint32_t ERROR_PROGRAM_START = 111;
static constexpr uint32_t ERROR_PROGRAM_PC = 112;
static constexpr uint32_t ERROR_PROGRAM_TIMESTAMP = 113;
static constexpr uint32_t ERROR_TERMINATE_SCHEDULE = 114;
static constexpr uint32_t ERROR_ENDPOINT = 115;
static constexpr uint32_t ERROR_MEMORY_BOUNDARY = 116;
static constexpr uint32_t ERROR_TIMESTAMP_DOMAIN = 117;

struct RvrOpcodeRange {
    uint32_t start;
    uint32_t end;
};

static_assert(sizeof(RvrOpcodeRange) == 8);

__device__ size_t memory_lower_bound(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    uint32_t timestamp
) {
    size_t left = 0;
    size_t right = memory.len();
    while (left < right) {
        size_t middle = left + (right - left) / 2;
        if (memory[middle].timestamp < timestamp) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return left;
}

__device__ bool resolve_instruction(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    uint32_t pc,
    RvrReplayInstruction const *&instruction
) {
    if (pc < pc_base || (pc - pc_base) % 4 != 0) return false;
    size_t slot = (pc - pc_base) / 4;
    if (slot >= instructions.len() || instructions[slot].words[0] == UINT32_MAX) return false;
    instruction = &instructions[slot];
    return true;
}

__global__ void prepare_program_steps(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    uint32_t timestamp_max_bits,
    uint32_t endpoint_kind,
    uint32_t resume_pc,
    uint32_t final_timestamp,
    uint32_t terminate_opcode,
    uint32_t *opcode_keys,
    RvrReplayStep *steps,
    uint32_t *error
) {
    size_t program_index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    size_t num_steps = program.len() - 1;
    if (program_index >= num_steps) return;

    auto const &from = program[program_index];
    auto const &to = program[program_index + 1];
    if (program_index == 0 && from.timestamp != 1) {
        preflight_set_error(error, ERROR_PROGRAM_START);
    }
    RvrReplayInstruction const *instruction = nullptr;
    if (!resolve_instruction(instructions, pc_base, from.pc, instruction)) {
        preflight_set_error(error, ERROR_PROGRAM_PC);
        return;
    }
    uint32_t opcode = instruction->words[0];
    bool is_terminate = opcode == terminate_opcode;
    if (to.timestamp < from.timestamp || (!is_terminate && to.timestamp == from.timestamp)) {
        preflight_set_error(error, ERROR_PROGRAM_TIMESTAMP);
        return;
    }
    if (is_terminate &&
        (endpoint_kind != 0 || program_index + 1 != num_steps || from.pc != to.pc ||
         from.timestamp != to.timestamp)) {
        preflight_set_error(error, ERROR_TERMINATE_SCHEDULE);
        return;
    }

    size_t memory_start = memory_lower_bound(memory, from.timestamp);
    size_t memory_end = memory_lower_bound(memory, to.timestamp);
    if ((program_index == 0 && memory_start != 0) ||
        (program_index + 1 == num_steps && memory_end != memory.len())) {
        preflight_set_error(error, ERROR_MEMORY_BOUNDARY);
        return;
    }

    if (program_index + 1 == num_steps) {
        if (to.timestamp >= (uint32_t{1} << timestamp_max_bits)) {
            preflight_set_error(error, ERROR_TIMESTAMP_DOMAIN);
            return;
        }
        if (endpoint_kind == 0) {
            if (!is_terminate) {
                preflight_set_error(error, ERROR_ENDPOINT);
                return;
            }
        } else {
            RvrReplayInstruction const *resume_instruction = nullptr;
            if (to.pc != resume_pc || to.timestamp != final_timestamp || is_terminate ||
                !resolve_instruction(instructions, pc_base, to.pc, resume_instruction)) {
                preflight_set_error(error, ERROR_ENDPOINT);
                return;
            }
        }
    }

    opcode_keys[program_index] = opcode;
    steps[program_index] = RvrReplayStep{
        .program_index = static_cast<uint32_t>(program_index),
        .memory_start = static_cast<uint32_t>(memory_start),
    };
}

__global__ void validate_empty_program(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    uint32_t timestamp_max_bits,
    uint32_t endpoint_kind,
    uint32_t resume_pc,
    uint32_t final_timestamp,
    uint32_t *error
) {
    auto const &sentinel = program[0];
    RvrReplayInstruction const *resume_instruction = nullptr;
    if (sentinel.timestamp >= (uint32_t{1} << timestamp_max_bits)) {
        preflight_set_error(error, ERROR_TIMESTAMP_DOMAIN);
    } else if (endpoint_kind != 1 || sentinel.timestamp != 1 || sentinel.pc != resume_pc ||
        sentinel.timestamp != final_timestamp || memory.len() != 0 ||
        !resolve_instruction(instructions, pc_base, sentinel.pc, resume_instruction)) {
        preflight_set_error(error, ERROR_ENDPOINT);
    }
}

__device__ size_t opcode_lower_bound(uint32_t const *keys, size_t len, uint32_t opcode) {
    size_t left = 0;
    size_t right = len;
    while (left < right) {
        size_t middle = left + (right - left) / 2;
        if (keys[middle] < opcode) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return left;
}

__device__ size_t opcode_upper_bound(uint32_t const *keys, size_t len, uint32_t opcode) {
    size_t left = 0;
    size_t right = len;
    while (left < right) {
        size_t middle = left + (right - left) / 2;
        if (keys[middle] <= opcode) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    return left;
}

__global__ void build_opcode_ranges(
    uint32_t const *sorted_keys,
    size_t num_steps,
    DeviceBufferConstView<uint32_t> active_opcodes,
    RvrOpcodeRange *ranges
) {
    size_t opcode_index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (opcode_index >= active_opcodes.len()) return;
    uint32_t opcode = active_opcodes[opcode_index];
    ranges[opcode_index] = RvrOpcodeRange{
        .start = static_cast<uint32_t>(opcode_lower_bound(sorted_keys, num_steps, opcode)),
        .end = static_cast<uint32_t>(opcode_upper_bound(sorted_keys, num_steps, opcode)),
    };
}

} // namespace

extern "C" int _rvr_program_index_get_temp_bytes(
    size_t num_steps,
    size_t *h_temp_bytes_out,
    cudaStream_t stream
) {
    size_t temp_bytes = 0;
    if (num_steps != 0) {
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_bytes,
            static_cast<uint32_t *>(nullptr),
            static_cast<uint32_t *>(nullptr),
            static_cast<RvrReplayStep *>(nullptr),
            static_cast<RvrReplayStep *>(nullptr),
            num_steps,
            0,
            32,
            stream
        );
    }
    *h_temp_bytes_out = temp_bytes;
    return CHECK_KERNEL();
}

extern "C" int _rvr_program_index(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<uint32_t> active_opcodes,
    uint32_t timestamp_max_bits,
    uint32_t endpoint_kind,
    uint32_t resume_pc,
    uint32_t final_timestamp,
    uint32_t terminate_opcode,
    uint32_t *opcode_keys_in,
    uint32_t *opcode_keys_out,
    RvrReplayStep *steps_in,
    RvrReplayStep *steps_out,
    RvrOpcodeRange *ranges,
    void *temp_storage,
    size_t temp_storage_bytes,
    uint32_t *error,
    cudaStream_t stream
) {
    size_t num_steps = program.len() - 1;
    if (num_steps == 0) {
        validate_empty_program<<<1, 1, 0, stream>>>(
            instructions,
            pc_base,
            program,
            memory,
            timestamp_max_bits,
            endpoint_kind,
            resume_pc,
            final_timestamp,
            error
        );
        if (int err = CHECK_KERNEL(); err) return err;
        if (active_opcodes.len() != 0) {
            if (cudaError_t err = cudaMemsetAsync(
                    ranges, 0, 2 * active_opcodes.len() * sizeof(uint32_t), stream
                );
                err != cudaSuccess) {
                return err;
            }
        }
        return 0;
    }
    auto [step_grid, step_block] = kernel_launch_params(num_steps);
    prepare_program_steps<<<step_grid, step_block, 0, stream>>>(
        instructions,
        pc_base,
        program,
        memory,
        timestamp_max_bits,
        endpoint_kind,
        resume_pc,
        final_timestamp,
        terminate_opcode,
        opcode_keys_in,
        steps_in,
        error
    );
    if (int err = CHECK_KERNEL(); err) return err;
    if (cudaError_t err = cub::DeviceRadixSort::SortPairs(
            temp_storage,
            temp_storage_bytes,
            opcode_keys_in,
            opcode_keys_out,
            steps_in,
            steps_out,
            num_steps,
            0,
            32,
            stream
        );
        err != cudaSuccess) {
        return err;
    }
    if (active_opcodes.len() != 0) {
        auto [range_grid, range_block] = kernel_launch_params(active_opcodes.len());
        build_opcode_ranges<<<range_grid, range_block, 0, stream>>>(
            opcode_keys_out, num_steps, active_opcodes, ranges
        );
        if (int err = CHECK_KERNEL(); err) return err;
    }
    return 0;
}
