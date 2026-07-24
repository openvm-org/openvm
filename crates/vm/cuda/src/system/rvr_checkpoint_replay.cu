#include "arch/rvr/preflight.cuh"
#include "launcher.cuh"
#include "primitives/buffer_view.cuh"

template <typename T> struct DeviceBufferView {
    T *ptr;
    size_t size;

    __device__ __host__ __forceinline__ T *data() const { return ptr; }
    __device__ __host__ __forceinline__ size_t len() const { return size / sizeof(T); }
    __device__ __host__ __forceinline__ T &operator[](size_t index) const {
        assert(index < len());
        return ptr[index];
    }
};

namespace {

static constexpr uint32_t BABY_BEAR_ORDER = 2013265921u;
static constexpr uint32_t REGISTER_BYTES = 8;
static constexpr uint32_t NUM_REGISTERS = 32;

static constexpr uint32_t ERROR_BAD_CHUNK = 301;
static constexpr uint32_t ERROR_BAD_PC = 302;
static constexpr uint32_t ERROR_UNSUPPORTED_OPCODE = 303;
static constexpr uint32_t ERROR_BAD_INSTRUCTION = 304;
static constexpr uint32_t ERROR_BAD_LOAD = 305;
static constexpr uint32_t ERROR_BAD_RESIDUAL = 306;
static constexpr uint32_t ERROR_BAD_ANCHOR = 307;
static constexpr uint32_t ERROR_BAD_TERMINATION = 308;
static constexpr uint32_t ERROR_OUTPUT_BOUNDS = 309;
static constexpr uint32_t ERROR_BAD_SEED = 310;

struct RvrCheckpoint {
    uint32_t pc;
    uint32_t timestamp;
    uint32_t retired;
    uint32_t residual_cursor;
    uint64_t regs[31];
};

static_assert(sizeof(RvrCheckpoint) == 264);

struct ReplayState {
    uint32_t pc;
    uint32_t timestamp;
    uint32_t retired;
    uint32_t residual_cursor;
    uint64_t regs[NUM_REGISTERS];
};

__device__ __forceinline__ uint64_t load_u64_le(uint8_t const *bytes) {
    uint64_t value = 0;
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) value |= uint64_t(bytes[i]) << (8 * i);
    return value;
}

__device__ __forceinline__ void u64_to_limbs(uint64_t value, uint16_t (&limbs)[4]) {
#pragma unroll
    for (uint32_t i = 0; i < 4; i++) limbs[i] = uint16_t(value >> (16 * i));
}

__device__ __forceinline__ bool canonical_register(uint32_t pointer) {
    return pointer < NUM_REGISTERS * REGISTER_BYTES && pointer % REGISTER_BYTES == 0;
}

__device__ __forceinline__ void load_initial_state(
    DeviceBufferConstView<uint8_t> initial_registers,
    uint32_t pc,
    uint32_t timestamp,
    ReplayState &state
) {
    state.pc = pc;
    state.timestamp = timestamp;
    state.retired = 0;
    state.residual_cursor = 0;
    state.regs[0] = 0;
#pragma unroll
    for (uint32_t reg = 1; reg < NUM_REGISTERS; reg++) {
        state.regs[reg] = load_u64_le(&initial_registers[reg * 8]);
    }
}

__device__ __forceinline__ void load_checkpoint(RvrCheckpoint const &checkpoint, ReplayState &state) {
    state.pc = checkpoint.pc;
    state.timestamp = checkpoint.timestamp;
    state.retired = checkpoint.retired;
    state.residual_cursor = checkpoint.residual_cursor;
    state.regs[0] = 0;
#pragma unroll
    for (uint32_t reg = 1; reg < NUM_REGISTERS; reg++) state.regs[reg] = checkpoint.regs[reg - 1];
}

__device__ __forceinline__ bool matches_checkpoint(
    ReplayState const &state,
    RvrCheckpoint const &checkpoint
) {
    if (state.pc != checkpoint.pc || state.timestamp != checkpoint.timestamp ||
        state.retired != checkpoint.retired ||
        state.residual_cursor != checkpoint.residual_cursor) return false;
#pragma unroll
    for (uint32_t reg = 1; reg < NUM_REGISTERS; reg++) {
        if (state.regs[reg] != checkpoint.regs[reg - 1]) return false;
    }
    return true;
}

__device__ __forceinline__ RvrReplayInstruction const *resolve_instruction(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    uint32_t pc
) {
    if (pc < pc_base || (pc - pc_base) % 4 != 0) return nullptr;
    size_t index = (pc - pc_base) / 4;
    if (index >= instructions.len() || instructions[index].words[0] == UINT32_MAX) return nullptr;
    return &instructions[index];
}

__device__ __forceinline__ bool validate_addi(
    RvrReplayInstruction const &instruction,
    uint32_t register_as,
    uint32_t immediate_as,
    uint32_t &rd,
    uint32_t &rs1,
    int32_t &immediate
) {
    uint32_t rd_ptr = instruction.words[1];
    uint32_t rs1_ptr = instruction.words[2];
    uint32_t encoded = instruction.words[3];
    uint32_t low11 = encoded & 0x7ff;
    uint32_t sign = (encoded >> 11) & 1;
    if (instruction.words[4] != register_as || instruction.words[5] != immediate_as ||
        !canonical_register(rd_ptr) || rd_ptr == 0 || !canonical_register(rs1_ptr) ||
        encoded != low11 + sign * 0xfff800) return false;
    rd = rd_ptr / REGISTER_BYTES;
    rs1 = rs1_ptr / REGISTER_BYTES;
    immediate = sign ? int32_t(low11) - (1 << 11) : int32_t(low11);
    return true;
}

__device__ __forceinline__ bool validate_bne(
    RvrReplayInstruction const &instruction,
    uint32_t register_as,
    uint32_t &rs1,
    uint32_t &rs2
) {
    uint32_t rs1_ptr = instruction.words[1];
    uint32_t rs2_ptr = instruction.words[2];
    if (instruction.words[4] != register_as || instruction.words[5] != register_as ||
        !canonical_register(rs1_ptr) || !canonical_register(rs2_ptr) ||
        instruction.words[3] >= BABY_BEAR_ORDER) return false;
    rs1 = rs1_ptr / REGISTER_BYTES;
    rs2 = rs2_ptr / REGISTER_BYTES;
    return true;
}

__device__ __forceinline__ bool validate_load_doubleword(
    RvrReplayInstruction const &instruction,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    ReplayState const &state,
    size_t initial_memory_bytes,
    uint32_t &rd,
    uint32_t &rs1,
    uint32_t &address
) {
    uint32_t rd_ptr = instruction.words[1];
    uint32_t rs1_ptr = instruction.words[2];
    uint32_t imm = instruction.words[3];
    uint32_t needs_write = instruction.words[6];
    uint32_t imm_sign = instruction.words[7];
    if (instruction.words[4] != register_as || instruction.words[5] != memory_as ||
        !canonical_register(rd_ptr) || rd_ptr == 0 || !canonical_register(rs1_ptr) ||
        imm > UINT16_MAX || needs_write != 1 || imm_sign > 1) return false;
    rd = rd_ptr / REGISTER_BYTES;
    rs1 = rs1_ptr / REGISTER_BYTES;
    uint64_t base = state.regs[rs1];
    if ((base >> 32) != 0) return false;
    int64_t signed_imm = imm_sign ? int64_t(imm) - (int64_t(1) << 16) : int64_t(imm);
    int64_t effective = int64_t(uint32_t(base)) + signed_imm;
    if (effective < 0 || effective > UINT32_MAX || (effective & 7) != 0) return false;
    uint64_t end = uint64_t(effective) + 8;
    if (end > initial_memory_bytes ||
        (pointer_max_bits < 32 && end > (uint64_t(1) << pointer_max_bits))) return false;
    address = uint32_t(effective);
    return true;
}

__device__ __forceinline__ uint32_t branch_target(uint32_t pc, uint32_t encoded_offset) {
    uint64_t sum = uint64_t(pc) + encoded_offset;
    if (sum >= BABY_BEAR_ORDER) sum -= BABY_BEAR_ORDER;
    return uint32_t(sum);
}

__device__ __forceinline__ void write_event(
    PreflightMemoryEvent &event,
    uint32_t timestamp,
    uint32_t address_space,
    uint32_t pointer,
    bool is_write,
    uint64_t value
) {
    event.timestamp = timestamp;
    event.address_space_and_kind = address_space | (is_write ? PREFLIGHT_WRITE_BIT : 0);
    event.pointer = pointer;
    u64_to_limbs(value, event.value);
}

__device__ bool replay_chunk(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<uint8_t> initial_registers,
    DeviceBufferConstView<uint8_t> initial_memory,
    DeviceBufferConstView<RvrCheckpoint> anchors,
    DeviceBufferConstView<uint64_t> residuals,
    size_t chunk,
    uint32_t addi_opcode,
    uint32_t load_doubleword_opcode,
    uint32_t bne_opcode,
    uint32_t terminate_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t immediate_as,
    uint32_t pointer_max_bits,
    uint32_t initial_pc,
    uint32_t initial_timestamp,
    PreflightProgramEvent *program,
    PreflightMemoryEvent *memory,
    size_t memory_capacity,
    uint32_t memory_start,
    uint32_t &memory_count,
    uint32_t *error
) {
    ReplayState state{};
    if (chunk == 0) {
        if (initial_registers.len() < NUM_REGISTERS * 8) {
            preflight_set_error(error, ERROR_BAD_CHUNK);
            return false;
        }
        load_initial_state(initial_registers, initial_pc, initial_timestamp, state);
    } else {
        load_checkpoint(anchors[chunk - 1], state);
    }
    RvrCheckpoint const &end = anchors[chunk];
    if (end.retired < state.retired || end.residual_cursor < state.residual_cursor) {
        preflight_set_error(error, ERROR_BAD_CHUNK);
        return false;
    }

    uint32_t expected_steps = end.retired - state.retired;
    uint32_t emitted = 0;
    bool terminated = false;
    for (uint32_t local_step = 0; local_step < expected_steps; local_step++) {
        auto instruction = resolve_instruction(instructions, pc_base, state.pc);
        if (instruction == nullptr) {
            preflight_set_error(error, ERROR_BAD_PC);
            return false;
        }
        uint32_t opcode = instruction->words[0];
        if (program != nullptr) program[state.retired] = PreflightProgramEvent{state.pc, state.timestamp};

        if (opcode == addi_opcode) {
            uint32_t rd, rs1;
            int32_t immediate;
            if (!validate_addi(*instruction, register_as, immediate_as, rd, rs1, immediate)) {
                preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                return false;
            }
            uint64_t result = state.regs[rs1] + uint64_t(int64_t(immediate));
            if (memory != nullptr) {
                if (uint64_t(memory_start) + emitted + 2 > memory_capacity) {
                    preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
                    return false;
                }
                write_event(memory[memory_start + emitted], state.timestamp, register_as,
                            instruction->words[2] / 2, false, state.regs[rs1]);
                write_event(memory[memory_start + emitted + 1], state.timestamp + 1, register_as,
                            instruction->words[1] / 2, true, result);
            }
            emitted += 2;
            state.regs[rd] = result;
            state.pc += 4;
            state.timestamp += 2;
        } else if (opcode == load_doubleword_opcode) {
            uint32_t rd, rs1, address;
            if (!validate_load_doubleword(*instruction, register_as, memory_as, pointer_max_bits,
                                          state, initial_memory.len(), rd, rs1, address)) {
                preflight_set_error(error, ERROR_BAD_LOAD);
                return false;
            }
            if (state.residual_cursor >= residuals.len()) {
                preflight_set_error(error, ERROR_BAD_RESIDUAL);
                return false;
            }
            uint64_t value = load_u64_le(&initial_memory[address]);
            if (residuals[state.residual_cursor] != value) {
                preflight_set_error(error, ERROR_BAD_RESIDUAL);
                return false;
            }
            if (memory != nullptr) {
                if (uint64_t(memory_start) + emitted + 3 > memory_capacity) {
                    preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
                    return false;
                }
                write_event(memory[memory_start + emitted], state.timestamp, register_as,
                            instruction->words[2] / 2, false, state.regs[rs1]);
                write_event(memory[memory_start + emitted + 1], state.timestamp + 1, memory_as,
                            address / 2, false, value);
                write_event(memory[memory_start + emitted + 2], state.timestamp + 3, register_as,
                            instruction->words[1] / 2, true, value);
            }
            emitted += 3;
            state.residual_cursor++;
            state.regs[rd] = value;
            state.pc += 4;
            state.timestamp += 4;
        } else if (opcode == bne_opcode) {
            uint32_t rs1, rs2;
            if (!validate_bne(*instruction, register_as, rs1, rs2)) {
                preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                return false;
            }
            if (memory != nullptr) {
                if (uint64_t(memory_start) + emitted + 2 > memory_capacity) {
                    preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
                    return false;
                }
                write_event(memory[memory_start + emitted], state.timestamp, register_as,
                            instruction->words[1] / 2, false, state.regs[rs1]);
                write_event(memory[memory_start + emitted + 1], state.timestamp + 1, register_as,
                            instruction->words[2] / 2, false, state.regs[rs2]);
            }
            emitted += 2;
            state.pc = state.regs[rs1] != state.regs[rs2]
                           ? branch_target(state.pc, instruction->words[3])
                           : state.pc + 4;
            state.timestamp += 2;
        } else if (opcode == terminate_opcode) {
            if (local_step + 1 != expected_steps || chunk + 1 != anchors.len()) {
                preflight_set_error(error, ERROR_BAD_TERMINATION);
                return false;
            }
            terminated = true;
        } else {
            preflight_set_error(error, ERROR_UNSUPPORTED_OPCODE);
            return false;
        }
        state.retired++;
    }

    if ((chunk + 1 == anchors.len()) != terminated) {
        preflight_set_error(error, ERROR_BAD_TERMINATION);
        return false;
    }
    if (!matches_checkpoint(state, end)) {
        preflight_set_error(error, ERROR_BAD_ANCHOR);
        return false;
    }
    memory_count = emitted;
    return true;
}

__global__ void checkpoint_count(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<uint8_t> initial_registers,
    DeviceBufferConstView<uint8_t> initial_memory,
    DeviceBufferConstView<RvrCheckpoint> anchors,
    DeviceBufferConstView<uint64_t> residuals,
    uint32_t addi_opcode,
    uint32_t load_doubleword_opcode,
    uint32_t bne_opcode,
    uint32_t terminate_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t immediate_as,
    uint32_t pointer_max_bits,
    uint32_t initial_pc,
    uint32_t initial_timestamp,
    uint32_t *memory_counts,
    uint32_t *error
) {
    size_t chunk = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (chunk >= anchors.len()) return;
    uint32_t count = 0;
    if (replay_chunk(instructions, pc_base, initial_registers, initial_memory, anchors, residuals,
                     chunk, addi_opcode, load_doubleword_opcode, bne_opcode, terminate_opcode,
                     register_as, memory_as, immediate_as, pointer_max_bits, initial_pc,
                     initial_timestamp, nullptr, nullptr, 0, 0, count, error)) {
        memory_counts[chunk] = count;
    }
}

__global__ void checkpoint_emit(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<uint8_t> initial_registers,
    DeviceBufferConstView<uint8_t> initial_memory,
    DeviceBufferConstView<RvrCheckpoint> anchors,
    DeviceBufferConstView<uint64_t> residuals,
    DeviceBufferConstView<uint32_t> memory_offsets,
    uint32_t addi_opcode,
    uint32_t load_doubleword_opcode,
    uint32_t bne_opcode,
    uint32_t terminate_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t immediate_as,
    uint32_t pointer_max_bits,
    uint32_t initial_pc,
    uint32_t initial_timestamp,
    DeviceBufferView<PreflightProgramEvent> program,
    DeviceBufferView<PreflightMemoryEvent> memory,
    uint32_t *error
) {
    size_t chunk = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (chunk >= anchors.len()) return;
    uint32_t count = 0;
    replay_chunk(instructions, pc_base, initial_registers, initial_memory, anchors, residuals,
                 chunk, addi_opcode, load_doubleword_opcode, bne_opcode, terminate_opcode,
                 register_as, memory_as, immediate_as, pointer_max_bits, initial_pc,
                 initial_timestamp, program.data(), memory.data(), memory.len(),
                 memory_offsets[chunk], count, error);
    if (chunk + 1 == anchors.len()) {
        auto const &final_anchor = anchors[chunk];
        if (final_anchor.retired >= program.len()) {
            preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
        } else {
            program[final_anchor.retired] =
                PreflightProgramEvent{final_anchor.pc, final_anchor.timestamp};
        }
    }
}

__global__ void checkpoint_seed_count(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    uint32_t register_as,
    uint32_t *seed_count,
    uint32_t *error
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t seen = 0;
    uint32_t count = 0;
    for (size_t i = 0; i < memory.len(); i++) {
        auto const &event = memory[i];
        if (preflight_address_space(event) != register_as) continue;
        if (event.pointer % 4 != 0 || event.pointer / 4 >= NUM_REGISTERS) {
            preflight_set_error(error, ERROR_BAD_SEED);
            return;
        }
        uint32_t bit = 1u << (event.pointer / 4);
        if ((seen & bit) == 0 && preflight_is_write(event)) count++;
        seen |= bit;
    }
    *seed_count = count;
}

__global__ void checkpoint_seed_emit(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<uint8_t> initial_registers,
    uint32_t register_as,
    DeviceBufferView<PreflightInitialWrite> seeds,
    uint32_t *error
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t seen = 0;
    uint32_t count = 0;
    for (size_t i = 0; i < memory.len(); i++) {
        auto const &event = memory[i];
        if (preflight_address_space(event) != register_as) continue;
        uint32_t reg = event.pointer / 4;
        uint32_t bit = 1u << reg;
        if ((seen & bit) == 0 && preflight_is_write(event)) {
            if (count >= seeds.len() || reg == 0 || initial_registers.len() < (reg + 1) * 8) {
                preflight_set_error(error, ERROR_BAD_SEED);
                return;
            }
            seeds[count].address_space = register_as;
            seeds[count].pointer = event.pointer;
            u64_to_limbs(load_u64_le(&initial_registers[reg * 8]), seeds[count].initial_value);
            count++;
        }
        seen |= bit;
    }
    if (count != seeds.len()) preflight_set_error(error, ERROR_BAD_SEED);
}

} // namespace

extern "C" int _rvr_checkpoint_count(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<uint8_t> initial_registers,
    DeviceBufferConstView<uint8_t> initial_memory,
    DeviceBufferConstView<RvrCheckpoint> anchors,
    DeviceBufferConstView<uint64_t> residuals,
    uint32_t addi_opcode,
    uint32_t load_doubleword_opcode,
    uint32_t bne_opcode,
    uint32_t terminate_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t immediate_as,
    uint32_t pointer_max_bits,
    uint32_t initial_pc,
    uint32_t initial_timestamp,
    uint32_t *memory_counts,
    uint32_t *error,
    cudaStream_t stream
) {
    if (anchors.len() == 0) return int(cudaErrorInvalidValue);
    auto [grid, block] = kernel_launch_params(anchors.len());
    checkpoint_count<<<grid, block, 0, stream>>>(
        instructions, pc_base, initial_registers, initial_memory, anchors, residuals,
        addi_opcode, load_doubleword_opcode, bne_opcode, terminate_opcode, register_as,
        memory_as, immediate_as, pointer_max_bits, initial_pc, initial_timestamp,
        memory_counts, error
    );
    return CHECK_KERNEL();
}

extern "C" int _rvr_checkpoint_emit(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<uint8_t> initial_registers,
    DeviceBufferConstView<uint8_t> initial_memory,
    DeviceBufferConstView<RvrCheckpoint> anchors,
    DeviceBufferConstView<uint64_t> residuals,
    DeviceBufferConstView<uint32_t> memory_offsets,
    uint32_t addi_opcode,
    uint32_t load_doubleword_opcode,
    uint32_t bne_opcode,
    uint32_t terminate_opcode,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t immediate_as,
    uint32_t pointer_max_bits,
    uint32_t initial_pc,
    uint32_t initial_timestamp,
    DeviceBufferView<PreflightProgramEvent> program,
    DeviceBufferView<PreflightMemoryEvent> memory,
    uint32_t *error,
    cudaStream_t stream
) {
    if (memory_offsets.len() != anchors.len()) return int(cudaErrorInvalidValue);
    auto [grid, block] = kernel_launch_params(anchors.len());
    checkpoint_emit<<<grid, block, 0, stream>>>(
        instructions, pc_base, initial_registers, initial_memory, anchors, residuals,
        memory_offsets, addi_opcode, load_doubleword_opcode, bne_opcode, terminate_opcode,
        register_as, memory_as, immediate_as, pointer_max_bits, initial_pc, initial_timestamp,
        program, memory, error
    );
    return CHECK_KERNEL();
}

extern "C" int _rvr_checkpoint_seed_count(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    uint32_t register_as,
    uint32_t *seed_count,
    uint32_t *error,
    cudaStream_t stream
) {
    checkpoint_seed_count<<<1, 1, 0, stream>>>(memory, register_as, seed_count, error);
    return CHECK_KERNEL();
}

extern "C" int _rvr_checkpoint_seed_emit(
    DeviceBufferConstView<PreflightMemoryEvent> memory,
    DeviceBufferConstView<uint8_t> initial_registers,
    uint32_t register_as,
    DeviceBufferView<PreflightInitialWrite> seeds,
    uint32_t *error,
    cudaStream_t stream
) {
    checkpoint_seed_emit<<<1, 1, 0, stream>>>(memory, initial_registers, register_as, seeds, error);
    return CHECK_KERNEL();
}
