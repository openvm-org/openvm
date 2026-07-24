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
static constexpr uint32_t MAX_HINT_WORDS = 1023;

static constexpr uint32_t ERROR_BAD_CHUNK = 301;
static constexpr uint32_t ERROR_BAD_PC = 302;
static constexpr uint32_t ERROR_UNSUPPORTED_OPCODE = 303;
static constexpr uint32_t ERROR_BAD_INSTRUCTION = 304;
static constexpr uint32_t ERROR_BAD_LOAD = 305;
static constexpr uint32_t ERROR_BAD_RESIDUAL = 306;
static constexpr uint32_t ERROR_BAD_ANCHOR = 307;
static constexpr uint32_t ERROR_BAD_TERMINATION = 308;
static constexpr uint32_t ERROR_OUTPUT_BOUNDS = 309;
static constexpr uint32_t ERROR_BAD_ENDPOINT = 310;
static constexpr uint8_t FULL_WRITE_MASK = 0xff;

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

struct RvrCheckpointOpcodeBases {
    uint32_t base_alu;
    uint32_t shift;
    uint32_t less_than;
    uint32_t load_store;
    uint32_t branch_equal;
    uint32_t branch_less_than;
    uint32_t jal_lui;
    uint32_t jalr;
    uint32_t auipc;
    uint32_t mul;
    uint32_t mulh;
    uint32_t divrem;
    uint32_t base_alu_w;
    uint32_t shift_w;
    uint32_t mul_w;
    uint32_t divrem_w;
    uint32_t base_alu_imm;
    uint32_t shift_imm;
    uint32_t less_than_imm;
    uint32_t base_alu_w_imm;
    uint32_t shift_w_imm;
    uint32_t hint_store;
    uint32_t phantom;
    uint32_t terminate;
};

static_assert(sizeof(RvrCheckpointOpcodeBases) == 24 * sizeof(uint32_t));

struct RvrCheckpointAccessSchedule {
    uint32_t first_span;
    uint32_t num_spans;
    uint8_t register_operands[3];
    uint8_t num_register_reads;
    uint8_t effect;
    uint8_t effect_operand;
    uint8_t register_write_source;
    uint8_t register_write_operand;
};

static_assert(sizeof(RvrCheckpointAccessSchedule) == 16);

struct RvrCheckpointAccessSpan {
    uint32_t address_space;
    uint32_t count;
    uint8_t base_register;
    uint8_t count_register;
    uint8_t count_shift;
    uint8_t flags;
};

static_assert(sizeof(RvrCheckpointAccessSpan) == 12);

static constexpr uint32_t NO_SCHEDULE = UINT32_MAX;
static constexpr uint8_t SPAN_WRITE_RESIDUAL = 1;
static constexpr uint8_t SPAN_COUNT_FROM_REGISTER = 2;
static constexpr uint8_t SPAN_WRITE_ZERO = 4;
static constexpr uint8_t EFFECT_NEXT = 0;
static constexpr uint8_t EFFECT_BRANCH_RESIDUAL = 1;
static constexpr uint8_t REGISTER_WRITE_NONE = 0;
static constexpr uint8_t REGISTER_WRITE_ZERO = 1;
static constexpr uint8_t REGISTER_WRITE_RESIDUAL = 2;

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

__device__ __forceinline__ bool opcode_in_family(
    uint32_t opcode, uint32_t base, uint32_t count, uint32_t &local
) {
    local = opcode - base;
    return opcode >= base && local < count;
}

__device__ __forceinline__ bool decode_register(uint32_t pointer, uint32_t &reg) {
    if (!canonical_register(pointer)) return false;
    reg = pointer / REGISTER_BYTES;
    return true;
}

__device__ __forceinline__ bool validate_tail(
    RvrReplayInstruction const &instruction,
    uint32_t expected_d,
    uint32_t expected_e
) {
    return instruction.words[4] == expected_d && instruction.words[5] == expected_e &&
           instruction.words[6] == 0 && instruction.words[7] == 0;
}

__device__ __forceinline__ bool decode_signed_12(uint32_t encoded, int64_t &immediate) {
    uint32_t low = encoded & 0xfff;
    uint32_t sign = low >> 11;
    if (encoded != low + sign * 0xfff000) return false;
    immediate = sign ? int64_t(low) - (int64_t(1) << 12) : int64_t(low);
    return true;
}

__device__ __forceinline__ uint64_t sign_extend_word(uint32_t value) {
    return uint64_t(int64_t(int32_t(value)));
}

__device__ __forceinline__ uint64_t arithmetic_shift_right(uint64_t value, uint32_t shift) {
    return uint64_t(int64_t(value) >> shift);
}

__device__ __forceinline__ uint64_t execute_rr_result(
    uint32_t opcode,
    RvrCheckpointOpcodeBases const &opcodes,
    uint64_t lhs,
    uint64_t rhs,
    bool &matched
) {
    uint32_t local;
    matched = true;
    if (opcode_in_family(opcode, opcodes.base_alu, 5, local)) {
        switch (local) {
        case 0: return lhs + rhs;
        case 1: return lhs - rhs;
        case 2: return lhs ^ rhs;
        case 3: return lhs | rhs;
        default: return lhs & rhs;
        }
    }
    if (opcode_in_family(opcode, opcodes.shift, 3, local)) {
        uint32_t shift = uint32_t(rhs) & 63;
        if (local == 0) return lhs << shift;
        if (local == 1) return lhs >> shift;
        return arithmetic_shift_right(lhs, shift);
    }
    if (opcode_in_family(opcode, opcodes.less_than, 2, local)) {
        return local == 0 ? uint64_t(int64_t(lhs) < int64_t(rhs)) : uint64_t(lhs < rhs);
    }
    if (opcode_in_family(opcode, opcodes.base_alu_w, 2, local)) {
        uint32_t result = local == 0 ? uint32_t(lhs) + uint32_t(rhs)
                                     : uint32_t(lhs) - uint32_t(rhs);
        return sign_extend_word(result);
    }
    if (opcode_in_family(opcode, opcodes.shift_w, 3, local)) {
        uint32_t shift = uint32_t(rhs) & 31;
        uint32_t value = uint32_t(lhs);
        if (local == 0) return sign_extend_word(value << shift);
        if (local == 1) return sign_extend_word(value >> shift);
        return sign_extend_word(uint32_t(int32_t(value) >> shift));
    }
    if (opcode == opcodes.mul) return lhs * rhs;
    if (opcode_in_family(opcode, opcodes.mulh, 3, local)) {
        if (local == 0) return uint64_t(__mul64hi(int64_t(lhs), int64_t(rhs)));
        if (local == 1) return __umul64hi(lhs, rhs) - ((lhs >> 63) ? rhs : 0);
        return __umul64hi(lhs, rhs);
    }
    if (opcode_in_family(opcode, opcodes.divrem, 4, local)) {
        if (local == 0) {
            if (rhs == 0) return UINT64_MAX;
            if (lhs == (uint64_t(1) << 63) && rhs == UINT64_MAX) return lhs;
            return uint64_t(int64_t(lhs) / int64_t(rhs));
        }
        if (local == 1) return rhs == 0 ? UINT64_MAX : lhs / rhs;
        if (local == 2) {
            if (rhs == 0) return lhs;
            if (lhs == (uint64_t(1) << 63) && rhs == UINT64_MAX) return 0;
            return uint64_t(int64_t(lhs) % int64_t(rhs));
        }
        return rhs == 0 ? lhs : lhs % rhs;
    }
    if (opcode == opcodes.mul_w) return sign_extend_word(uint32_t(lhs) * uint32_t(rhs));
    if (opcode_in_family(opcode, opcodes.divrem_w, 4, local)) {
        uint32_t lhs_w = uint32_t(lhs);
        uint32_t rhs_w = uint32_t(rhs);
        if (local == 0) {
            if (rhs_w == 0) return UINT64_MAX;
            if (lhs_w == (uint32_t(1) << 31) && rhs_w == UINT32_MAX) {
                return sign_extend_word(lhs_w);
            }
            return sign_extend_word(uint32_t(int32_t(lhs_w) / int32_t(rhs_w)));
        }
        if (local == 1) {
            return sign_extend_word(rhs_w == 0 ? UINT32_MAX : lhs_w / rhs_w);
        }
        if (local == 2) {
            if (rhs_w == 0) return sign_extend_word(lhs_w);
            if (lhs_w == (uint32_t(1) << 31) && rhs_w == UINT32_MAX) return 0;
            return sign_extend_word(uint32_t(int32_t(lhs_w) % int32_t(rhs_w)));
        }
        return sign_extend_word(rhs_w == 0 ? lhs_w : lhs_w % rhs_w);
    }
    matched = false;
    return 0;
}

__device__ __forceinline__ bool rr_uses_immediate_as(
    uint32_t opcode, RvrCheckpointOpcodeBases const &opcodes
) {
    uint32_t local;
    return opcode == opcodes.mul || opcode_in_family(opcode, opcodes.mulh, 3, local) ||
           opcode_in_family(opcode, opcodes.divrem, 4, local) || opcode == opcodes.mul_w ||
           opcode_in_family(opcode, opcodes.divrem_w, 4, local);
}

__device__ __forceinline__ uint64_t execute_ri_result(
    uint32_t opcode,
    RvrCheckpointOpcodeBases const &opcodes,
    uint64_t lhs,
    uint32_t encoded,
    bool &matched,
    bool &valid
) {
    uint32_t local;
    int64_t immediate;
    matched = true;
    valid = true;
    if (opcode_in_family(opcode, opcodes.base_alu_imm, 4, local)) {
        valid = decode_signed_12(encoded, immediate);
        if (!valid) return 0;
        uint64_t imm = uint64_t(immediate);
        if (local == 0) return lhs + imm;
        if (local == 1) return lhs ^ imm;
        if (local == 2) return lhs | imm;
        return lhs & imm;
    }
    if (opcode_in_family(opcode, opcodes.shift_imm, 3, local)) {
        valid = encoded < 64;
        if (!valid) return 0;
        if (local == 0) return lhs << encoded;
        if (local == 1) return lhs >> encoded;
        return arithmetic_shift_right(lhs, encoded);
    }
    if (opcode_in_family(opcode, opcodes.less_than_imm, 2, local)) {
        valid = decode_signed_12(encoded, immediate);
        if (!valid) return 0;
        return local == 0 ? uint64_t(int64_t(lhs) < immediate)
                          : uint64_t(lhs < uint64_t(immediate));
    }
    if (opcode == opcodes.base_alu_w_imm) {
        valid = decode_signed_12(encoded, immediate);
        if (!valid) return 0;
        return sign_extend_word(uint32_t(lhs) + uint32_t(immediate));
    }
    if (opcode_in_family(opcode, opcodes.shift_w_imm, 3, local)) {
        valid = encoded < 32;
        if (!valid) return 0;
        uint32_t value = uint32_t(lhs);
        if (local == 0) return sign_extend_word(value << encoded);
        if (local == 1) return sign_extend_word(value >> encoded);
        return sign_extend_word(uint32_t(int32_t(value) >> encoded));
    }
    matched = false;
    return 0;
}

struct LoadStoreInstruction {
    bool is_load;
    bool sign_extend;
    uint32_t width;
    uint32_t rd_or_rs2;
    uint32_t rs1;
    uint32_t address_space;
    uint32_t address;
    uint32_t aligned_address;
    uint32_t shift;
    bool crosses;
    bool needs_write;
};

struct HintStoreInstruction {
    bool is_single;
    uint32_t num_words;
    uint32_t mem_ptr_reg;
    uint32_t num_words_reg;
    uint32_t mem_ptr;
};

__device__ __forceinline__ bool validate_hint_store(
    RvrReplayInstruction const &instruction,
    RvrCheckpointOpcodeBases const &opcodes,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    ReplayState const &state,
    size_t initial_memory_bytes,
    HintStoreInstruction &decoded
) {
    uint32_t local;
    if (!opcode_in_family(instruction.words[0], opcodes.hint_store, 2, local) ||
        instruction.words[3] != 0 || instruction.words[4] != register_as ||
        instruction.words[5] != memory_as || instruction.words[6] != 0 ||
        instruction.words[7] != 0 || !decode_register(instruction.words[2], decoded.mem_ptr_reg)) {
        return false;
    }
    decoded.is_single = local == 0;
    if (decoded.is_single) {
        if (instruction.words[1] != 0) return false;
        decoded.num_words_reg = 0;
        decoded.num_words = 1;
    } else {
        if (!decode_register(instruction.words[1], decoded.num_words_reg)) return false;
        uint64_t num_words = state.regs[decoded.num_words_reg];
        if (num_words == 0 || num_words > MAX_HINT_WORDS) return false;
        decoded.num_words = uint32_t(num_words);
    }

    uint64_t mem_ptr = state.regs[decoded.mem_ptr_reg];
    if ((mem_ptr >> 32) != 0 || (mem_ptr & (REGISTER_BYTES - 1)) != 0) return false;
    uint64_t end = mem_ptr + uint64_t(decoded.num_words) * REGISTER_BYTES;
    if (end < mem_ptr || end > initial_memory_bytes ||
        (pointer_max_bits < 32 && end > (uint64_t(1) << pointer_max_bits))) {
        return false;
    }
    decoded.mem_ptr = uint32_t(mem_ptr);
    return true;
}

__device__ __forceinline__ uint64_t normalize_load_result(
    uint64_t value, uint32_t width, bool sign_extend
) {
    if (width == 8) return value;
    uint32_t bits = width * 8;
    uint64_t mask = (uint64_t(1) << bits) - 1;
    value &= mask;
    if (sign_extend && (value & (uint64_t(1) << (bits - 1))) != 0) value |= ~mask;
    return value;
}

__device__ __forceinline__ bool validate_load_store(
    RvrReplayInstruction const &instruction,
    RvrCheckpointOpcodeBases const &opcodes,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    ReplayState const &state,
    size_t initial_memory_bytes,
    LoadStoreInstruction &decoded
) {
    uint32_t local;
    if (!opcode_in_family(instruction.words[0], opcodes.load_store, 11, local)) return false;
    decoded.is_load = local <= 3 || local >= 8;
    decoded.sign_extend = local >= 8;
    switch (local) {
    case 0:
    case 4: decoded.width = 8; break;
    case 3:
    case 5:
    case 10: decoded.width = 4; break;
    case 2:
    case 6:
    case 9: decoded.width = 2; break;
    case 1:
    case 7:
    case 8: decoded.width = 1; break;
    default: return false;
    }

    uint32_t rd_or_rs2_ptr = instruction.words[1];
    uint32_t rs1_ptr = instruction.words[2];
    uint32_t imm = instruction.words[3];
    uint32_t needs_write = instruction.words[6];
    uint32_t imm_sign = instruction.words[7];
    if (instruction.words[4] != register_as || !canonical_register(rd_or_rs2_ptr) ||
        !canonical_register(rs1_ptr) || imm > UINT16_MAX || imm_sign > 1) return false;
    if (decoded.is_load) {
        if (instruction.words[5] != memory_as ||
            needs_write != uint32_t(rd_or_rs2_ptr != 0)) return false;
    } else if ((instruction.words[5] != memory_as && instruction.words[5] != memory_as + 1) ||
               needs_write != 1) {
        return false;
    }
    decoded.rd_or_rs2 = rd_or_rs2_ptr / REGISTER_BYTES;
    decoded.rs1 = rs1_ptr / REGISTER_BYTES;
    decoded.address_space = instruction.words[5];
    decoded.needs_write = needs_write != 0;
    uint64_t base = state.regs[decoded.rs1];
    if ((base >> 32) != 0) return false;
    int64_t signed_imm = imm_sign ? int64_t(imm) - (int64_t(1) << 16) : int64_t(imm);
    int64_t effective = int64_t(uint32_t(base)) + signed_imm;
    if (effective < 0 || effective > UINT32_MAX) return false;
    decoded.address = uint32_t(effective);
    decoded.shift = decoded.address & 7;
    decoded.aligned_address = decoded.address - decoded.shift;
    decoded.crosses = decoded.shift + decoded.width > 8;
    uint64_t block_end = uint64_t(decoded.aligned_address) + (decoded.crosses ? 16 : 8);
    // The load/store adapter AIRs range check the byte address's high u16 limb to
    // `pointer_max_bits - U16_BITS` bits, committing the byte-address domain
    // `[0, 2^pointer_max_bits)`.
    if ((decoded.is_load && block_end > initial_memory_bytes) ||
        (pointer_max_bits < 32 && block_end > (uint64_t(1) << pointer_max_bits))) {
        return false;
    }
    return true;
}

__device__ __forceinline__ uint32_t branch_target(uint32_t pc, uint32_t encoded_offset) {
    uint64_t sum = uint64_t(pc) + encoded_offset;
    if (sum >= BABY_BEAR_ORDER) sum -= BABY_BEAR_ORDER;
    return uint32_t(sum);
}

__device__ __forceinline__ bool execute_branch_condition(
    uint32_t opcode,
    RvrCheckpointOpcodeBases const &opcodes,
    uint64_t lhs,
    uint64_t rhs,
    bool &matched
) {
    uint32_t local;
    matched = true;
    if (opcode_in_family(opcode, opcodes.branch_equal, 2, local)) {
        return local == 0 ? lhs == rhs : lhs != rhs;
    }
    if (opcode_in_family(opcode, opcodes.branch_less_than, 4, local)) {
        if (local == 0) return int64_t(lhs) < int64_t(rhs);
        if (local == 1) return lhs < rhs;
        if (local == 2) return int64_t(lhs) >= int64_t(rhs);
        return lhs >= rhs;
    }
    matched = false;
    return false;
}

__device__ __forceinline__ bool validate_rr_instruction(
    RvrReplayInstruction const &instruction,
    RvrCheckpointOpcodeBases const &opcodes,
    uint32_t register_as,
    uint32_t immediate_as,
    uint32_t &rd,
    uint32_t &rs1,
    uint32_t &rs2,
    uint64_t &result,
    ReplayState const &state
) {
    if (!decode_register(instruction.words[1], rd) || rd == 0 ||
        !decode_register(instruction.words[2], rs1) ||
        !decode_register(instruction.words[3], rs2)) return false;
    bool matched;
    result = execute_rr_result(
        instruction.words[0], opcodes, state.regs[rs1], state.regs[rs2], matched
    );
    uint32_t expected_e = rr_uses_immediate_as(instruction.words[0], opcodes)
                              ? immediate_as
                              : register_as;
    return matched && validate_tail(instruction, register_as, expected_e);
}

__device__ __forceinline__ bool validate_ri_instruction(
    RvrReplayInstruction const &instruction,
    RvrCheckpointOpcodeBases const &opcodes,
    uint32_t register_as,
    uint32_t immediate_as,
    uint32_t &rd,
    uint32_t &rs1,
    uint64_t &result,
    ReplayState const &state
) {
    if (!decode_register(instruction.words[1], rd) || rd == 0 ||
        !decode_register(instruction.words[2], rs1) ||
        !validate_tail(instruction, register_as, immediate_as)) return false;
    bool matched;
    bool valid;
    result = execute_ri_result(
        instruction.words[0], opcodes, state.regs[rs1], instruction.words[3], matched, valid
    );
    return matched && valid;
}

__device__ __forceinline__ bool validate_branch_instruction(
    RvrReplayInstruction const &instruction,
    RvrCheckpointOpcodeBases const &opcodes,
    uint32_t register_as,
    uint32_t &rs1,
    uint32_t &rs2,
    bool &take,
    ReplayState const &state
) {
    if (!decode_register(instruction.words[1], rs1) ||
        !decode_register(instruction.words[2], rs2) ||
        instruction.words[3] >= BABY_BEAR_ORDER ||
        !validate_tail(instruction, register_as, register_as)) return false;
    bool matched;
    take = execute_branch_condition(
        instruction.words[0], opcodes, state.regs[rs1], state.regs[rs2], matched
    );
    return matched;
}

__device__ __forceinline__ void write_event(
    PreflightMemoryEvent &event,
    uint8_t *write_mask,
    uint32_t timestamp,
    uint32_t address_space,
    uint32_t pointer,
    bool is_write,
    uint64_t value
) {
    event.timestamp = timestamp;
    event.address_space_and_kind = address_space | (is_write ? PREFLIGHT_WRITE_BIT : 0);
    event.pointer = pointer;
    // Reads are unresolved chronology intents. Only writes carry payload,
    // which makes zero outside the write mask a cheap fail-closed invariant.
    u64_to_limbs(is_write ? value : 0, event.value);
    if (write_mask != nullptr) *write_mask = is_write ? FULL_WRITE_MASK : 0;
}

__device__ __forceinline__ void write_memory_intent(
    PreflightMemoryEvent &event,
    uint8_t *write_mask,
    uint32_t timestamp,
    uint32_t address_space,
    uint32_t aligned_address,
    uint64_t bytes,
    uint8_t mask
) {
    event.timestamp = timestamp;
    event.address_space_and_kind = address_space | (mask != 0 ? PREFLIGHT_WRITE_BIT : 0);
    event.pointer = aligned_address / 2;
    u64_to_limbs(bytes, event.value);
    if (write_mask != nullptr) *write_mask = mask;
}

__device__ __forceinline__ bool replay_access_schedule(
    RvrReplayInstruction const &instruction,
    RvrCheckpointAccessSchedule const &schedule,
    DeviceBufferConstView<RvrCheckpointAccessSpan> spans,
    DeviceBufferConstView<uint64_t> residuals,
    DeviceBufferConstView<uint8_t> initial_memory,
    uint32_t register_as,
    uint32_t pointer_max_bits,
    ReplayState &state,
    PreflightMemoryEvent *memory,
    uint8_t *write_masks,
    size_t memory_capacity,
    uint32_t memory_start,
    uint32_t &emitted,
    uint32_t *error
) {
    if (uint64_t(schedule.first_span) + schedule.num_spans > spans.len()) {
        preflight_set_error(error, ERROR_BAD_INSTRUCTION);
        return false;
    }
    if (schedule.effect != EFFECT_NEXT && schedule.effect != EFFECT_BRANCH_RESIDUAL) {
        preflight_set_error(error, ERROR_BAD_INSTRUCTION);
        return false;
    }
    bool has_register_write = schedule.register_write_source != REGISTER_WRITE_NONE;
    if ((schedule.register_write_source != REGISTER_WRITE_NONE &&
         schedule.register_write_source != REGISTER_WRITE_ZERO &&
         schedule.register_write_source != REGISTER_WRITE_RESIDUAL) ||
        (has_register_write ? !(schedule.register_write_operand >= 1 &&
                                schedule.register_write_operand < 8)
                            : schedule.register_write_operand != 0)) {
        preflight_set_error(error, ERROR_BAD_INSTRUCTION);
        return false;
    }
    uint32_t write_register = 0;
    if (has_register_write &&
        !decode_register(instruction.words[schedule.register_write_operand], write_register)) {
        preflight_set_error(error, ERROR_BAD_INSTRUCTION);
        return false;
    }
    bool register_write_enabled = has_register_write && write_register != 0;

    uint64_t register_values[3] = {};
    for (uint32_t index = 0; index < schedule.num_register_reads; index++) {
        uint8_t operand = schedule.register_operands[index];
        uint32_t reg = instruction.words[operand] / REGISTER_BYTES;
        register_values[index] = state.regs[reg];
    }

    // Validate the complete instruction before mutating replay state or output.
    // Count and emit therefore fail at the same boundary, and a malformed later
    // span cannot leave a partial memory schedule behind.
    uint64_t span_events = 0;
    uint64_t write_residuals = 0;
    for (uint32_t span_index = 0; span_index < schedule.num_spans; span_index++) {
        auto const &span = spans[schedule.first_span + span_index];
        constexpr uint8_t KNOWN_SPAN_FLAGS =
            SPAN_WRITE_RESIDUAL | SPAN_COUNT_FROM_REGISTER | SPAN_WRITE_ZERO;
        if ((span.flags & ~KNOWN_SPAN_FLAGS) != 0) {
            preflight_set_error(error, ERROR_BAD_INSTRUCTION);
            return false;
        }
        uint32_t count = span.count;
        if ((span.flags & SPAN_COUNT_FROM_REGISTER) != 0) {
            uint64_t encoded_count = register_values[span.count_register];
            uint64_t low_mask = span.count_shift == 0 ? 0 : (uint64_t(1) << span.count_shift) - 1;
            uint64_t shifted = encoded_count >> span.count_shift;
            if ((encoded_count & low_mask) != 0 || shifted > span.count) {
                preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                return false;
            }
            count = uint32_t(shifted);
        } else if (count == 0) {
            preflight_set_error(error, ERROR_BAD_INSTRUCTION);
            return false;
        }
        if (uint64_t(count) > UINT32_MAX - span_events) {
            preflight_set_error(error, ERROR_BAD_INSTRUCTION);
            return false;
        }
        span_events += count;
        if (count == 0) continue;

        uint64_t base = register_values[span.base_register];
        uint64_t byte_len = uint64_t(count) * REGISTER_BYTES;
        uint64_t end = base + byte_len;
        if ((base >> 32) != 0 || (base & (REGISTER_BYTES - 1)) != 0 || end < base ||
            end > uint64_t(UINT32_MAX) + 1 || end > initial_memory.len() ||
            (pointer_max_bits < 32 && end > (uint64_t(1) << pointer_max_bits))) {
            preflight_set_error(error, ERROR_BAD_INSTRUCTION);
            return false;
        }
        if ((span.flags & SPAN_WRITE_RESIDUAL) != 0 &&
            (span.flags & SPAN_WRITE_ZERO) != 0) {
            preflight_set_error(error, ERROR_BAD_INSTRUCTION);
            return false;
        }
        if ((span.flags & SPAN_WRITE_RESIDUAL) != 0) write_residuals += count;
    }

    uint64_t timestamp_slots = uint64_t(schedule.num_register_reads) + span_events +
                               uint64_t(has_register_write);
    uint64_t total_events = uint64_t(schedule.num_register_reads) + span_events +
                            uint64_t(register_write_enabled);
    if (timestamp_slots > UINT32_MAX - state.timestamp || total_events > UINT32_MAX - emitted) {
        preflight_set_error(error, ERROR_BAD_INSTRUCTION);
        return false;
    }
    if (memory != nullptr && uint64_t(memory_start) + emitted + total_events > memory_capacity) {
        preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
        return false;
    }
    uint64_t register_write_residuals =
        schedule.register_write_source == REGISTER_WRITE_RESIDUAL ? 1u : 0u;
    uint64_t effect_residual_index = uint64_t(state.residual_cursor) + write_residuals +
                                     register_write_residuals;
    uint64_t required_residuals = write_residuals + register_write_residuals +
                                  (schedule.effect == EFFECT_BRANCH_RESIDUAL ? 1u : 0u);
    if (required_residuals > UINT32_MAX - state.residual_cursor ||
        uint64_t(state.residual_cursor) + required_residuals > residuals.len()) {
        preflight_set_error(error, ERROR_BAD_RESIDUAL);
        return false;
    }
    uint64_t effect_residual = 0;
    if (schedule.effect == EFFECT_BRANCH_RESIDUAL) {
        effect_residual = residuals[effect_residual_index];
        if (effect_residual > 1) {
            preflight_set_error(error, ERROR_BAD_RESIDUAL);
            return false;
        }
    }

    for (uint32_t index = 0; index < schedule.num_register_reads; index++) {
        uint8_t operand = schedule.register_operands[index];
        uint32_t reg = instruction.words[operand] / REGISTER_BYTES;
        if (memory != nullptr) {
            uint32_t cursor = memory_start + emitted;
            write_event(memory[cursor], &write_masks[cursor], state.timestamp, register_as,
                        instruction.words[operand] / 2, false, state.regs[reg]);
        }
        emitted++;
        state.timestamp++;
    }

    for (uint32_t span_index = 0; span_index < schedule.num_spans; span_index++) {
        auto const &span = spans[schedule.first_span + span_index];
        uint32_t count = span.count;
        if ((span.flags & SPAN_COUNT_FROM_REGISTER) != 0) {
            count = uint32_t(register_values[span.count_register] >> span.count_shift);
        }
        if (count == 0) continue;

        uint64_t base = register_values[span.base_register];
        bool is_residual_write = (span.flags & SPAN_WRITE_RESIDUAL) != 0;
        bool is_write = is_residual_write || (span.flags & SPAN_WRITE_ZERO) != 0;
        for (uint32_t word = 0; word < count; word++) {
            if (memory != nullptr) {
                uint32_t cursor = memory_start + emitted;
                uint64_t value =
                    is_residual_write ? residuals[state.residual_cursor + word] : 0;
                write_memory_intent(memory[cursor], &write_masks[cursor], state.timestamp,
                                    span.address_space,
                                    uint32_t(base) + REGISTER_BYTES * word, value,
                                    is_write ? FULL_WRITE_MASK : 0);
            }
            emitted++;
            state.timestamp++;
        }
        if (is_residual_write) state.residual_cursor += count;
    }
    if (has_register_write) {
        uint64_t value = schedule.register_write_source == REGISTER_WRITE_RESIDUAL
                             ? residuals[state.residual_cursor]
                             : 0;
        if (register_write_enabled) {
            if (memory != nullptr) {
                uint32_t cursor = memory_start + emitted;
                write_event(memory[cursor], &write_masks[cursor], state.timestamp, register_as,
                            instruction.words[schedule.register_write_operand] / 2, true, value);
            }
            emitted++;
            state.regs[write_register] = value;
        }
        if (schedule.register_write_source == REGISTER_WRITE_RESIDUAL) {
            state.residual_cursor++;
        }
        state.timestamp++;
    }
    if (schedule.effect == EFFECT_BRANCH_RESIDUAL) {
        // All write residuals, if any, precede the terminal control residual.
        state.residual_cursor++;
        state.pc = effect_residual != 0
                       ? branch_target(state.pc, instruction.words[schedule.effect_operand])
                       : state.pc + 4;
    } else {
        state.pc += 4;
    }
    return true;
}

__device__ bool replay_chunk(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<uint8_t> initial_registers,
    DeviceBufferConstView<uint8_t> initial_memory,
    DeviceBufferConstView<RvrCheckpoint> anchors,
    DeviceBufferConstView<uint64_t> residuals,
    DeviceBufferConstView<uint32_t> schedule_dispatch,
    DeviceBufferConstView<RvrCheckpointAccessSchedule> schedules,
    DeviceBufferConstView<RvrCheckpointAccessSpan> spans,
    size_t chunk,
    RvrCheckpointOpcodeBases opcodes,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t immediate_as,
    uint32_t pointer_max_bits,
    uint32_t initial_pc,
    uint32_t initial_timestamp,
    uint32_t endpoint_kind,
    PreflightProgramEvent *program,
    PreflightMemoryEvent *memory,
    uint8_t *write_masks,
    size_t memory_capacity,
    uint32_t memory_start,
    uint32_t &memory_count,
    uint32_t *error
) {
    if (endpoint_kind > 1) {
        preflight_set_error(error, ERROR_BAD_ENDPOINT);
        return false;
    }
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
    // Fixed-arity opcode paths advance `state.timestamp` by at most 4 per step and
    // carry no overflow guard: a wrapped timestamp cannot pass the strictly
    // monotonic per-address chronology check, the host timestamp-domain checks, or
    // the end-anchor comparison. Variable-length paths (hint store, access
    // schedules) bound their own timestamp advance explicitly.
    for (uint32_t local_step = 0; local_step < expected_steps; local_step++) {
        auto instruction = resolve_instruction(instructions, pc_base, state.pc);
        if (instruction == nullptr) {
            preflight_set_error(error, ERROR_BAD_PC);
            return false;
        }
        uint32_t opcode = instruction->words[0];
        if (program != nullptr) program[state.retired] = PreflightProgramEvent{state.pc, state.timestamp};

        uint32_t rd, rs1, rs2;
        uint64_t result;
        if (validate_rr_instruction(
                *instruction, opcodes, register_as, immediate_as, rd, rs1, rs2, result, state
            )) {
            if (memory != nullptr) {
                if (uint64_t(memory_start) + emitted + 3 > memory_capacity) {
                    preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
                    return false;
                }
                write_event(memory[memory_start + emitted], &write_masks[memory_start + emitted],
                            state.timestamp, register_as, instruction->words[2] / 2, false,
                            state.regs[rs1]);
                write_event(memory[memory_start + emitted + 1],
                            &write_masks[memory_start + emitted + 1], state.timestamp + 1,
                            register_as, instruction->words[3] / 2, false, state.regs[rs2]);
                write_event(memory[memory_start + emitted + 2],
                            &write_masks[memory_start + emitted + 2], state.timestamp + 2,
                            register_as, instruction->words[1] / 2, true, result);
            }
            emitted += 3;
            state.regs[rd] = result;
            state.pc += 4;
            state.timestamp += 3;
        } else if (validate_ri_instruction(
                       *instruction, opcodes, register_as, immediate_as, rd, rs1, result, state
                   )) {
            if (memory != nullptr) {
                if (uint64_t(memory_start) + emitted + 2 > memory_capacity) {
                    preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
                    return false;
                }
                write_event(memory[memory_start + emitted], &write_masks[memory_start + emitted],
                            state.timestamp, register_as, instruction->words[2] / 2, false,
                            state.regs[rs1]);
                write_event(memory[memory_start + emitted + 1],
                            &write_masks[memory_start + emitted + 1], state.timestamp + 1,
                            register_as, instruction->words[1] / 2, true, result);
            }
            emitted += 2;
            state.regs[rd] = result;
            state.pc += 4;
            state.timestamp += 2;
        } else if (opcode >= opcodes.hint_store && opcode < opcodes.hint_store + 2) {
            HintStoreInstruction decoded{};
            if (!validate_hint_store(*instruction, opcodes, register_as, memory_as,
                                     pointer_max_bits, state, initial_memory.len(), decoded)) {
                preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                return false;
            }
            uint32_t event_count = decoded.num_words + (decoded.is_single ? 1 : 2);
            if (state.residual_cursor > residuals.len() ||
                size_t(decoded.num_words) > residuals.len() - state.residual_cursor) {
                preflight_set_error(error, ERROR_BAD_RESIDUAL);
                return false;
            }
            if (decoded.num_words > (UINT32_MAX - state.timestamp) / 3) {
                preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                return false;
            }
            if (memory != nullptr) {
                if (uint64_t(memory_start) + emitted + event_count > memory_capacity) {
                    preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
                    return false;
                }
                write_event(memory[memory_start + emitted], &write_masks[memory_start + emitted],
                            state.timestamp, register_as, instruction->words[2] / 2, false,
                            state.regs[decoded.mem_ptr_reg]);
                uint32_t write_start = emitted + 1;
                if (!decoded.is_single) {
                    write_event(memory[memory_start + emitted + 1],
                                &write_masks[memory_start + emitted + 1], state.timestamp + 1,
                                register_as, instruction->words[1] / 2, false,
                                state.regs[decoded.num_words_reg]);
                    write_start++;
                }
                for (uint32_t word = 0; word < decoded.num_words; word++) {
                    size_t residual_index = state.residual_cursor + word;
                    uint32_t event_index = memory_start + write_start + word;
                    write_memory_intent(
                        memory[event_index], &write_masks[event_index],
                        state.timestamp + 2 + 3 * word, memory_as,
                        decoded.mem_ptr + REGISTER_BYTES * word, residuals[residual_index],
                        FULL_WRITE_MASK
                    );
                }
            }
            emitted += event_count;
            state.residual_cursor += decoded.num_words;
            state.pc += 4;
            state.timestamp += 3 * decoded.num_words;
        } else if (opcode >= opcodes.load_store && opcode < opcodes.load_store + 11) {
            LoadStoreInstruction decoded{};
            if (!validate_load_store(*instruction, opcodes, register_as, memory_as,
                                     pointer_max_bits, state, initial_memory.len(), decoded)) {
                preflight_set_error(error, ERROR_BAD_LOAD);
                return false;
            }
            if (decoded.is_load) {
                uint64_t value = 0;
                if (decoded.needs_write) {
                    if (state.residual_cursor >= residuals.len()) {
                        preflight_set_error(error, ERROR_BAD_RESIDUAL);
                        return false;
                    }
                    value = residuals[state.residual_cursor++];
                    if (normalize_load_result(value, decoded.width, decoded.sign_extend) != value) {
                        preflight_set_error(error, ERROR_BAD_RESIDUAL);
                        return false;
                    }
                }
                bool byte_load = decoded.width == 1;
                uint32_t event_count = 2 + uint32_t(decoded.crosses) + uint32_t(decoded.needs_write);
                if (memory != nullptr) {
                    if (uint64_t(memory_start) + emitted + event_count > memory_capacity) {
                        preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
                        return false;
                    }
                    write_event(memory[memory_start + emitted], &write_masks[memory_start + emitted],
                                state.timestamp, register_as, instruction->words[2] / 2, false,
                                state.regs[decoded.rs1]);
                    write_memory_intent(
                        memory[memory_start + emitted + 1],
                        &write_masks[memory_start + emitted + 1], state.timestamp + 1,
                        decoded.address_space, decoded.aligned_address, 0, 0
                    );
                    if (decoded.crosses) {
                        write_memory_intent(
                            memory[memory_start + emitted + 2],
                            &write_masks[memory_start + emitted + 2], state.timestamp + 2,
                            decoded.address_space, decoded.aligned_address + 8, 0, 0
                        );
                    }
                    if (decoded.needs_write) {
                        uint32_t write_index = emitted + 2 + uint32_t(decoded.crosses);
                        uint32_t write_timestamp = state.timestamp + (byte_load ? 2 : 3);
                        write_event(memory[memory_start + write_index],
                                    &write_masks[memory_start + write_index], write_timestamp,
                                    register_as, instruction->words[1] / 2, true, value);
                    }
                }
                emitted += event_count;
                if (decoded.needs_write) state.regs[decoded.rd_or_rs2] = value;
                state.timestamp += byte_load ? 3 : 4;
            } else {
                uint32_t event_count = 3 + uint32_t(decoded.crosses);
                if (memory != nullptr) {
                    if (uint64_t(memory_start) + emitted + event_count > memory_capacity) {
                        preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
                        return false;
                    }
                    write_event(memory[memory_start + emitted], &write_masks[memory_start + emitted],
                                state.timestamp, register_as, instruction->words[2] / 2, false,
                                state.regs[decoded.rs1]);
                    uint64_t source = state.regs[decoded.rd_or_rs2];
                    write_event(memory[memory_start + emitted + 1],
                                &write_masks[memory_start + emitted + 1], state.timestamp + 1,
                                register_as, instruction->words[1] / 2, false, source);
                    uint64_t block_values[2] = {0, 0};
                    uint8_t block_masks[2] = {0, 0};
                    for (uint32_t byte = 0; byte < decoded.width; byte++) {
                        uint32_t position = decoded.shift + byte;
                        uint32_t block = position / 8;
                        uint32_t within = position % 8;
                        block_values[block] |= ((source >> (8 * byte)) & 0xff) << (8 * within);
                        block_masks[block] |= uint8_t(1u << within);
                    }
                    write_memory_intent(
                        memory[memory_start + emitted + 2],
                        &write_masks[memory_start + emitted + 2], state.timestamp + 2,
                        decoded.address_space, decoded.aligned_address, block_values[0],
                        block_masks[0]
                    );
                    if (decoded.crosses) {
                        write_memory_intent(
                            memory[memory_start + emitted + 3],
                            &write_masks[memory_start + emitted + 3], state.timestamp + 3,
                            decoded.address_space, decoded.aligned_address + 8, block_values[1],
                            block_masks[1]
                        );
                    }
                }
                emitted += event_count;
                state.timestamp += decoded.width == 1 ? 3 : 4;
            }
            state.pc += 4;
        } else {
            bool take;
            if (validate_branch_instruction(
                    *instruction, opcodes, register_as, rs1, rs2, take, state
                )) {
                if (memory != nullptr) {
                    if (uint64_t(memory_start) + emitted + 2 > memory_capacity) {
                        preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
                        return false;
                    }
                    write_event(memory[memory_start + emitted], &write_masks[memory_start + emitted],
                                state.timestamp, register_as,
                                instruction->words[1] / 2, false, state.regs[rs1]);
                    write_event(memory[memory_start + emitted + 1],
                                &write_masks[memory_start + emitted + 1], state.timestamp + 1,
                                register_as,
                                instruction->words[2] / 2, false, state.regs[rs2]);
                }
                emitted += 2;
                state.pc = take ? branch_target(state.pc, instruction->words[3]) : state.pc + 4;
                state.timestamp += 2;
            } else if (opcode == opcodes.jal_lui) {
                uint32_t needs_write = instruction->words[6];
                if (!decode_register(instruction->words[1], rd) || instruction->words[2] != 0 ||
                    instruction->words[4] != register_as || instruction->words[5] != 0 ||
                    needs_write != uint32_t(rd != 0) || instruction->words[7] != 0 ||
                    instruction->words[3] >= BABY_BEAR_ORDER) {
                    preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                    return false;
                }
                result = state.pc + 4;
                if (needs_write) {
                    if (memory != nullptr) {
                        if (uint64_t(memory_start) + emitted + 1 > memory_capacity) {
                            preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
                            return false;
                        }
                        write_event(memory[memory_start + emitted],
                                    &write_masks[memory_start + emitted], state.timestamp,
                                    register_as,
                                    instruction->words[1] / 2, true, result);
                    }
                    emitted++;
                    state.regs[rd] = result;
                }
                state.pc = branch_target(state.pc, instruction->words[3]);
                state.timestamp++;
            } else if (opcode == opcodes.jal_lui + 1) {
                if (!decode_register(instruction->words[1], rd) || rd == 0 ||
                    instruction->words[2] != 0 || instruction->words[3] >= (1u << 20) ||
                    instruction->words[4] != register_as || instruction->words[5] != 0 ||
                    instruction->words[6] != 1 || instruction->words[7] != 0) {
                    preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                    return false;
                }
                result = sign_extend_word(instruction->words[3] << 12);
                if (memory != nullptr) {
                    if (uint64_t(memory_start) + emitted + 1 > memory_capacity) {
                        preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
                        return false;
                    }
                    write_event(memory[memory_start + emitted], &write_masks[memory_start + emitted],
                                state.timestamp, register_as,
                                instruction->words[1] / 2, true, result);
                }
                emitted++;
                state.regs[rd] = result;
                state.pc += 4;
                state.timestamp++;
            } else if (opcode == opcodes.jalr) {
                uint32_t needs_write = instruction->words[6];
                uint32_t imm_sign = instruction->words[7];
                if (!decode_register(instruction->words[1], rd) ||
                    !decode_register(instruction->words[2], rs1) ||
                    instruction->words[3] > UINT16_MAX ||
                    instruction->words[4] != register_as || instruction->words[5] != 0 ||
                    needs_write != uint32_t(rd != 0) || imm_sign > 1 ||
                    (state.regs[rs1] >> 32) != 0) {
                    preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                    return false;
                }
                int64_t signed_offset =
                    imm_sign ? int64_t(instruction->words[3]) - (int64_t(1) << 16)
                             : int64_t(instruction->words[3]);
                uint64_t target =
                    (state.regs[rs1] + uint64_t(signed_offset)) & ~uint64_t(1);
                if (target > UINT32_MAX) {
                    preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                    return false;
                }
                result = state.pc + 4;
                if (memory != nullptr) {
                    uint32_t event_count = 1 + needs_write;
                    if (uint64_t(memory_start) + emitted + event_count > memory_capacity) {
                        preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
                        return false;
                    }
                    write_event(memory[memory_start + emitted], &write_masks[memory_start + emitted],
                                state.timestamp, register_as,
                                instruction->words[2] / 2, false, state.regs[rs1]);
                    if (needs_write) {
                        write_event(memory[memory_start + emitted + 1],
                                    &write_masks[memory_start + emitted + 1], state.timestamp + 1,
                                    register_as, instruction->words[1] / 2, true, result);
                    }
                }
                emitted += 1 + needs_write;
                state.pc = uint32_t(target);
                if (needs_write) state.regs[rd] = result;
                state.timestamp += 2;
            } else if (opcode == opcodes.auipc) {
                if (!decode_register(instruction->words[1], rd) || rd == 0 ||
                    instruction->words[2] != 0 || instruction->words[3] >= (1u << 24) ||
                    !validate_tail(*instruction, register_as, 0)) {
                    preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                    return false;
                }
                int64_t offset = int64_t(int32_t(instruction->words[3] << 8));
                result = uint64_t(state.pc) + uint64_t(offset);
                if (memory != nullptr) {
                    if (uint64_t(memory_start) + emitted + 1 > memory_capacity) {
                        preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
                        return false;
                    }
                    write_event(memory[memory_start + emitted], &write_masks[memory_start + emitted],
                                state.timestamp, register_as,
                                instruction->words[1] / 2, true, result);
                }
                emitted++;
                state.regs[rd] = result;
                state.pc += 4;
                state.timestamp++;
            } else if (opcode == opcodes.phantom) {
                if (instruction->words[4] != 0 || instruction->words[5] != 0 ||
                    instruction->words[6] != 0 || instruction->words[7] != 0) {
                    preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                    return false;
                }
                state.pc += 4;
                state.timestamp++;
            } else if (opcode == opcodes.terminate) {
                if (endpoint_kind != 0 || local_step + 1 != expected_steps ||
                    chunk + 1 != anchors.len()) {
                    preflight_set_error(error, ERROR_BAD_TERMINATION);
                    return false;
                }
                terminated = true;
            } else if (opcode < schedule_dispatch.len() &&
                       schedule_dispatch[opcode] != NO_SCHEDULE) {
                uint32_t schedule_index = schedule_dispatch[opcode];
                if (schedule_index >= schedules.len() ||
                    !replay_access_schedule(*instruction, schedules[schedule_index], spans,
                                            residuals, initial_memory, register_as,
                                            pointer_max_bits, state, memory, write_masks,
                                            memory_capacity, memory_start, emitted, error)) {
                    if (schedule_index >= schedules.len()) {
                        preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                    }
                    return false;
                }
            } else {
                preflight_set_error(error, ERROR_UNSUPPORTED_OPCODE);
                return false;
            }
        }
        state.retired++;
    }

    bool is_final_chunk = chunk + 1 == anchors.len();
    if (is_final_chunk ? (terminated != (endpoint_kind == 0)) : terminated) {
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
    DeviceBufferConstView<uint32_t> schedule_dispatch,
    DeviceBufferConstView<RvrCheckpointAccessSchedule> schedules,
    DeviceBufferConstView<RvrCheckpointAccessSpan> spans,
    RvrCheckpointOpcodeBases opcodes,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t immediate_as,
    uint32_t pointer_max_bits,
    uint32_t initial_pc,
    uint32_t initial_timestamp,
    uint32_t endpoint_kind,
    uint32_t *memory_counts,
    uint32_t *error
) {
    size_t chunk = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (chunk >= anchors.len()) return;
    uint32_t count = 0;
    if (replay_chunk(instructions, pc_base, initial_registers, initial_memory, anchors, residuals,
                     schedule_dispatch, schedules, spans, chunk, opcodes, register_as, memory_as, immediate_as,
                     pointer_max_bits, initial_pc, initial_timestamp, endpoint_kind, nullptr,
                     nullptr, nullptr, 0, 0, count, error)) {
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
    DeviceBufferConstView<uint32_t> schedule_dispatch,
    DeviceBufferConstView<RvrCheckpointAccessSchedule> schedules,
    DeviceBufferConstView<RvrCheckpointAccessSpan> spans,
    RvrCheckpointOpcodeBases opcodes,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t immediate_as,
    uint32_t pointer_max_bits,
    uint32_t initial_pc,
    uint32_t initial_timestamp,
    uint32_t endpoint_kind,
    DeviceBufferView<PreflightProgramEvent> program,
    DeviceBufferView<PreflightMemoryEvent> memory,
    DeviceBufferView<uint8_t> write_masks,
    uint32_t *error
) {
    size_t chunk = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (chunk >= anchors.len()) return;
    uint32_t count = 0;
    replay_chunk(instructions, pc_base, initial_registers, initial_memory, anchors, residuals,
                 schedule_dispatch, schedules, spans, chunk, opcodes, register_as, memory_as, immediate_as,
                 pointer_max_bits, initial_pc, initial_timestamp, endpoint_kind, program.data(),
                 memory.data(), write_masks.data(), memory.len(), memory_offsets[chunk], count,
                 error);
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

} // namespace

extern "C" int _rvr_checkpoint_count(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<uint8_t> initial_registers,
    DeviceBufferConstView<uint8_t> initial_memory,
    DeviceBufferConstView<RvrCheckpoint> anchors,
    DeviceBufferConstView<uint64_t> residuals,
    DeviceBufferConstView<uint32_t> schedule_dispatch,
    DeviceBufferConstView<RvrCheckpointAccessSchedule> schedules,
    DeviceBufferConstView<RvrCheckpointAccessSpan> spans,
    RvrCheckpointOpcodeBases opcodes,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t immediate_as,
    uint32_t pointer_max_bits,
    uint32_t initial_pc,
    uint32_t initial_timestamp,
    uint32_t endpoint_kind,
    uint32_t *memory_counts,
    uint32_t *error,
    cudaStream_t stream
) {
    if (anchors.len() == 0) return int(cudaErrorInvalidValue);
    auto [grid, block] = kernel_launch_params(anchors.len());
    checkpoint_count<<<grid, block, 0, stream>>>(
        instructions, pc_base, initial_registers, initial_memory, anchors, residuals,
        schedule_dispatch, schedules, spans, opcodes, register_as,
        memory_as, immediate_as, pointer_max_bits, initial_pc, initial_timestamp,
        endpoint_kind, memory_counts, error
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
    DeviceBufferConstView<uint32_t> schedule_dispatch,
    DeviceBufferConstView<RvrCheckpointAccessSchedule> schedules,
    DeviceBufferConstView<RvrCheckpointAccessSpan> spans,
    RvrCheckpointOpcodeBases opcodes,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t immediate_as,
    uint32_t pointer_max_bits,
    uint32_t initial_pc,
    uint32_t initial_timestamp,
    uint32_t endpoint_kind,
    DeviceBufferView<PreflightProgramEvent> program,
    DeviceBufferView<PreflightMemoryEvent> memory,
    DeviceBufferView<uint8_t> write_masks,
    uint32_t *error,
    cudaStream_t stream
) {
    if (memory_offsets.len() != anchors.len() || write_masks.len() != memory.len()) {
        return int(cudaErrorInvalidValue);
    }
    auto [grid, block] = kernel_launch_params(anchors.len());
    checkpoint_emit<<<grid, block, 0, stream>>>(
        instructions, pc_base, initial_registers, initial_memory, anchors, residuals,
        memory_offsets, schedule_dispatch, schedules, spans, opcodes,
        register_as, memory_as, immediate_as, pointer_max_bits, initial_pc, initial_timestamp,
        endpoint_kind, program, memory, write_masks, error
    );
    return CHECK_KERNEL();
}
