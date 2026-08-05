#include "arch/rvr/replay.cuh"
#include "fp.h"
#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "checkpoint_replay_opcodes.cuh"

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
static constexpr uint32_t ERROR_BAD_REPLAY_VALUE = 306;
static constexpr uint32_t ERROR_BAD_ANCHOR = 307;
static constexpr uint32_t ERROR_BAD_TERMINATION = 308;
static constexpr uint32_t ERROR_OUTPUT_BOUNDS = 309;
static constexpr uint32_t ERROR_BAD_ENDPOINT = 310;
static constexpr uint8_t FULL_WRITE_MASK = 0xff;

struct RvrCheckpoint {
    uint32_t pc;
    uint32_t timestamp;
    uint32_t retired;
    uint32_t replay_value_cursor;
    uint64_t regs[31];
};

static_assert(sizeof(RvrCheckpoint) == 264);

struct ReplayState {
    uint32_t pc;
    uint32_t timestamp;
    uint32_t retired;
    uint32_t replay_value_cursor;
    uint64_t regs[NUM_REGISTERS];
};

static constexpr uint32_t RVR_REPLAY_REGISTER_OPERANDS = 3;

struct RvrReplayAccessSchedule {
    uint32_t first_span;
    uint32_t num_spans;
    uint8_t register_operands[RVR_REPLAY_REGISTER_OPERANDS];
    uint8_t num_register_reads;
    uint8_t effect;
    uint8_t effect_operand;
    uint8_t register_write_source;
    uint8_t register_write_operand;
};

static_assert(sizeof(RvrReplayAccessSchedule) == 16);

struct RvrCheckpointAccessSpan {
    uint32_t address_space;
    uint32_t count;
    uint8_t base_index;
    uint8_t base_source;
    uint8_t count_register;
    uint8_t count_shift;
    uint8_t count_source;
    uint8_t value_source;
    uint16_t value_index;
};

static_assert(sizeof(RvrCheckpointAccessSpan) == 16);
static_assert(offsetof(RvrCheckpointAccessSpan, address_space) == 0);
static_assert(offsetof(RvrCheckpointAccessSpan, count) == 4);
static_assert(offsetof(RvrCheckpointAccessSpan, base_index) == 8);
static_assert(offsetof(RvrCheckpointAccessSpan, value_source) == 13);
static_assert(offsetof(RvrCheckpointAccessSpan, value_index) == 14);

struct RvrCheckpointEventCount {
    uint32_t memory;
    uint32_t field;
};

static_assert(sizeof(RvrCheckpointEventCount) == 2 * sizeof(uint32_t));
static_assert(offsetof(RvrCheckpointEventCount, memory) == 0);
static_assert(offsetof(RvrCheckpointEventCount, field) == 4);

static constexpr uint32_t NO_SCHEDULE = UINT32_MAX;
static constexpr uint8_t SPAN_BASE_REGISTER = 0;
static constexpr uint8_t SPAN_BASE_DEFERRAL_INPUT = 1;
static constexpr uint8_t SPAN_BASE_DEFERRAL_OUTPUT = 2;
static constexpr uint8_t SPAN_COUNT_FIXED = 0;
static constexpr uint8_t SPAN_COUNT_REGISTER = 1;
static constexpr uint8_t SPAN_COUNT_REPLAY_VALUE = 2;
static constexpr uint8_t SPAN_READ_U16 = 0;
static constexpr uint8_t SPAN_WRITE_U16_REPLAY_VALUE = 1;
static constexpr uint8_t SPAN_WRITE_U16_ZERO = 2;
static constexpr uint8_t SPAN_READ_FIELD32 = 3;
static constexpr uint8_t SPAN_WRITE_FIELD32_CANONICAL_PAIRS = 4;
static constexpr uint8_t SPAN_WRITE_U16_STATIC = 5;
static constexpr uint8_t EFFECT_NEXT = 0;
static constexpr uint8_t EFFECT_BRANCH_REPLAY_VALUE = 1;
static constexpr uint8_t REGISTER_WRITE_NONE = 0;
static constexpr uint8_t REGISTER_WRITE_ZERO = 1;
static constexpr uint8_t REGISTER_WRITE_REPLAY_VALUE = 2;

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

__device__ __forceinline__ void load_initial_state(
    DeviceBufferConstView<uint8_t> initial_registers,
    uint32_t pc,
    uint32_t timestamp,
    ReplayState &state
) {
    state.pc = pc;
    state.timestamp = timestamp;
    state.retired = 0;
    state.replay_value_cursor = 0;
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
    state.replay_value_cursor = checkpoint.replay_value_cursor;
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
        state.replay_value_cursor != checkpoint.replay_value_cursor) return false;
#pragma unroll
    for (uint32_t reg = 1; reg < NUM_REGISTERS; reg++) {
        if (state.regs[reg] != checkpoint.regs[reg - 1]) return false;
    }
    return true;
}

__device__ __forceinline__ bool opcode_in_family(
    uint32_t opcode, uint32_t base, uint32_t count, uint32_t &local
) {
    local = opcode - base;
    return opcode >= base && local < count;
}

__device__ __forceinline__ bool decode_register(uint32_t pointer, uint32_t &reg) {
    if (!replay_canonical_register_pointer(pointer)) return false;
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
    uint64_t lhs,
    uint64_t rhs,
    bool &matched
) {
    uint32_t local;
    matched = true;
    if (opcode_in_family(opcode, BASE_ALU_OPCODE_BASE, BASE_ALU_OPCODE_COUNT, local)) {
        switch (local) {
        case 0: return lhs + rhs;
        case 1: return lhs - rhs;
        case 2: return lhs ^ rhs;
        case 3: return lhs | rhs;
        default: return lhs & rhs;
        }
    }
    if (opcode_in_family(opcode, SHIFT_OPCODE_BASE, SHIFT_OPCODE_COUNT, local)) {
        uint32_t shift = uint32_t(rhs) & 63;
        if (local == 0) return lhs << shift;
        if (local == 1) return lhs >> shift;
        return arithmetic_shift_right(lhs, shift);
    }
    if (opcode_in_family(opcode, LESS_THAN_OPCODE_BASE, LESS_THAN_OPCODE_COUNT, local)) {
        return local == 0 ? uint64_t(int64_t(lhs) < int64_t(rhs)) : uint64_t(lhs < rhs);
    }
    if (opcode_in_family(opcode, BASE_ALU_W_OPCODE_BASE, BASE_ALU_W_OPCODE_COUNT, local)) {
        uint32_t result = local == 0 ? uint32_t(lhs) + uint32_t(rhs)
                                     : uint32_t(lhs) - uint32_t(rhs);
        return sign_extend_word(result);
    }
    if (opcode_in_family(opcode, SHIFT_W_OPCODE_BASE, SHIFT_W_OPCODE_COUNT, local)) {
        uint32_t shift = uint32_t(rhs) & 31;
        uint32_t value = uint32_t(lhs);
        if (local == 0) return sign_extend_word(value << shift);
        if (local == 1) return sign_extend_word(value >> shift);
        return sign_extend_word(uint32_t(int32_t(value) >> shift));
    }
    if (opcode == MUL_OPCODE_BASE) return lhs * rhs;
    if (opcode_in_family(opcode, MULH_OPCODE_BASE, MULH_OPCODE_COUNT, local)) {
        if (local == 0) return uint64_t(__mul64hi(int64_t(lhs), int64_t(rhs)));
        if (local == 1) return __umul64hi(lhs, rhs) - ((lhs >> 63) ? rhs : 0);
        return __umul64hi(lhs, rhs);
    }
    if (opcode_in_family(opcode, DIVREM_OPCODE_BASE, DIVREM_OPCODE_COUNT, local)) {
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
    if (opcode == MUL_W_OPCODE_BASE) return sign_extend_word(uint32_t(lhs) * uint32_t(rhs));
    if (opcode_in_family(opcode, DIVREM_W_OPCODE_BASE, DIVREM_W_OPCODE_COUNT, local)) {
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

__device__ __forceinline__ bool rr_uses_immediate_as(uint32_t opcode) {
    uint32_t local;
    return opcode == MUL_OPCODE_BASE || opcode_in_family(opcode, MULH_OPCODE_BASE, MULH_OPCODE_COUNT, local) ||
           opcode_in_family(opcode, DIVREM_OPCODE_BASE, DIVREM_OPCODE_COUNT, local) || opcode == MUL_W_OPCODE_BASE ||
           opcode_in_family(opcode, DIVREM_W_OPCODE_BASE, DIVREM_W_OPCODE_COUNT, local);
}

__device__ __forceinline__ bool is_rr_opcode(uint32_t opcode) {
    bool matched;
    execute_rr_result(opcode, 0, 0, matched);
    return matched;
}

__device__ __forceinline__ uint64_t execute_ri_result(
    uint32_t opcode,
    uint64_t lhs,
    uint32_t encoded,
    bool &matched,
    bool &valid
) {
    uint32_t local;
    int64_t immediate;
    matched = true;
    valid = true;
    if (opcode_in_family(opcode, BASE_ALU_IMM_OPCODE_BASE, BASE_ALU_IMM_OPCODE_COUNT, local)) {
        valid = decode_signed_12(encoded, immediate);
        if (!valid) return 0;
        uint64_t imm = uint64_t(immediate);
        if (local == 0) return lhs + imm;
        if (local == 1) return lhs ^ imm;
        if (local == 2) return lhs | imm;
        return lhs & imm;
    }
    if (opcode_in_family(opcode, SHIFT_IMM_OPCODE_BASE, SHIFT_IMM_OPCODE_COUNT, local)) {
        valid = encoded < 64;
        if (!valid) return 0;
        if (local == 0) return lhs << encoded;
        if (local == 1) return lhs >> encoded;
        return arithmetic_shift_right(lhs, encoded);
    }
    if (opcode_in_family(opcode, LESS_THAN_IMM_OPCODE_BASE, LESS_THAN_IMM_OPCODE_COUNT, local)) {
        valid = decode_signed_12(encoded, immediate);
        if (!valid) return 0;
        return local == 0 ? uint64_t(int64_t(lhs) < immediate)
                          : uint64_t(lhs < uint64_t(immediate));
    }
    if (opcode == BASE_ALU_W_IMM_OPCODE_BASE) {
        valid = decode_signed_12(encoded, immediate);
        if (!valid) return 0;
        return sign_extend_word(uint32_t(lhs) + uint32_t(immediate));
    }
    if (opcode_in_family(opcode, SHIFT_W_IMM_OPCODE_BASE, SHIFT_W_IMM_OPCODE_COUNT, local)) {
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

__device__ __forceinline__ bool is_ri_opcode(uint32_t opcode) {
    bool matched;
    bool valid;
    execute_ri_result(opcode, 0, 0, matched, valid);
    return matched;
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
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    ReplayState const &state,
    size_t initial_memory_bytes,
    HintStoreInstruction &decoded
) {
    uint32_t local;
    if (!opcode_in_family(instruction.words[0], HINT_STORE_OPCODE_BASE, HINT_STORE_OPCODE_COUNT, local) ||
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
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t pointer_max_bits,
    ReplayState const &state,
    size_t initial_memory_bytes,
    LoadStoreInstruction &decoded
) {
    uint32_t local;
    if (!opcode_in_family(instruction.words[0], LOAD_STORE_OPCODE_BASE, LOAD_STORE_OPCODE_COUNT, local)) return false;
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
    if (instruction.words[4] != register_as ||
        !replay_canonical_register_pointer(rd_or_rs2_ptr) ||
        !replay_canonical_register_pointer(rs1_ptr) || imm > UINT16_MAX || imm_sign > 1)
        return false;
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
    uint64_t lhs,
    uint64_t rhs,
    bool &matched
) {
    uint32_t local;
    matched = true;
    if (opcode_in_family(opcode, BRANCH_EQUAL_OPCODE_BASE, BRANCH_EQUAL_OPCODE_COUNT, local)) {
        return local == 0 ? lhs == rhs : lhs != rhs;
    }
    if (opcode_in_family(opcode, BRANCH_LESS_THAN_OPCODE_BASE, BRANCH_LESS_THAN_OPCODE_COUNT, local)) {
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
    result = execute_rr_result(instruction.words[0], state.regs[rs1], state.regs[rs2], matched);
    uint32_t expected_e = rr_uses_immediate_as(instruction.words[0])
                              ? immediate_as
                              : register_as;
    return matched && validate_tail(instruction, register_as, expected_e);
}

__device__ __forceinline__ bool validate_ri_instruction(
    RvrReplayInstruction const &instruction,
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
        instruction.words[0], state.regs[rs1], instruction.words[3], matched, valid
    );
    return matched && valid;
}

__device__ __forceinline__ bool validate_branch_instruction(
    RvrReplayInstruction const &instruction,
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
        instruction.words[0], state.regs[rs1], state.regs[rs2], matched
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
    // Register values are already known while replaying a checkpoint interval.
    // Chronology still links accesses, but it need not scatter these values back
    // into the log after sorting by address.
    u64_to_limbs(value, event.value);
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

__device__ __forceinline__ bool resolve_access_span_count(
    RvrCheckpointAccessSpan const &span,
    RvrReplayAccessSchedule const &schedule,
    uint64_t const (&register_values)[3],
    DeviceBufferConstView<uint64_t> replay_values,
    uint64_t replay_value_index,
    uint32_t &count,
    uint32_t *error
) {
    if (span.count_source == SPAN_COUNT_FIXED) {
        if (span.count == 0 || span.count_register != 0 || span.count_shift != 0) {
            preflight_set_error(error, ERROR_BAD_INSTRUCTION);
            return false;
        }
        count = span.count;
        return true;
    }
    if (span.count_source == SPAN_COUNT_REGISTER) {
        if (span.count_register >= schedule.num_register_reads || span.count_shift >= 64) {
            preflight_set_error(error, ERROR_BAD_INSTRUCTION);
            return false;
        }
        uint64_t encoded_count = register_values[span.count_register];
        uint64_t low_mask =
            span.count_shift == 0 ? 0 : (uint64_t(1) << span.count_shift) - 1;
        uint64_t shifted = encoded_count >> span.count_shift;
        if ((encoded_count & low_mask) != 0 || shifted > span.count) {
            preflight_set_error(error, ERROR_BAD_INSTRUCTION);
            return false;
        }
        count = uint32_t(shifted);
        return true;
    }
    if (span.count_source == SPAN_COUNT_REPLAY_VALUE) {
        if (span.count == 0 || span.count_register != 0 || span.count_shift != 0 ||
            replay_value_index >= replay_values.len() || replay_values[replay_value_index] > span.count) {
            preflight_set_error(error, ERROR_BAD_REPLAY_VALUE);
            return false;
        }
        count = uint32_t(replay_values[replay_value_index]);
        return true;
    }
    preflight_set_error(error, ERROR_BAD_INSTRUCTION);
    return false;
}

__device__ __forceinline__ bool replay_access_schedule(
    RvrReplayInstruction const &instruction,
    RvrReplayAccessSchedule const &schedule,
    DeviceBufferConstView<RvrCheckpointAccessSpan> spans,
    DeviceBufferConstView<uint64_t> static_values,
    DeviceBufferConstView<uint64_t> replay_values,
    DeviceBufferConstView<uint8_t> initial_memory,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t deferral_as,
    uint32_t byte_pointer_max_bits,
    uint32_t cell_pointer_max_bits,
    ReplayState &state,
    PreflightMemoryEvent *memory,
    uint8_t *write_masks,
    size_t memory_capacity,
    uint32_t memory_start,
    uint32_t &emitted,
    RvrFieldBlock *field_values,
    size_t field_capacity,
    uint32_t field_start,
    uint32_t &field_emitted,
    uint32_t *error
) {
    if (uint64_t(schedule.first_span) + schedule.num_spans > spans.len()) {
        preflight_set_error(error, ERROR_BAD_INSTRUCTION);
        return false;
    }
    if (schedule.effect != EFFECT_NEXT && schedule.effect != EFFECT_BRANCH_REPLAY_VALUE) {
        preflight_set_error(error, ERROR_BAD_INSTRUCTION);
        return false;
    }
    bool has_register_write = schedule.register_write_source != REGISTER_WRITE_NONE;
    if ((schedule.register_write_source != REGISTER_WRITE_NONE &&
         schedule.register_write_source != REGISTER_WRITE_ZERO &&
         schedule.register_write_source != REGISTER_WRITE_REPLAY_VALUE) ||
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
    uint64_t span_replay_values = 0;
    uint64_t field_events = 0;
    for (uint32_t span_index = 0; span_index < schedule.num_spans; span_index++) {
        auto const &span = spans[schedule.first_span + span_index];
        bool field = span.value_source == SPAN_READ_FIELD32 ||
                     span.value_source == SPAN_WRITE_FIELD32_CANONICAL_PAIRS;
        bool known_value = span.value_source == SPAN_READ_U16 ||
                           span.value_source == SPAN_WRITE_U16_REPLAY_VALUE ||
                           span.value_source == SPAN_WRITE_U16_ZERO ||
                           span.value_source == SPAN_WRITE_U16_STATIC || field;
        bool known_base = span.base_source == SPAN_BASE_REGISTER ||
                          span.base_source == SPAN_BASE_DEFERRAL_INPUT ||
                          span.base_source == SPAN_BASE_DEFERRAL_OUTPUT;
        bool known_count = span.count_source == SPAN_COUNT_FIXED ||
                           span.count_source == SPAN_COUNT_REGISTER ||
                           span.count_source == SPAN_COUNT_REPLAY_VALUE;
        bool static_write = span.value_source == SPAN_WRITE_U16_STATIC;
        if (!known_value || !known_base || !known_count ||
            (field ? span.address_space != deferral_as : span.address_space != memory_as) ||
            (field ? span.base_source == SPAN_BASE_REGISTER
                   : span.base_source != SPAN_BASE_REGISTER) ||
            (field && span.count_source != SPAN_COUNT_FIXED) ||
            (static_write &&
             (span.count_source != SPAN_COUNT_FIXED ||
              uint64_t(span.value_index) + span.count > static_values.len())) ||
            (!static_write && span.value_index != 0)) {
            preflight_set_error(error, ERROR_BAD_INSTRUCTION);
            return false;
        }
        if (span_replay_values > UINT32_MAX - state.replay_value_cursor) {
            preflight_set_error(error, ERROR_BAD_REPLAY_VALUE);
            return false;
        }
        uint32_t count;
        if (!resolve_access_span_count(
                span,
                schedule,
                register_values,
                replay_values,
                uint64_t(state.replay_value_cursor) + span_replay_values,
                count,
                error
            )) return false;
        if (span.count_source == SPAN_COUNT_REPLAY_VALUE) span_replay_values++;
        if (uint64_t(count) > UINT32_MAX - span_events) {
            preflight_set_error(error, ERROR_BAD_INSTRUCTION);
            return false;
        }
        span_events += count;
        if (field) field_events += count;
        if (count == 0) continue;

        uint64_t base;
        uint64_t stride;
        if (span.base_source == SPAN_BASE_REGISTER) {
            if (span.base_index >= schedule.num_register_reads) {
                preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                return false;
            }
            base = register_values[span.base_index];
            stride = REGISTER_BYTES;
        } else {
            if (span.base_index == 0 || span.base_index >= 8 || count != 2) {
                preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                return false;
            }
            base = uint64_t(instruction.words[span.base_index]) * 16u +
                   (span.base_source == SPAN_BASE_DEFERRAL_OUTPUT ? 8u : 0u);
            stride = 4;
        }
        uint64_t end = base + uint64_t(count) * stride;
        bool bad_u16_bounds =
            !field && (end > initial_memory.len() || (base & (REGISTER_BYTES - 1)) != 0);
        uint32_t pointer_bits = field ? cell_pointer_max_bits : byte_pointer_max_bits;
        if ((base >> 32) != 0 || end < base || end > uint64_t(UINT32_MAX) + 1 ||
            bad_u16_bounds || (pointer_bits < 32 && end > (uint64_t(1) << pointer_bits))) {
            preflight_set_error(error, ERROR_BAD_INSTRUCTION);
            return false;
        }
        if (span.value_source == SPAN_WRITE_U16_REPLAY_VALUE) span_replay_values += count;
        if (span.value_source == SPAN_WRITE_FIELD32_CANONICAL_PAIRS) {
            span_replay_values += 2u * count;
        }
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
    if (field_values != nullptr &&
        uint64_t(field_start) + field_emitted + field_events > field_capacity) {
        preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
        return false;
    }
    uint64_t register_write_replay_values =
        schedule.register_write_source == REGISTER_WRITE_REPLAY_VALUE ? 1u : 0u;
    uint64_t effect_replay_value_index = uint64_t(state.replay_value_cursor) + span_replay_values +
                                     register_write_replay_values;
    uint64_t required_replay_values = span_replay_values + register_write_replay_values +
                                  (schedule.effect == EFFECT_BRANCH_REPLAY_VALUE ? 1u : 0u);
    if (required_replay_values > UINT32_MAX - state.replay_value_cursor ||
        uint64_t(state.replay_value_cursor) + required_replay_values > replay_values.len()) {
        preflight_set_error(error, ERROR_BAD_REPLAY_VALUE);
        return false;
    }
    uint64_t checked_replay_value = state.replay_value_cursor;
    for (uint32_t span_index = 0; span_index < schedule.num_spans; span_index++) {
        auto const &span = spans[schedule.first_span + span_index];
        uint32_t count;
        if (!resolve_access_span_count(
                span, schedule, register_values, replay_values, checked_replay_value, count, error
            )) return false;
        if (span.count_source == SPAN_COUNT_REPLAY_VALUE) checked_replay_value++;
        if (span.value_source == SPAN_WRITE_U16_REPLAY_VALUE) {
            checked_replay_value += count;
        } else if (span.value_source == SPAN_WRITE_FIELD32_CANONICAL_PAIRS) {
            for (uint32_t block = 0; block < count; block++) {
#pragma unroll
                for (uint32_t pair = 0; pair < 2; pair++) {
                    uint64_t packed = replay_values[checked_replay_value++];
                    if (uint32_t(packed) >= BABY_BEAR_ORDER ||
                        uint32_t(packed >> 32) >= BABY_BEAR_ORDER) {
                        preflight_set_error(error, ERROR_BAD_REPLAY_VALUE);
                        return false;
                    }
                }
            }
        }
    }
    uint64_t effect_replay_value = 0;
    if (schedule.effect == EFFECT_BRANCH_REPLAY_VALUE) {
        effect_replay_value = replay_values[effect_replay_value_index];
        if (effect_replay_value > 1) {
            preflight_set_error(error, ERROR_BAD_REPLAY_VALUE);
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
        uint32_t count;
        if (!resolve_access_span_count(
                span,
                schedule,
                register_values,
                replay_values,
                state.replay_value_cursor,
                count,
                error
            )) return false;
        if (span.count_source == SPAN_COUNT_REPLAY_VALUE) state.replay_value_cursor++;
        if (count == 0) continue;

        bool field = span.value_source == SPAN_READ_FIELD32 ||
                     span.value_source == SPAN_WRITE_FIELD32_CANONICAL_PAIRS;
        uint64_t base = span.base_source == SPAN_BASE_REGISTER
                            ? register_values[span.base_index]
                            : uint64_t(instruction.words[span.base_index]) * 16u +
                                  (span.base_source == SPAN_BASE_DEFERRAL_OUTPUT ? 8u : 0u);
        bool is_u16_replay_value = span.value_source == SPAN_WRITE_U16_REPLAY_VALUE;
        bool is_field_replay_value =
            span.value_source == SPAN_WRITE_FIELD32_CANONICAL_PAIRS;
        bool is_static_write = span.value_source == SPAN_WRITE_U16_STATIC;
        bool is_write = is_u16_replay_value || is_field_replay_value ||
                        span.value_source == SPAN_WRITE_U16_ZERO || is_static_write;
        for (uint32_t word = 0; word < count; word++) {
            if (memory != nullptr) {
                uint32_t cursor = memory_start + emitted;
                if (field) {
                    uint32_t reference = field_start + field_emitted;
                    auto &event = memory[cursor];
                    event.timestamp = state.timestamp;
                    event.address_space_and_kind =
                        span.address_space | (is_write ? PREFLIGHT_WRITE_BIT : 0);
                    event.pointer = uint32_t(base) + 4u * word;
                    event.value[0] = uint16_t(reference);
                    event.value[1] = uint16_t(reference >> 16);
                    event.value[2] = 0;
                    event.value[3] = 0;
                    write_masks[cursor] = is_write ? FULL_WRITE_MASK : 0;
                    auto &block = field_values[reference];
#pragma unroll
                    for (uint32_t lane = 0; lane < 4; lane++) block.values[lane] = 0;
                    if (is_field_replay_value) {
#pragma unroll
                        for (uint32_t pair = 0; pair < 2; pair++) {
                            uint64_t packed = replay_values[state.replay_value_cursor + pair];
                            block.values[2 * pair] = uint32_t(packed);
                            block.values[2 * pair + 1] = uint32_t(packed >> 32);
                        }
                    }
                } else {
                    uint64_t value = is_u16_replay_value
                                         ? replay_values[state.replay_value_cursor]
                                         : is_static_write
                                               ? static_values[span.value_index + word]
                                               : 0;
                    write_memory_intent(memory[cursor], &write_masks[cursor], state.timestamp,
                                        span.address_space,
                                        uint32_t(base) + REGISTER_BYTES * word, value,
                                        is_write ? FULL_WRITE_MASK : 0);
                }
            }
            if (field) field_emitted++;
            if (is_u16_replay_value) state.replay_value_cursor++;
            if (is_field_replay_value) state.replay_value_cursor += 2;
            emitted++;
            state.timestamp++;
        }
    }
    if (has_register_write) {
        uint64_t value = schedule.register_write_source == REGISTER_WRITE_REPLAY_VALUE
                             ? replay_values[state.replay_value_cursor]
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
        if (schedule.register_write_source == REGISTER_WRITE_REPLAY_VALUE) {
            state.replay_value_cursor++;
        }
        state.timestamp++;
    }
    if (schedule.effect == EFFECT_BRANCH_REPLAY_VALUE) {
        // All write replay values, if any, precede the terminal control replay value.
        state.replay_value_cursor++;
        state.pc = effect_replay_value != 0
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
    DeviceBufferConstView<uint64_t> replay_values,
    DeviceBufferConstView<uint32_t> schedule_dispatch,
    DeviceBufferConstView<RvrReplayAccessSchedule> schedules,
    DeviceBufferConstView<RvrCheckpointAccessSpan> spans,
    DeviceBufferConstView<uint64_t> static_values,
    size_t chunk,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t immediate_as,
    uint32_t deferral_as,
    uint32_t byte_pointer_max_bits,
    uint32_t cell_pointer_max_bits,
    uint32_t initial_pc,
    uint32_t initial_timestamp,
    uint32_t endpoint_kind,
    DeviceBufferView<PreflightProgramEvent> program,
    PreflightMemoryEvent *memory,
    uint8_t *write_masks,
    size_t memory_capacity,
    uint32_t memory_start,
    RvrFieldBlock *field_values,
    size_t field_capacity,
    uint32_t field_start,
    uint32_t &memory_count,
    uint32_t &field_count,
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
    if (end.retired < state.retired || end.replay_value_cursor < state.replay_value_cursor) {
        preflight_set_error(error, ERROR_BAD_CHUNK);
        return false;
    }
    if (program.data() != nullptr && end.retired > program.len()) {
        preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
        return false;
    }

    uint32_t expected_steps = end.retired - state.retired;
    uint32_t emitted = 0;
    uint32_t field_emitted = 0;
    bool terminated = false;
    // Fixed-arity opcode paths advance `state.timestamp` by at most 4 per step and
    // carry no overflow guard: a wrapped timestamp cannot pass the strictly
    // monotonic per-address chronology check, the host timestamp-domain checks, or
    // the end-anchor comparison. Variable-length paths (hint store, access
    // schedules) bound their own timestamp advance explicitly.
    for (uint32_t local_step = 0; local_step < expected_steps; local_step++) {
        auto instruction = resolve_replay_instruction(instructions, pc_base, state.pc);
        if (instruction == nullptr) {
            preflight_set_error(error, ERROR_BAD_PC);
            return false;
        }
        uint32_t opcode = instruction->words[0];
        if (program.data() != nullptr) {
            program[state.retired] = PreflightProgramEvent{state.pc, state.timestamp};
        }

        uint32_t rd, rs1, rs2;
        uint64_t result;
        if (validate_rr_instruction(
                *instruction, register_as, immediate_as, rd, rs1, rs2, result, state
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
                       *instruction, register_as, immediate_as, rd, rs1, result, state
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
        } else if (is_rr_opcode(opcode) || is_ri_opcode(opcode)) {
            preflight_set_error(error, ERROR_BAD_INSTRUCTION);
            return false;
        } else if (
            opcode >= HINT_STORE_OPCODE_BASE &&
            opcode < HINT_STORE_OPCODE_BASE + HINT_STORE_OPCODE_COUNT
        ) {
            HintStoreInstruction decoded{};
            if (!validate_hint_store(*instruction, register_as, memory_as,
                                     byte_pointer_max_bits, state, initial_memory.len(), decoded)) {
                preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                return false;
            }
            uint32_t event_count = decoded.num_words + (decoded.is_single ? 1 : 2);
            if (state.replay_value_cursor > replay_values.len() ||
                size_t(decoded.num_words) > replay_values.len() - state.replay_value_cursor) {
                preflight_set_error(error, ERROR_BAD_REPLAY_VALUE);
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
                    size_t replay_value_index = state.replay_value_cursor + word;
                    uint32_t event_index = memory_start + write_start + word;
                    write_memory_intent(
                        memory[event_index], &write_masks[event_index],
                        state.timestamp + 2 + 3 * word, memory_as,
                        decoded.mem_ptr + REGISTER_BYTES * word, replay_values[replay_value_index],
                        FULL_WRITE_MASK
                    );
                }
            }
            emitted += event_count;
            state.replay_value_cursor += decoded.num_words;
            state.pc += 4;
            state.timestamp += 3 * decoded.num_words;
        } else if (
            opcode >= LOAD_STORE_OPCODE_BASE &&
            opcode < LOAD_STORE_OPCODE_BASE + LOAD_STORE_OPCODE_COUNT
        ) {
            LoadStoreInstruction decoded{};
            if (!validate_load_store(*instruction, register_as, memory_as,
                                     byte_pointer_max_bits, state, initial_memory.len(), decoded)) {
                preflight_set_error(error, ERROR_BAD_LOAD);
                return false;
            }
            if (decoded.is_load) {
                uint64_t value = 0;
                if (decoded.needs_write) {
                    if (state.replay_value_cursor >= replay_values.len()) {
                        preflight_set_error(error, ERROR_BAD_REPLAY_VALUE);
                        return false;
                    }
                    value = replay_values[state.replay_value_cursor++];
                    if (normalize_load_result(value, decoded.width, decoded.sign_extend) != value) {
                        preflight_set_error(error, ERROR_BAD_REPLAY_VALUE);
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
                    *instruction, register_as, rs1, rs2, take, state
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
            } else if (opcode == JAL_LUI_OPCODE_BASE) {
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
            } else if (
                opcode == JAL_LUI_OPCODE_BASE + JAL_LUI_OPCODE_COUNT - 1
            ) {
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
            } else if (opcode == JALR_OPCODE_BASE) {
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
            } else if (opcode == AUIPC_OPCODE_BASE) {
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
            } else if (opcode == PHANTOM_OPCODE_BASE) {
                if (instruction->words[4] != 0 || instruction->words[5] != 0 ||
                    instruction->words[6] != 0 || instruction->words[7] != 0) {
                    preflight_set_error(error, ERROR_BAD_INSTRUCTION);
                    return false;
                }
                state.pc += 4;
                state.timestamp++;
            } else if (opcode == TERMINATE_OPCODE_BASE) {
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
                                            static_values, replay_values, initial_memory, register_as,
                                            memory_as, deferral_as,
                                            byte_pointer_max_bits, cell_pointer_max_bits, state,
                                            memory, write_masks,
                                            memory_capacity, memory_start, emitted, field_values,
                                            field_capacity, field_start, field_emitted, error)) {
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
    field_count = field_emitted;
    return true;
}

__global__ void checkpoint_count(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<uint8_t> initial_registers,
    DeviceBufferConstView<uint8_t> initial_memory,
    DeviceBufferConstView<RvrCheckpoint> anchors,
    DeviceBufferConstView<uint64_t> replay_values,
    DeviceBufferConstView<uint32_t> schedule_dispatch,
    DeviceBufferConstView<RvrReplayAccessSchedule> schedules,
    DeviceBufferConstView<RvrCheckpointAccessSpan> spans,
    DeviceBufferConstView<uint64_t> static_values,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t immediate_as,
    uint32_t deferral_as,
    uint32_t byte_pointer_max_bits,
    uint32_t cell_pointer_max_bits,
    uint32_t initial_pc,
    uint32_t initial_timestamp,
    uint32_t endpoint_kind,
    RvrCheckpointEventCount *event_counts,
    uint32_t *error
) {
    size_t chunk = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (chunk >= anchors.len()) return;
    uint32_t memory_count = 0;
    uint32_t field_count = 0;
    if (replay_chunk(instructions, pc_base, initial_registers, initial_memory, anchors, replay_values,
                     schedule_dispatch, schedules, spans, static_values, chunk,
                     register_as, memory_as, immediate_as, deferral_as,
                     byte_pointer_max_bits, cell_pointer_max_bits, initial_pc, initial_timestamp,
                     endpoint_kind, DeviceBufferView<PreflightProgramEvent>{nullptr, 0},
                     nullptr, nullptr, 0, 0, nullptr, 0, 0,
                     memory_count, field_count, error)) {
        event_counts[chunk] = RvrCheckpointEventCount{memory_count, field_count};
    }
}

__global__ void checkpoint_emit(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<uint8_t> initial_registers,
    DeviceBufferConstView<uint8_t> initial_memory,
    DeviceBufferConstView<RvrCheckpoint> anchors,
    DeviceBufferConstView<uint64_t> replay_values,
    DeviceBufferConstView<RvrCheckpointEventCount> event_offsets,
    DeviceBufferConstView<uint32_t> schedule_dispatch,
    DeviceBufferConstView<RvrReplayAccessSchedule> schedules,
    DeviceBufferConstView<RvrCheckpointAccessSpan> spans,
    DeviceBufferConstView<uint64_t> static_values,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t immediate_as,
    uint32_t deferral_as,
    uint32_t byte_pointer_max_bits,
    uint32_t cell_pointer_max_bits,
    uint32_t initial_pc,
    uint32_t initial_timestamp,
    uint32_t endpoint_kind,
    DeviceBufferView<PreflightProgramEvent> program,
    DeviceBufferView<PreflightMemoryEvent> memory,
    DeviceBufferView<uint8_t> write_masks,
    DeviceBufferView<RvrFieldBlock> field_values,
    uint32_t *error
) {
    size_t chunk = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (chunk >= anchors.len()) return;
    uint32_t memory_count = 0;
    uint32_t field_count = 0;
    auto const offsets = event_offsets[chunk];
    if (!replay_chunk(instructions, pc_base, initial_registers, initial_memory, anchors, replay_values,
                      schedule_dispatch, schedules, spans, static_values, chunk,
                      register_as, memory_as, immediate_as, deferral_as, byte_pointer_max_bits,
                      cell_pointer_max_bits, initial_pc, initial_timestamp, endpoint_kind,
                      program, memory.data(), write_masks.data(), memory.len(), offsets.memory,
                      field_values.data(), field_values.len(), offsets.field, memory_count,
                      field_count, error)) {
        return;
    }
    uint32_t expected_memory_end =
        chunk + 1 < event_offsets.len() ? event_offsets[chunk + 1].memory
                                        : static_cast<uint32_t>(memory.len());
    uint32_t expected_field_end =
        chunk + 1 < event_offsets.len() ? event_offsets[chunk + 1].field
                                        : static_cast<uint32_t>(field_values.len());
    if (uint64_t(offsets.memory) + memory_count != expected_memory_end ||
        uint64_t(offsets.field) + field_count != expected_field_end) {
        preflight_set_error(error, ERROR_OUTPUT_BOUNDS);
        return;
    }
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
    DeviceBufferConstView<uint64_t> replay_values,
    DeviceBufferConstView<uint32_t> schedule_dispatch,
    DeviceBufferConstView<RvrReplayAccessSchedule> schedules,
    DeviceBufferConstView<RvrCheckpointAccessSpan> spans,
    DeviceBufferConstView<uint64_t> static_values,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t immediate_as,
    uint32_t deferral_as,
    uint32_t byte_pointer_max_bits,
    uint32_t cell_pointer_max_bits,
    uint32_t initial_pc,
    uint32_t initial_timestamp,
    uint32_t endpoint_kind,
    RvrCheckpointEventCount *event_counts,
    uint32_t *error,
    cudaStream_t stream
) {
    if (anchors.len() == 0) return int(cudaErrorInvalidValue);
    auto [grid, block] = kernel_launch_params(anchors.len(), REPLAY_THREADS);
    checkpoint_count<<<grid, block, 0, stream>>>(
        instructions, pc_base, initial_registers, initial_memory, anchors, replay_values,
        schedule_dispatch, schedules, spans, static_values, register_as,
        memory_as, immediate_as, deferral_as, byte_pointer_max_bits, cell_pointer_max_bits,
        initial_pc, initial_timestamp,
        endpoint_kind, event_counts, error
    );
    return CHECK_KERNEL();
}

extern "C" int _rvr_checkpoint_emit(
    DeviceBufferConstView<RvrReplayInstruction> instructions,
    uint32_t pc_base,
    DeviceBufferConstView<uint8_t> initial_registers,
    DeviceBufferConstView<uint8_t> initial_memory,
    DeviceBufferConstView<RvrCheckpoint> anchors,
    DeviceBufferConstView<uint64_t> replay_values,
    DeviceBufferConstView<RvrCheckpointEventCount> event_offsets,
    DeviceBufferConstView<uint32_t> schedule_dispatch,
    DeviceBufferConstView<RvrReplayAccessSchedule> schedules,
    DeviceBufferConstView<RvrCheckpointAccessSpan> spans,
    DeviceBufferConstView<uint64_t> static_values,
    uint32_t register_as,
    uint32_t memory_as,
    uint32_t immediate_as,
    uint32_t deferral_as,
    uint32_t byte_pointer_max_bits,
    uint32_t cell_pointer_max_bits,
    uint32_t initial_pc,
    uint32_t initial_timestamp,
    uint32_t endpoint_kind,
    DeviceBufferView<PreflightProgramEvent> program,
    DeviceBufferView<PreflightMemoryEvent> memory,
    DeviceBufferView<uint8_t> write_masks,
    DeviceBufferView<RvrFieldBlock> field_values,
    uint32_t *error,
    cudaStream_t stream
) {
    if (event_offsets.len() != anchors.len() || write_masks.len() != memory.len()) {
        return int(cudaErrorInvalidValue);
    }
    auto [grid, block] = kernel_launch_params(anchors.len(), REPLAY_THREADS);
    checkpoint_emit<<<grid, block, 0, stream>>>(
        instructions, pc_base, initial_registers, initial_memory, anchors, replay_values,
        event_offsets, schedule_dispatch, schedules, spans, static_values,
        register_as, memory_as, immediate_as, deferral_as, byte_pointer_max_bits,
        cell_pointer_max_bits, initial_pc, initial_timestamp,
        endpoint_kind, program, memory, write_masks, field_values, error
    );
    return CHECK_KERNEL();
}
