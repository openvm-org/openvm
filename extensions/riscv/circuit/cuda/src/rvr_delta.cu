#include "primitives/buffer_view.cuh"
#include "riscv/cores/load.cuh"
#include "riscv/cores/load_sign_extend.cuh"
#include "riscv/cores/store.cuh"
#include "riscv/rvr_compact.cuh"
#include "fp.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>
#include <stdint.h>

using namespace riscv;

namespace {

static constexpr uint32_t DELTA_STRIDE = 24;
static constexpr uint32_t PC_STEP = 4;
static constexpr uint8_t INVALID_AIR = UINT8_MAX;
static constexpr uint64_t INVALID_ADDRESS = UINT64_MAX;

enum DeltaPattern : uint8_t {
    DELTA_ALU3 = 0,
    DELTA_ALU3_REG = 1,
    DELTA_LOAD = 2,
    DELTA_STORE = 3,
    DELTA_BRANCH2 = 4,
    DELTA_WR1 = 5,
    DELTA_WR1_ALWAYS = 6,
    DELTA_RW1 = 7,
    DELTA_ADDI = 8,
    DELTA_HINT_STORE = 9,
};

struct DeltaRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint64_t v1;
    uint64_t v2;
};
static_assert(sizeof(DeltaRecord) == DELTA_STRIDE, "delta record size drift");

struct MemoryLogEntry {
    uint32_t timestamp;
    uint32_t prev_timestamp;
    uint8_t kind;
    uint8_t addr_space;
    uint8_t width;
    uint8_t _pad0;
    uint32_t _pad1;
    uint64_t address;
    uint64_t value;
    uint64_t prev_value;
};
static_assert(sizeof(MemoryLogEntry) == 40, "memory log size drift");

struct DeltaMemoryLogEntry {
    uint32_t timestamp;
    uint32_t address;
    uint64_t value;
    uint8_t kind;
    uint8_t addr_space;
    uint8_t width;
    uint8_t complete;
    uint32_t reserved;
};
static_assert(sizeof(DeltaMemoryLogEntry) == 24, "delta memory log size drift");

struct ResidualMemoryEvent {
    uint32_t timestamp;
    uint8_t kind;
    uint8_t addr_space;
    uint64_t address;
    uint64_t value;
};

struct ProgramLogEntry {
    uint32_t timestamp;
    uint32_t pc_and_flags;
    uint64_t write_value;
};
static_assert(sizeof(ProgramLogEntry) == 16, "program log size drift");

struct ProgramRunEntry {
    uint32_t first_pc;
    uint32_t instruction_count;
    uint32_t chronology_offset;
    uint32_t complete;
};
static_assert(sizeof(ProgramRunEntry) == 16, "program run size drift");

struct DeviceProgramEntry {
    uint32_t pc;
    uint32_t filtered_index;
};
static_assert(sizeof(DeviceProgramEntry) == 8, "device program entry size drift");

static constexpr uint32_t PROGRAM_WRITE_COMPLETE = 1u;
static constexpr uint32_t PROGRAM_CROSSING_RESIDUAL = 2u;

struct DeviceInitialMemory {
    uint64_t base;
    uint64_t len;
    uint32_t cell_size;
    uint32_t reserved;
};
static_assert(sizeof(DeviceInitialMemory) == 24, "initial memory descriptor size drift");

struct TouchedMemoryRecord {
    uint32_t addr_space;
    uint32_t block_ptr;
    uint32_t timestamp;
    uint32_t values[4];
};
static_assert(sizeof(TouchedMemoryRecord) == 28, "touched-memory output size drift");

enum ValueUpdate : uint8_t {
    VALUE_NONE = 0,
    VALUE_SET = 2,
    VALUE_STORE_PATCH = 3,
};

struct EventPayload {
    uint64_t address_key;
    uint64_t value;
    uint32_t timestamp;
    uint32_t output_index_plus_one;
    uint32_t write_record_plus_one;
    uint32_t residual_index_plus_one;
    uint8_t value_update;
    uint8_t width;
    uint8_t byte_offset;
    uint8_t _pad0;
};
static_assert(sizeof(EventPayload) == 40, "event payload size drift");

static constexpr uint8_t EXPECT_BEFORE_LOAD = 1;
static constexpr uint8_t EXPECT_BEFORE_STORE = 2;
static constexpr uint8_t EXPECT_AFTER_STORE = 3;

struct DeltaAirOutputDesc {
    uint64_t base;
    uint32_t count;
    uint32_t stride;
    uint32_t sorted_start;
    uint32_t kind;
};
static_assert(sizeof(DeltaAirOutputDesc) == 24, "delta output descriptor size drift");

enum DeltaAirKind : uint32_t {
    DELTA_KIND_ADD_SUB = 0,
    DELTA_KIND_BITWISE = 1,
    DELTA_KIND_LESS_THAN = 2,
    DELTA_KIND_SHIFT_LOGICAL = 3,
    DELTA_KIND_SHIFT_RIGHT_ARITHMETIC = 4,
    DELTA_KIND_ADD_SUB_W = 5,
    DELTA_KIND_SHIFT_W_LOGICAL = 6,
    DELTA_KIND_SHIFT_W_RIGHT_ARITHMETIC = 7,
    DELTA_KIND_LOAD_BYTE = 8,
    DELTA_KIND_LOAD_SIGN_EXTEND_BYTE = 9,
    DELTA_KIND_BRANCH_EQUAL = 10,
    DELTA_KIND_BRANCH_LESS_THAN = 11,
    DELTA_KIND_JAL_LUI = 12,
    DELTA_KIND_JALR = 13,
    DELTA_KIND_AUIPC = 14,
    DELTA_KIND_MUL = 15,
    DELTA_KIND_MULH = 16,
    DELTA_KIND_MUL_W = 17,
    DELTA_KIND_DIV_REM = 18,
    DELTA_KIND_DIV_REM_W = 19,
    DELTA_KIND_LOAD_HALFWORD = 20,
    DELTA_KIND_LOAD_WORD = 21,
    DELTA_KIND_LOAD_DOUBLEWORD = 22,
    DELTA_KIND_STORE_BYTE = 23,
    DELTA_KIND_STORE_HALFWORD = 24,
    DELTA_KIND_STORE_WORD = 25,
    DELTA_KIND_STORE_DOUBLEWORD = 26,
    DELTA_KIND_LOAD_SIGN_EXTEND_HALFWORD = 27,
    DELTA_KIND_LOAD_SIGN_EXTEND_WORD = 28,
    DELTA_KIND_ADDI = 29,
    DELTA_KIND_HINT_STORE = 30,
    DELTA_KIND_COUNT = 31,
};

__device__ __forceinline__ void fail(uint32_t *error, uint32_t code) {
    atomicCAS(error, 0u, code);
}

__device__ __forceinline__ uint64_t address_key(
    uint8_t addr_space, uint64_t address, uint32_t *error
) {
    if (address >> 56) {
        fail(error, 1);
        return INVALID_ADDRESS;
    }
    return (uint64_t(addr_space) << 56) | address;
}

__device__ __forceinline__ bool residual_memory_event(
    uint8_t const *memory,
    size_t index,
    size_t stride,
    ResidualMemoryEvent &event,
    uint32_t *error
) {
    if (stride == sizeof(MemoryLogEntry)) {
        MemoryLogEntry entry = reinterpret_cast<MemoryLogEntry const *>(memory)[index];
        event = {entry.timestamp, entry.kind, entry.addr_space, entry.address, entry.value};
        return true;
    }
    if (stride == sizeof(DeltaMemoryLogEntry)) {
        DeltaMemoryLogEntry entry =
            reinterpret_cast<DeltaMemoryLogEntry const *>(memory)[index];
        bool valid_width =
            entry.width == 1 || entry.width == 2 || entry.width == 4 || entry.width == 8 ||
            (entry.kind == 2 && entry.width == 0);
        if (entry.complete != 1 || entry.reserved != 0 || entry.kind > 2 ||
            !valid_width) {
            fail(error, 18);
            return false;
        }
        event = {
            entry.timestamp,
            entry.kind,
            entry.addr_space,
            uint64_t(entry.address),
            entry.value,
        };
        return true;
    }
    fail(error, 19);
    return false;
}

__device__ __forceinline__ bool operand_for_pc(
    uint32_t pc,
    uint32_t pc_base,
    RvrOperandEntry const *table,
    size_t operand_count,
    RvrOperandEntry &entry,
    uint32_t *error
) {
    if (pc < pc_base || ((pc - pc_base) % PC_STEP) != 0) {
        fail(error, 2);
        return false;
    }
    size_t slot = (pc - pc_base) / PC_STEP;
    if (slot >= operand_count) {
        fail(error, 3);
        return false;
    }
    entry = table[slot];
    if (entry.air_idx == INVALID_AIR || entry.access_pattern > DELTA_HINT_STORE) {
        fail(error, 4);
        return false;
    }
    return true;
}

__global__ void expand_program_runs(
    ProgramRunEntry const *runs,
    size_t run_count,
    size_t instruction_count,
    RvrOperandEntry const *table,
    size_t operand_count,
    uint32_t pc_base,
    uint32_t *frequencies,
    size_t frequency_count,
    DeviceProgramEntry *chronology,
    uint32_t *error
) {
    size_t run_index = blockIdx.x;
    if (run_index >= run_count) return;
    ProgramRunEntry run = runs[run_index];
    size_t begin = run.chronology_offset;
    size_t end = begin + run.instruction_count;
    if (run.complete != 1 || run.instruction_count == 0 || end < begin ||
        end > instruction_count) {
        fail(error, 24);
        return;
    }
    for (size_t local = threadIdx.x; local < run.instruction_count;
         local += blockDim.x) {
        uint64_t pc64 = uint64_t(run.first_pc) + uint64_t(local) * PC_STEP;
        if (pc64 > UINT32_MAX || pc64 < pc_base ||
            ((uint32_t(pc64) - pc_base) % PC_STEP) != 0) {
            fail(error, 25);
            continue;
        }
        size_t slot = (uint32_t(pc64) - pc_base) / PC_STEP;
        if (slot >= operand_count) {
            fail(error, 26);
            continue;
        }
        uint32_t filtered_index = table[slot].filtered_index;
        if (filtered_index >= frequency_count) {
            fail(error, 27);
            continue;
        }
        chronology[begin + local] = {
            uint32_t(pc64),
            filtered_index,
        };
        atomicAdd(&frequencies[filtered_index], 1u);
    }
}

__device__ __forceinline__ uint32_t delta_memory_effective_address(
    DeltaRecord const &record, RvrOperandEntry const &entry
) {
    int32_t imm = int32_t(entry.c & 0xffffu);
    if (entry.flags & RVR_OPERAND_FLAG_LS_IMM_SIGN) {
        imm = int32_t(int16_t(imm));
    }
    return uint32_t(record.v1) + uint32_t(imm);
}

__device__ __forceinline__ uint8_t delta_memory_width(
    RvrOperandEntry const &entry, uint32_t *error
) {
    switch (entry.local_opcode) {
    case 0:
    case 4: return 8;
    case 3:
    case 5:
    case 10: return 4;
    case 2:
    case 6:
    case 9: return 2;
    case 1:
    case 7:
    case 8: return 1;
    default:
        if (error != nullptr) fail(error, 13);
        return 0;
    }
}

__device__ __forceinline__ bool delta_crosses(
    DeltaRecord const &record, RvrOperandEntry const &entry, uint32_t *error
) {
    uint8_t width = delta_memory_width(entry, error);
    return (delta_memory_effective_address(record, entry) & 7u) + width > 8u;
}

__device__ __forceinline__ bool delta_access(
    DeltaRecord const &record,
    RvrOperandEntry const &entry,
    uint32_t slot,
    uint8_t &addr_space,
    uint64_t &address
) {
    auto reg = [&](uint32_t ptr) {
        addr_space = 1;
        address = uint64_t(ptr) & ~uint64_t(7);
        return true;
    };
    auto memory = [&]() {
        uint32_t effective = delta_memory_effective_address(record, entry);
        addr_space =
            (entry.flags & RVR_OPERAND_FLAG_LS_PUBLIC_VALUES) ? uint8_t(3) : uint8_t(2);
        address = uint64_t(effective & ~uint32_t(7));
        return true;
    };

    switch (entry.access_pattern) {
    case DELTA_ADDI:
        if (slot == 0) return reg(entry.b);
        if (slot == 1) return reg(entry.a);
        return false;
    case DELTA_ALU3:
        if (slot == 0) return reg(entry.b);
        if (slot == 1)
            return (entry.flags & RVR_OPERAND_FLAG_RS2_IMM) ? false : reg(entry.c);
        if (slot == 2) return reg(entry.a);
        return false;
    case DELTA_ALU3_REG:
        if (slot == 0) return reg(entry.b);
        if (slot == 1) return reg(entry.c);
        if (slot == 2) return reg(entry.a);
        return false;
    case DELTA_LOAD:
        if (slot == 0) return reg(entry.b);
        if (delta_crosses(record, entry, nullptr)) {
            if (slot == 3 && (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED)) return reg(entry.a);
            return false;
        }
        if (slot == 1) return memory();
        if (slot == (delta_memory_width(entry, nullptr) > 1 ? 3u : 2u) &&
            (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED))
            return reg(entry.a);
        return false;
    case DELTA_STORE:
        if (slot == 0) return reg(entry.b);
        if (slot == 1) return reg(entry.a);
        if (slot == 2 && !delta_crosses(record, entry, nullptr)) return memory();
        return false;
    case DELTA_BRANCH2:
        if (slot == 0) return reg(entry.a);
        if (slot == 1) return reg(entry.b);
        return false;
    case DELTA_WR1:
        return slot == 0 && (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED) && reg(entry.a);
    case DELTA_WR1_ALWAYS:
        return slot == 0 && reg(entry.a);
    case DELTA_RW1:
        if (slot == 0) return reg(entry.b);
        if (slot == 1 && (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED)) return reg(entry.a);
        return false;
    case DELTA_HINT_STORE: {
        uint32_t row = uint32_t(record.v1 >> 32) & 0x3ffu;
        if (slot == 0) return row == 0 && reg(entry.b);
        if (slot == 1) return row == 0 && entry.local_opcode != 0 && reg(entry.a);
        if (slot == 2) {
            addr_space = 2;
            address = uint32_t(record.v1);
            return true;
        }
        return false;
    }
    default:
        return false;
    }
}

__device__ __forceinline__ bool delta_write_access(
    DeltaRecord const &record, RvrOperandEntry const &entry, uint32_t slot,
    uint32_t *error
) {
    switch (entry.access_pattern) {
    case DELTA_ADDI: return slot == 1;
    case DELTA_ALU3:
    case DELTA_ALU3_REG:
    case DELTA_STORE: return slot == 2 && !delta_crosses(record, entry, error);
    case DELTA_LOAD:
        return slot == (delta_memory_width(entry, error) > 1 ? 3u : 2u) &&
               (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED);
    case DELTA_WR1:
        return slot == 0 && (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED);
    case DELTA_WR1_ALWAYS:
        return slot == 0;
    case DELTA_RW1:
        return slot == 1 && (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED);
    case DELTA_HINT_STORE: return slot == 2;
    default:
        return false;
    }
}

__device__ __forceinline__ uint64_t sign_extend_word(uint32_t value) {
    return uint64_t(int64_t(int32_t(value)));
}

__device__ __forceinline__ bool delta_divrem_result(
    uint8_t local_opcode, uint64_t b, uint64_t c, uint64_t &result
) {
    switch (local_opcode) {
    case 0: {
        int64_t lhs = int64_t(b), rhs = int64_t(c);
        if (rhs == 0)
            result = UINT64_MAX;
        else if (lhs == INT64_MIN && rhs == -1)
            result = uint64_t(lhs);
        else
            result = uint64_t(lhs / rhs);
        return true;
    }
    case 1:
        result = c == 0 ? UINT64_MAX : b / c;
        return true;
    case 2: {
        int64_t lhs = int64_t(b), rhs = int64_t(c);
        if (rhs == 0)
            result = b;
        else if (lhs == INT64_MIN && rhs == -1)
            result = 0;
        else
            result = uint64_t(lhs % rhs);
        return true;
    }
    case 3:
        result = c == 0 ? b : b % c;
        return true;
    default: return false;
    }
}

__device__ __forceinline__ bool delta_divrem_w_result(
    uint8_t local_opcode, uint64_t b64, uint64_t c64, uint64_t &result
) {
    uint32_t b = uint32_t(b64), c = uint32_t(c64), word;
    switch (local_opcode) {
    case 0: {
        int32_t lhs = int32_t(b), rhs = int32_t(c);
        if (rhs == 0)
            word = UINT32_MAX;
        else if (lhs == INT32_MIN && rhs == -1)
            word = uint32_t(lhs);
        else
            word = uint32_t(lhs / rhs);
        break;
    }
    case 1: word = c == 0 ? UINT32_MAX : b / c; break;
    case 2: {
        int32_t lhs = int32_t(b), rhs = int32_t(c);
        if (rhs == 0)
            word = b;
        else if (lhs == INT32_MIN && rhs == -1)
            word = 0;
        else
            word = uint32_t(lhs % rhs);
        break;
    }
    case 3: word = c == 0 ? b : b % c; break;
    default: return false;
    }
    result = sign_extend_word(word);
    return true;
}

__device__ __forceinline__ bool delta_post_write_value(
    DeltaRecord const &record,
    RvrOperandEntry const &entry,
    uint32_t kind,
    uint64_t &result
) {
    switch (kind) {
    case DELTA_KIND_ADDI:
        if (entry.local_opcode != 0) return false;
        result = record.v1 + record.v2;
        return true;
    case DELTA_KIND_ADD_SUB:
        if (entry.local_opcode == 0)
            result = record.v1 + record.v2;
        else if (entry.local_opcode == 1)
            result = record.v1 - record.v2;
        else
            return false;
        return true;
    case DELTA_KIND_BITWISE:
        if (entry.local_opcode == 2)
            result = record.v1 ^ record.v2;
        else if (entry.local_opcode == 3)
            result = record.v1 | record.v2;
        else if (entry.local_opcode == 4)
            result = record.v1 & record.v2;
        else
            return false;
        return true;
    case DELTA_KIND_LESS_THAN:
        if (entry.local_opcode == 0)
            result = int64_t(record.v1) < int64_t(record.v2);
        else if (entry.local_opcode == 1)
            result = record.v1 < record.v2;
        else
            return false;
        return true;
    case DELTA_KIND_SHIFT_LOGICAL: {
        uint32_t shamt = uint32_t(record.v2 & 63u);
        if (entry.local_opcode == 0)
            result = record.v1 << shamt;
        else if (entry.local_opcode == 1)
            result = record.v1 >> shamt;
        else
            return false;
        return true;
    }
    case DELTA_KIND_SHIFT_RIGHT_ARITHMETIC:
        if (entry.local_opcode != 2) return false;
        result = uint64_t(int64_t(record.v1) >> uint32_t(record.v2 & 63u));
        return true;
    case DELTA_KIND_ADD_SUB_W: {
        uint32_t word;
        if (entry.local_opcode == 0)
            word = uint32_t(record.v1) + uint32_t(record.v2);
        else if (entry.local_opcode == 1)
            word = uint32_t(record.v1) - uint32_t(record.v2);
        else
            return false;
        result = sign_extend_word(word);
        return true;
    }
    case DELTA_KIND_SHIFT_W_LOGICAL: {
        uint32_t shamt = uint32_t(record.v2 & 31u), word;
        if (entry.local_opcode == 0)
            word = uint32_t(record.v1) << shamt;
        else if (entry.local_opcode == 1)
            word = uint32_t(record.v1) >> shamt;
        else
            return false;
        result = sign_extend_word(word);
        return true;
    }
    case DELTA_KIND_SHIFT_W_RIGHT_ARITHMETIC:
        if (entry.local_opcode != 0) return false;
        result = sign_extend_word(
            uint32_t(int32_t(uint32_t(record.v1)) >> uint32_t(record.v2 & 31u))
        );
        return true;
    case DELTA_KIND_LOAD_BYTE:
    case DELTA_KIND_LOAD_HALFWORD:
    case DELTA_KIND_LOAD_WORD:
    case DELTA_KIND_LOAD_DOUBLEWORD:
    case DELTA_KIND_STORE_BYTE:
    case DELTA_KIND_STORE_HALFWORD:
    case DELTA_KIND_STORE_WORD:
    case DELTA_KIND_STORE_DOUBLEWORD:
    case DELTA_KIND_LOAD_SIGN_EXTEND_BYTE:
    case DELTA_KIND_LOAD_SIGN_EXTEND_HALFWORD:
    case DELTA_KIND_LOAD_SIGN_EXTEND_WORD: {
        if (!(entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED)) return false;
        if (delta_crosses(record, entry, nullptr)) {
            result = record.v2;
            return true;
        }
        uint32_t shift = (delta_memory_effective_address(record, entry) & 7u) * 8u;
        uint64_t shifted = record.v2 >> shift;
        switch (entry.local_opcode) {
        case 0: result = shifted; break;
        case 1: result = uint8_t(shifted); break;
        case 2: result = uint16_t(shifted); break;
        case 3: result = uint32_t(shifted); break;
        case 8: result = uint64_t(int64_t(int8_t(shifted))); break;
        case 9: result = uint64_t(int64_t(int16_t(shifted))); break;
        case 10: result = uint64_t(int64_t(int32_t(shifted))); break;
        default: return false;
        }
        return true;
    }
    case DELTA_KIND_JAL_LUI:
        if (!(entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED)) return false;
        result = (entry.flags & RVR_OPERAND_FLAG_IS_JAL)
                     ? uint64_t(record.from_pc + PC_STEP)
                     : uint64_t(int64_t(int32_t(entry.c << 12)));
        return true;
    case DELTA_KIND_JALR:
        if (!(entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED)) return false;
        result = uint64_t(record.from_pc + PC_STEP);
        return true;
    case DELTA_KIND_AUIPC:
        result = uint64_t(record.from_pc) + uint64_t(int64_t(int32_t(entry.c << 8)));
        return true;
    case DELTA_KIND_MUL: result = record.v1 * record.v2; return true;
    case DELTA_KIND_MULH:
        if (entry.local_opcode == 0)
            result = uint64_t(__mul64hi((long long)record.v1, (long long)record.v2));
        else if (entry.local_opcode == 1)
            result = __umul64hi(record.v1, record.v2) -
                     (int64_t(record.v1) < 0 ? record.v2 : 0);
        else if (entry.local_opcode == 2)
            result = __umul64hi(record.v1, record.v2);
        else
            return false;
        return true;
    case DELTA_KIND_MUL_W:
        result = sign_extend_word(uint32_t(record.v1) * uint32_t(record.v2));
        return true;
    case DELTA_KIND_DIV_REM:
        return delta_divrem_result(entry.local_opcode, record.v1, record.v2, result);
    case DELTA_KIND_DIV_REM_W:
        return delta_divrem_w_result(entry.local_opcode, record.v1, record.v2, result);
    default: return false;
    }
}

__global__ void build_events(
    DeltaRecord const *delta,
    size_t delta_count,
    uint8_t const *memory,
    size_t memory_count,
    size_t memory_stride,
    ProgramLogEntry const *program,
    size_t program_count,
    RvrOperandEntry const *table,
    size_t operand_count,
    uint32_t pc_base,
    uint8_t const *arena_native_flags,
    DeltaAirOutputDesc const *outputs,
    size_t num_airs,
    uint32_t *timestamp_keys,
    EventPayload *payloads,
    uint32_t *error
) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t delta_events = delta_count * 4;
    size_t memory_begin = delta_events;
    size_t program_begin = memory_begin + memory_count;
    size_t event_count = program_begin + program_count * 3;
    if (idx >= event_count) return;

    EventPayload payload{};
    payload.address_key = INVALID_ADDRESS;
    payload.timestamp = UINT32_MAX;
    if (idx < delta_events) {
        size_t record_idx = idx / 4;
        uint32_t access_slot = idx % 4;
        DeltaRecord record = delta[record_idx];
        RvrOperandEntry entry;
        if (operand_for_pc(record.from_pc, pc_base, table, operand_count, entry, error)) {
            uint8_t as;
            uint64_t address;
            if (delta_access(record, entry, access_slot, as, address)) {
                payload.address_key = address_key(as, address, error);
                payload.timestamp = record.from_timestamp + access_slot;
                payload.output_index_plus_one = uint32_t(idx + 1);
                if (delta_write_access(record, entry, access_slot, error)) {
                    payload.write_record_plus_one = uint32_t(record_idx + 1);
                    if (entry.access_pattern == DELTA_STORE) {
                        payload.value = record.v2;
                        payload.value_update = VALUE_STORE_PATCH;
                        payload.width = delta_memory_width(entry, error);
                        payload.byte_offset = uint8_t(
                            delta_memory_effective_address(record, entry) & 7u
                        );
                    } else if (entry.access_pattern == DELTA_HINT_STORE) {
                        payload.value = record.v2;
                        payload.value_update = VALUE_SET;
                    } else {
                        if (entry.air_idx >= num_airs ||
                            outputs[entry.air_idx].kind >= DELTA_KIND_COUNT ||
                            !delta_post_write_value(
                                record, entry, outputs[entry.air_idx].kind, payload.value
                            )) {
                            fail(error, 17);
                            return;
                        }
                        payload.value_update = VALUE_SET;
                    }
                }
            }
        }
    } else if (idx < program_begin) {
        ResidualMemoryEvent event;
        if (residual_memory_event(
                memory, idx - memory_begin, memory_stride, event, error
            )) {
            payload.address_key =
                address_key(event.addr_space, event.address & ~uint64_t(7), error);
            payload.timestamp = event.timestamp;
            payload.residual_index_plus_one = uint32_t(idx - memory_begin + 1);
            if (event.kind == 1) {
                payload.value = event.value;
                payload.value_update = VALUE_SET;
            }
        }
    } else {
        size_t local = idx - program_begin;
        ProgramLogEntry program_entry = program[local / 3];
        uint32_t access_slot = local % 3;
        uint32_t program_pc = program_entry.pc_and_flags & ~uint32_t(3);
        if (program_pc >= pc_base && ((program_pc - pc_base) % PC_STEP) == 0) {
            size_t table_slot = (program_pc - pc_base) / PC_STEP;
            if (table_slot < operand_count) {
                RvrOperandEntry entry = table[table_slot];
                if ((program_entry.pc_and_flags & PROGRAM_CROSSING_RESIDUAL) == 0 &&
                    entry.air_idx < num_airs && entry.access_pattern <= DELTA_ADDI &&
                    arena_native_flags[entry.air_idx]) {
                    DeltaRecord synthetic{program_pc, program_entry.timestamp, 0, 0};
                    uint8_t as;
                    uint64_t address;
                    if (delta_access(synthetic, entry, access_slot, as, address)) {
                        payload.address_key = address_key(as, address, error);
                        payload.timestamp = program_entry.timestamp + access_slot;
                        if (delta_write_access(synthetic, entry, access_slot, error)) {
                            if ((program_entry.pc_and_flags & PROGRAM_WRITE_COMPLETE) == 0) {
                                fail(error, 16);
                                return;
                            }
                            payload.value = program_entry.write_value;
                            payload.value_update = VALUE_SET;
                        }
                    }
                }
            }
        }
    }
    timestamp_keys[idx] = payload.timestamp;
    payloads[idx] = payload;
}

__global__ void extract_address_keys(
    EventPayload const *events, uint64_t *keys, EventPayload *payloads, size_t count
) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    EventPayload event = events[idx];
    keys[idx] = event.address_key;
    payloads[idx] = event;
}

__global__ void mark_address_group_starts(
    uint64_t const *keys, uint32_t *indices, uint8_t *flags, size_t count
) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    indices[idx] = uint32_t(idx);
    flags[idx] = keys[idx] != INVALID_ADDRESS && (idx == 0 || keys[idx - 1] != keys[idx]);
}

__global__ void validate_group_count(
    uint32_t const *group_count, size_t capacity, uint32_t *error
) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && size_t(*group_count) > capacity) {
        fail(error, 20);
    }
}

__global__ void replay_address_groups(
    uint64_t const *keys,
    EventPayload const *events,
    size_t count,
    uint32_t const *group_starts,
    uint32_t const *actual_group_count,
    size_t group_capacity,
    DeviceInitialMemory const *initial_memory,
    size_t initial_memory_count,
    TouchedMemoryRecord *touched_output,
    uint32_t *prev_timestamps,
    uint64_t *prev_values,
    uint32_t *memory_prev_timestamps,
    uint64_t *memory_prev_values,
    uint64_t const *expected_blocks,
    uint8_t const *expected_modes,
    size_t expected_count,
    uint32_t *error
) {
    size_t group_idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t group_count = size_t(*actual_group_count);
    if (group_count > group_capacity || group_idx >= group_count) return;
    size_t idx = group_starts[group_idx];

    uint32_t previous_timestamp = 0;
    uint64_t key = keys[idx];
    uint32_t address_space = uint32_t(key >> 56);
    uint64_t byte_address = key & ((uint64_t(1) << 56) - 1);
    if (address_space >= initial_memory_count) {
        fail(error, 22);
        return;
    }
    DeviceInitialMemory image = initial_memory[address_space];
    if (image.reserved != 0 || image.cell_size != 2 || image.base == 0 ||
        byte_address > image.len || image.len - byte_address < sizeof(uint64_t)) {
        fail(error, 23);
        return;
    }
    uint64_t current_value = *reinterpret_cast<uint64_t const *>(
        image.base + byte_address
    );
    for (size_t cursor = idx; cursor < count && keys[cursor] == keys[idx]; ++cursor) {
        EventPayload event = events[cursor];
        uint8_t expected_mode = 0;
        uint32_t access_slot = UINT32_MAX;
        size_t record_idx = SIZE_MAX;
        if (event.output_index_plus_one != 0) {
            size_t event_idx = event.output_index_plus_one - 1;
            record_idx = event_idx / 3;
            access_slot = uint32_t(event_idx % 3);
            if (record_idx < expected_count) expected_mode = expected_modes[record_idx];
        }
        if (expected_mode > EXPECT_AFTER_STORE) {
            fail(error, 24);
            return;
        }
        bool check_before =
            (expected_mode == EXPECT_BEFORE_LOAD && access_slot == 1) ||
            (expected_mode == EXPECT_BEFORE_STORE && access_slot == 2);
        if (check_before && current_value != expected_blocks[record_idx]) {
            fail(error, 25);
            return;
        }
        if (event.output_index_plus_one != 0) {
            prev_timestamps[event.output_index_plus_one - 1] = previous_timestamp;
        }
        if (event.write_record_plus_one != 0) {
            prev_values[event.write_record_plus_one - 1] = current_value;
        }
        if (event.residual_index_plus_one != 0) {
            size_t residual = event.residual_index_plus_one - 1;
            memory_prev_timestamps[residual] = previous_timestamp;
            memory_prev_values[residual] = current_value;
        }
        if (event.value_update == VALUE_SET) {
            current_value = event.value;
        } else if (event.value_update == VALUE_STORE_PATCH) {
            uint32_t shift = uint32_t(event.byte_offset) * 8u;
            uint64_t mask = event.width == 8
                                ? UINT64_MAX
                                : ((uint64_t(1) << (uint32_t(event.width) * 8u)) - 1u);
            current_value = (current_value & ~(mask << shift)) |
                            ((event.value & mask) << shift);
        } else if (event.value_update != VALUE_NONE) {
            fail(error, 16);
            return;
        }
        if (expected_mode == EXPECT_AFTER_STORE && access_slot == 2 &&
            current_value != expected_blocks[record_idx]) {
            fail(error, 26);
            return;
        }
        previous_timestamp = event.timestamp;
    }
    if (byte_address > uint64_t(UINT32_MAX) * 2u + 1u) {
        fail(error, 21);
        return;
    }
    TouchedMemoryRecord record;
    record.addr_space = address_space;
    record.block_ptr = uint32_t(byte_address / 2u);
    record.timestamp = previous_timestamp;
    record.values[0] = Fp(uint16_t(current_value)).asRaw();
    record.values[1] = Fp(uint16_t(current_value >> 16)).asRaw();
    record.values[2] = Fp(uint16_t(current_value >> 32)).asRaw();
    record.values[3] = Fp(uint16_t(current_value >> 48)).asRaw();
    touched_output[group_idx] = record;
}

__global__ void build_air_keys(
    DeltaRecord const *delta,
    size_t count,
    RvrOperandEntry const *table,
    size_t operand_count,
    uint32_t pc_base,
    uint8_t *air_keys,
    uint32_t *indices,
    uint32_t *error
) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    RvrOperandEntry entry;
    if (!operand_for_pc(delta[idx].from_pc, pc_base, table, operand_count, entry, error)) {
        air_keys[idx] = INVALID_AIR;
    } else {
        air_keys[idx] = entry.air_idx;
    }
    indices[idx] = uint32_t(idx);
}

__device__ __forceinline__ void store_u32(uint8_t *dst, uint32_t value) {
    *reinterpret_cast<uint32_t *>(dst) = value;
}

__device__ __forceinline__ void store_u64_words(uint8_t *dst, uint64_t value) {
    store_u32(dst, uint32_t(value));
    store_u32(dst + 4, uint32_t(value >> 32));
}

__device__ __forceinline__ bool multi_block_kind(uint32_t kind) {
    return kind == DELTA_KIND_LOAD_HALFWORD || kind == DELTA_KIND_LOAD_WORD ||
           kind == DELTA_KIND_LOAD_DOUBLEWORD || kind == DELTA_KIND_STORE_HALFWORD ||
           kind == DELTA_KIND_STORE_WORD || kind == DELTA_KIND_STORE_DOUBLEWORD ||
           kind == DELTA_KIND_LOAD_SIGN_EXTEND_HALFWORD ||
           kind == DELTA_KIND_LOAD_SIGN_EXTEND_WORD;
}

__device__ __forceinline__ bool find_residual_block(
    uint8_t const *memory,
    size_t memory_count,
    size_t memory_stride,
    uint32_t timestamp,
    uint8_t kind,
    uint8_t addr_space,
    uint64_t address,
    uint32_t const *memory_prev_timestamps,
    uint64_t const *memory_prev_values,
    ResidualMemoryEvent &event,
    uint32_t &prev_timestamp,
    uint64_t &prev_value,
    uint32_t *error
) {
    for (size_t i = 0; i < memory_count; ++i) {
        ResidualMemoryEvent candidate;
        if (!residual_memory_event(memory, i, memory_stride, candidate, error)) continue;
        if (candidate.timestamp == timestamp && candidate.kind == kind &&
            candidate.addr_space == addr_space && candidate.address == address) {
            event = candidate;
            prev_timestamp = memory_prev_timestamps[i];
            prev_value = memory_prev_values[i];
            return true;
        }
    }
    fail(error, 28);
    return false;
}

__device__ __forceinline__ void u64_to_u16(uint64_t value, uint16_t (&out)[BLOCK_FE_WIDTH]) {
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; ++i) out[i] = uint16_t(value >> (16 * i));
}

__device__ __forceinline__ uint64_t compact_u64(uint32_t const (&words)[2]) {
    return uint64_t(words[0]) | (uint64_t(words[1]) << 32);
}

__device__ __forceinline__ bool find_full_memory_block(
    MemoryLogEntry const *memory,
    size_t memory_count,
    uint32_t timestamp,
    uint8_t kind,
    uint8_t addr_space,
    uint64_t address,
    MemoryLogEntry &event,
    uint32_t *error
) {
    for (size_t i = 0; i < memory_count; ++i) {
        MemoryLogEntry candidate = memory[i];
        if (candidate.timestamp == timestamp && candidate.kind == kind &&
            candidate.addr_space == addr_space && candidate.address == address &&
            candidate.width == 8) {
            event = candidate;
            return true;
        }
    }
    fail(error, 31);
    return false;
}

__device__ __forceinline__ Rv64LoadMultiByteAdapterRecord delta_load_adapter(
    DeltaRecord const &record,
    RvrOperandEntry const &entry,
    uint32_t const *record_prev,
    uint64_t write_prev_value
) {
    Rv64LoadMultiByteAdapterRecord out{};
    out.from_pc = record.from_pc;
    out.from_timestamp = record.from_timestamp;
    out.rs1_val = uint32_t(record.v1);
    out.rs1_aux_record.prev_timestamp = record_prev[0];
    out.read_data_aux[0].prev_timestamp = record_prev[1];
    out.read_data_aux[1].prev_timestamp = UINT32_MAX;
    out.imm = uint16_t(entry.c);
    out.imm_sign = (entry.flags & RVR_OPERAND_FLAG_LS_IMM_SIGN) != 0;
    out.rs1_ptr = uint8_t(entry.b);
    out.rd_ptr = (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED) ? uint8_t(entry.a) : UINT8_MAX;
    if (out.rd_ptr != UINT8_MAX) {
        out.write_prev_timestamp = record_prev[3];
        u64_to_u16(write_prev_value, out.write_prev_data);
    }
    return out;
}

__device__ __forceinline__ Rv64StoreMultiByteAdapterRecord delta_store_adapter(
    DeltaRecord const &record,
    RvrOperandEntry const &entry,
    uint32_t const *record_prev
) {
    Rv64StoreMultiByteAdapterRecord out{};
    out.from_pc = record.from_pc;
    out.from_timestamp = record.from_timestamp;
    out.rs1_val = uint32_t(record.v1);
    out.rs1_aux_record.prev_timestamp = record_prev[0];
    out.read_data_aux.prev_timestamp = record_prev[1];
    out.write_prev_timestamps[0] = record_prev[2];
    out.write_prev_timestamps[1] = UINT32_MAX;
    out.imm = uint16_t(entry.c);
    out.rs1_ptr = uint8_t(entry.b);
    out.rs2_ptr = uint8_t(entry.a);
    out.imm_sign = (entry.flags & RVR_OPERAND_FLAG_LS_IMM_SIGN) != 0;
    out.mem_as = (entry.flags & RVR_OPERAND_FLAG_LS_PUBLIC_VALUES) ? 3 : 2;
    return out;
}

__global__ void partition_records(
    DeltaRecord const *delta,
    uint32_t const *prev,
    uint64_t const *prev_values,
    uint8_t const *memory,
    size_t memory_count,
    size_t memory_stride,
    uint32_t const *memory_prev_timestamps,
    uint64_t const *memory_prev_values,
    uint8_t const *sorted_airs,
    uint32_t const *sorted_indices,
    size_t count,
    RvrOperandEntry const *table,
    size_t operand_count,
    uint32_t pc_base,
    DeltaAirOutputDesc const *outputs,
    size_t num_airs,
    uint32_t *error
) {
    size_t sorted_idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sorted_idx >= count) return;
    uint8_t air = sorted_airs[sorted_idx];
    if (air >= num_airs) {
        fail(error, 5);
        return;
    }
    DeltaAirOutputDesc desc = outputs[air];
    if (!desc.base || sorted_idx < desc.sorted_start ||
        sorted_idx - desc.sorted_start >= desc.count) {
        fail(error, 6);
        return;
    }
    uint32_t record_idx = sorted_indices[sorted_idx];
    DeltaRecord record = delta[record_idx];
    RvrOperandEntry entry;
    if (!operand_for_pc(record.from_pc, pc_base, table, operand_count, entry, error) ||
        entry.air_idx != air) {
        fail(error, 7);
        return;
    }
    uint8_t *dst = reinterpret_cast<uint8_t *>(desc.base) +
                   (sorted_idx - desc.sorted_start) * desc.stride;
    uint32_t const *record_prev = prev + size_t(record_idx) * 4;
    uint64_t write_prev_value = prev_values[record_idx];
    if (multi_block_kind(desc.kind)) {
        if (desc.stride != 60) {
            fail(error, 29);
            return;
        }
        uint8_t access_width = delta_memory_width(entry, error);
        uint32_t effective = delta_memory_effective_address(record, entry);
        uint64_t block0_addr = uint64_t(effective & ~uint32_t(7));
        bool crosses = (effective & 7u) + access_width > 8u;
        uint8_t addr_space =
            (entry.flags & RVR_OPERAND_FLAG_LS_PUBLIC_VALUES) ? uint8_t(3) : uint8_t(2);
        if (entry.access_pattern == DELTA_LOAD) {
            Rv64LoadMultiByteAdapterRecord adapter =
                delta_load_adapter(record, entry, record_prev, write_prev_value);
            uint64_t block0 = record.v2;
            uint64_t block1 = 0;
            if (crosses) {
                ResidualMemoryEvent event0, event1;
                uint32_t prev_ts0, prev_ts1;
                uint64_t ignored0, ignored1;
                if (!find_residual_block(
                        memory, memory_count, memory_stride, record.from_timestamp + 1,
                        0, addr_space, block0_addr, memory_prev_timestamps,
                        memory_prev_values, event0, prev_ts0, ignored0, error
                    ) ||
                    !find_residual_block(
                        memory, memory_count, memory_stride, record.from_timestamp + 2,
                        0, addr_space, block0_addr + 8, memory_prev_timestamps,
                        memory_prev_values, event1, prev_ts1, ignored1, error
                    ))
                    return;
                adapter.read_data_aux[0].prev_timestamp = prev_ts0;
                adapter.read_data_aux[1].prev_timestamp = prev_ts1;
                block0 = event0.value;
                block1 = event1.value;
            }
            if (desc.kind == DELTA_KIND_LOAD_SIGN_EXTEND_HALFWORD ||
                desc.kind == DELTA_KIND_LOAD_SIGN_EXTEND_WORD) {
                Rv64LoadSignExtendRecord full{};
                full.adapter = adapter;
                u64_to_u16(block0, full.core.read_data[0]);
                u64_to_u16(block1, full.core.read_data[1]);
                *reinterpret_cast<Rv64LoadSignExtendRecord *>(dst) = full;
            } else {
                Rv64LoadRecord full{};
                full.adapter = adapter;
                u64_to_u16(block0, full.core.read_data[0]);
                u64_to_u16(block1, full.core.read_data[1]);
                *reinterpret_cast<Rv64LoadRecord *>(dst) = full;
            }
        } else if (entry.access_pattern == DELTA_STORE) {
            Rv64StoreMultiByteAdapterRecord adapter =
                delta_store_adapter(record, entry, record_prev);
            uint64_t prev0 = write_prev_value;
            uint64_t prev1 = 0;
            if (crosses) {
                ResidualMemoryEvent event0, event1;
                uint32_t prev_ts0, prev_ts1;
                if (!find_residual_block(
                        memory, memory_count, memory_stride, record.from_timestamp + 2,
                        1, addr_space, block0_addr, memory_prev_timestamps,
                        memory_prev_values, event0, prev_ts0, prev0, error
                    ) ||
                    !find_residual_block(
                        memory, memory_count, memory_stride, record.from_timestamp + 3,
                        1, addr_space, block0_addr + 8, memory_prev_timestamps,
                        memory_prev_values, event1, prev_ts1, prev1, error
                    ))
                    return;
                adapter.write_prev_timestamps[0] = prev_ts0;
                adapter.write_prev_timestamps[1] = prev_ts1;
            }
            Rv64StoreRecord full{};
            full.adapter = adapter;
            u64_to_u16(record.v2, full.core.read_data);
            u64_to_u16(prev0, full.core.prev_data[0]);
            u64_to_u16(prev1, full.core.prev_data[1]);
            *reinterpret_cast<Rv64StoreRecord *>(dst) = full;
        } else {
            fail(error, 30);
        }
        return;
    }
    store_u32(dst, record.from_pc);
    store_u32(dst + 4, record.from_timestamp);
    switch (entry.access_pattern) {
    case DELTA_ADDI:
        if (desc.stride != sizeof(RvrAlu3Compact)) {
            fail(error, 8);
            return;
        }
        store_u32(dst + 8, record_prev[0]);
        store_u32(dst + 12, 0u);
        store_u32(dst + 16, record_prev[1]);
        store_u64_words(dst + 20, write_prev_value);
        store_u64_words(dst + 28, record.v1);
        store_u64_words(dst + 36, record.v2);
        break;
    case DELTA_ALU3:
    case DELTA_ALU3_REG:
    case DELTA_STORE:
        if (desc.stride != sizeof(RvrAlu3Compact)) {
            fail(error, 8);
            return;
        }
        store_u32(dst + 8, record_prev[0]);
        store_u32(dst + 12, record_prev[1]);
        store_u32(dst + 16, record_prev[2]);
        store_u64_words(dst + 20, write_prev_value);
        store_u64_words(dst + 28, record.v1);
        store_u64_words(dst + 36, record.v2);
        break;
    case DELTA_LOAD:
        if (desc.stride != sizeof(RvrAlu3Compact)) {
            fail(error, 8);
            return;
        }
        store_u32(dst + 8, record_prev[0]);
        store_u32(dst + 12, record_prev[1]);
        store_u32(dst + 16, record_prev[delta_memory_width(entry, error) > 1 ? 3 : 2]);
        store_u64_words(dst + 20, write_prev_value);
        store_u64_words(dst + 28, record.v1);
        store_u64_words(dst + 36, record.v2);
        break;
    case DELTA_BRANCH2:
        if (desc.stride != sizeof(RvrBranch2Compact)) {
            fail(error, 9);
            return;
        }
        store_u32(dst + 8, record_prev[0]);
        store_u32(dst + 12, record_prev[1]);
        store_u64_words(dst + 16, record.v1);
        store_u64_words(dst + 24, record.v2);
        break;
    case DELTA_WR1:
    case DELTA_WR1_ALWAYS:
        if (desc.stride != sizeof(RvrWr1Compact)) {
            fail(error, 10);
            return;
        }
        store_u32(dst + 8, record_prev[0]);
        store_u64_words(dst + 12, write_prev_value);
        break;
    case DELTA_RW1:
        if (desc.stride != sizeof(RvrRw1Compact)) {
            fail(error, 11);
            return;
        }
        store_u32(dst + 8, record_prev[0]);
        store_u32(dst + 12, record_prev[1]);
        store_u64_words(dst + 16, record.v1);
        store_u64_words(dst + 24, write_prev_value);
        break;
    case DELTA_HINT_STORE: {
        if (desc.stride != 64 || desc.kind != DELTA_KIND_HINT_STORE) {
            fail(error, 13);
            return;
        }
        uint32_t row = uint32_t(record.v1 >> 32) & 0x3ffu;
        uint32_t num_words = uint32_t(record.v1 >> 42) & 0x3ffu;
        uint32_t address = uint32_t(record.v1);
        if (num_words == 0 || row >= num_words || address < row * 8u) {
            fail(error, 14);
            return;
        }
        store_u32(dst, record.from_pc);
        store_u32(dst + 4, record.from_timestamp - row * 3u);
        store_u32(dst + 8, row);
        store_u32(dst + 12, num_words);
        store_u32(dst + 16, entry.b);
        store_u32(dst + 20, address - row * 8u);
        store_u32(dst + 24, record_prev[0]);
        store_u32(dst + 28, entry.local_opcode == 0 ? UINT32_MAX : entry.a);
        store_u32(dst + 32, record_prev[1]);
        store_u32(dst + 36, record_prev[2]);
        store_u64_words(dst + 40, write_prev_value);
        store_u64_words(dst + 48, record.v2);
        store_u64_words(dst + 56, 0);
        break;
    }
    default:
        fail(error, 12);
    }
}

__global__ void expand_compact_multiblock_records(
    RvrAlu3Compact const *records,
    size_t count,
    MemoryLogEntry const *memory,
    size_t memory_count,
    RvrOperandEntry const *table,
    size_t operand_count,
    uint32_t pc_base,
    uint32_t kind,
    uint8_t *output,
    uint32_t *error
) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    RvrAlu3Compact record = records[idx];
    RvrOperandEntry entry;
    if (!operand_for_pc(record.from_pc, pc_base, table, operand_count, entry, error) ||
        !multi_block_kind(kind)) {
        fail(error, 32);
        return;
    }
    uint8_t width = delta_memory_width(entry, error);
    uint32_t rs1 = record.b[0];
    int32_t offset = (entry.flags & RVR_OPERAND_FLAG_LS_IMM_SIGN)
                         ? int32_t(int16_t(entry.c))
                         : int32_t(entry.c);
    uint32_t effective = rs1 + uint32_t(offset);
    uint64_t block0_address = uint64_t(effective & ~uint32_t(7));
    bool crosses = (effective & 7u) + width > 8u;
    uint8_t addr_space =
        (entry.flags & RVR_OPERAND_FLAG_LS_PUBLIC_VALUES) ? uint8_t(3) : uint8_t(2);
    uint8_t *dst = output + idx * 60;
    if (entry.access_pattern == DELTA_LOAD) {
        Rv64LoadMultiByteAdapterRecord adapter = rvr_decode_alu3_load_multi(record, entry);
        uint64_t block0 = compact_u64(record.c);
        uint64_t block1 = 0;
        if (crosses) {
            MemoryLogEntry event0, event1;
            if (!find_full_memory_block(
                    memory, memory_count, record.from_timestamp + 1, 0, addr_space,
                    block0_address, event0, error
                ) ||
                !find_full_memory_block(
                    memory, memory_count, record.from_timestamp + 2, 0, addr_space,
                    block0_address + 8, event1, error
                ))
                return;
            adapter.read_data_aux[0].prev_timestamp = event0.prev_timestamp;
            adapter.read_data_aux[1].prev_timestamp = event1.prev_timestamp;
            block0 = event0.value;
            block1 = event1.value;
        }
        if (kind == DELTA_KIND_LOAD_SIGN_EXTEND_HALFWORD ||
            kind == DELTA_KIND_LOAD_SIGN_EXTEND_WORD) {
            Rv64LoadSignExtendRecord full{};
            full.adapter = adapter;
            u64_to_u16(block0, full.core.read_data[0]);
            u64_to_u16(block1, full.core.read_data[1]);
            *reinterpret_cast<Rv64LoadSignExtendRecord *>(dst) = full;
        } else {
            Rv64LoadRecord full{};
            full.adapter = adapter;
            u64_to_u16(block0, full.core.read_data[0]);
            u64_to_u16(block1, full.core.read_data[1]);
            *reinterpret_cast<Rv64LoadRecord *>(dst) = full;
        }
    } else if (entry.access_pattern == DELTA_STORE) {
        Rv64StoreMultiByteAdapterRecord adapter = rvr_decode_alu3_store_multi(record, entry);
        uint64_t prev0 = compact_u64(record.write_prev_data);
        uint64_t prev1 = 0;
        if (crosses) {
            MemoryLogEntry event0, event1;
            if (!find_full_memory_block(
                    memory, memory_count, record.from_timestamp + 2, 1, addr_space,
                    block0_address, event0, error
                ) ||
                !find_full_memory_block(
                    memory, memory_count, record.from_timestamp + 3, 1, addr_space,
                    block0_address + 8, event1, error
                ))
                return;
            adapter.write_prev_timestamps[0] = event0.prev_timestamp;
            adapter.write_prev_timestamps[1] = event1.prev_timestamp;
            prev0 = event0.prev_value;
            prev1 = event1.prev_value;
        }
        Rv64StoreRecord full{};
        full.adapter = adapter;
        u64_to_u16(compact_u64(record.c), full.core.read_data);
        u64_to_u16(prev0, full.core.prev_data[0]);
        u64_to_u16(prev1, full.core.prev_data[1]);
        *reinterpret_cast<Rv64StoreRecord *>(dst) = full;
    } else {
        fail(error, 33);
    }
}

} // namespace

extern "C" int _rvr_expand_compact_multiblock(
    DeviceBufferConstView<uint8_t> d_records,
    size_t record_count,
    DeviceBufferConstView<uint8_t> d_memory,
    size_t memory_count,
    RvrOperandEntry const *d_operand_table,
    size_t operand_count,
    uint32_t pc_base,
    uint32_t kind,
    DeviceRawBufferConstView d_output,
    uint32_t *d_error,
    cudaStream_t stream
) {
    if (d_records.size != record_count * sizeof(RvrAlu3Compact) ||
        d_memory.size != memory_count * sizeof(MemoryLogEntry) ||
        d_output.size != record_count * 60 || d_error == nullptr ||
        (record_count != 0 && (d_records.ptr == nullptr || d_output.ptr == 0)) ||
        (memory_count != 0 && d_memory.ptr == nullptr)) {
        return int(cudaErrorInvalidValue);
    }
    if (record_count == 0) return int(cudaSuccess);
    dim3 block(256);
    dim3 grid((record_count + block.x - 1) / block.x);
    expand_compact_multiblock_records<<<grid, block, 0, stream>>>(
        reinterpret_cast<RvrAlu3Compact const *>(d_records.ptr),
        record_count,
        reinterpret_cast<MemoryLogEntry const *>(d_memory.ptr),
        memory_count,
        d_operand_table,
        operand_count,
        pc_base,
        kind,
        reinterpret_cast<uint8_t *>(d_output.ptr),
        d_error
    );
    return int(cudaGetLastError());
}

extern "C" int _rvr_delta_predecode(
    DeviceBufferConstView<uint8_t> d_delta_bytes,
    size_t delta_count,
    DeviceBufferConstView<uint8_t> d_memory_bytes,
    size_t memory_count,
    size_t memory_stride,
    DeviceBufferConstView<uint8_t> d_program_bytes,
    size_t program_count,
    ProgramRunEntry const *d_program_runs,
    size_t program_run_count,
    size_t program_instruction_count,
    uint32_t *d_program_frequencies,
    size_t program_frequency_count,
    DeviceProgramEntry *d_program_chronology,
    DeviceBufferConstView<uint8_t> d_initial_memory_bytes,
    size_t initial_memory_count,
    DeviceRawBufferConstView d_touched_output_bytes,
    uint32_t *d_touched_count,
    uint32_t *d_memory_prev_timestamps,
    uint64_t *d_memory_prev_values,
    RvrOperandEntry const *d_operand_table,
    size_t operand_count,
    uint32_t pc_base,
    uint8_t const *d_arena_native_flags,
    size_t num_airs,
    DeltaAirOutputDesc const *d_outputs,
    DeviceBufferConstView<uint64_t> d_expected_blocks,
    DeviceBufferConstView<uint8_t> d_expected_modes,
    uint32_t *d_error,
    cudaStream_t stream
) {
    if (delta_count > UINT32_MAX / 4 ||
        d_delta_bytes.size != delta_count * sizeof(DeltaRecord) ||
        (memory_stride != sizeof(MemoryLogEntry) &&
         memory_stride != sizeof(DeltaMemoryLogEntry)) ||
        d_memory_bytes.size != memory_count * memory_stride ||
        d_program_bytes.size != program_count * sizeof(ProgramLogEntry) ||
        (program_run_count != 0 && d_program_runs == nullptr) ||
        (program_frequency_count != 0 && d_program_frequencies == nullptr) ||
        (program_instruction_count != 0 && d_program_chronology == nullptr) ||
        d_initial_memory_bytes.size != initial_memory_count * sizeof(DeviceInitialMemory) ||
        d_touched_output_bytes.size <
            (delta_count * 4 + memory_count + program_count * 3) *
                sizeof(TouchedMemoryRecord) ||
        (memory_count != 0 &&
         (d_memory_prev_timestamps == nullptr || d_memory_prev_values == nullptr)) ||
        d_expected_blocks.size != d_expected_modes.size * sizeof(uint64_t) ||
        (d_expected_modes.size != 0 && d_expected_modes.size != delta_count) ||
        d_touched_count == nullptr) {
        return int(cudaErrorInvalidValue);
    }
    if (program_frequency_count != 0) {
        cudaError_t error = cudaMemsetAsync(
            d_program_frequencies, 0, program_frequency_count * sizeof(uint32_t), stream
        );
        if (error != cudaSuccess) return int(error);
    }
    if (program_run_count != 0) {
        dim3 block(256);
        dim3 grid(program_run_count);
        expand_program_runs<<<grid, block, 0, stream>>>(
            d_program_runs,
            program_run_count,
            program_instruction_count,
            d_operand_table,
            operand_count,
            pc_base,
            d_program_frequencies,
            program_frequency_count,
            d_program_chronology,
            d_error
        );
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) return int(error);
    } else if (program_instruction_count != 0) {
        return int(cudaErrorInvalidValue);
    }
    size_t event_count =
        delta_count * 4 + memory_count + program_count * 3;
    if (event_count == 0) {
        return int(cudaMemsetAsync(d_touched_count, 0, sizeof(uint32_t), stream));
    }

    uint32_t *ts_a = nullptr, *ts_b = nullptr, *indices_a = nullptr, *indices_b = nullptr;
    uint32_t *event_indices = nullptr, *group_starts = nullptr;
    uint64_t *addr_a = nullptr, *addr_b = nullptr;
    uint8_t *air_a = nullptr, *air_b = nullptr, *group_flags = nullptr;
    EventPayload *events_a = nullptr, *events_b = nullptr;
    uint32_t *prev = nullptr;
    uint64_t *prev_values = nullptr;
    void *temp = nullptr;
    size_t ts_temp = 0, addr_temp = 0, air_temp = 0, select_temp = 0;
    int result = 0;

#define CUDA_TRY(expr)                                                                            \
    do {                                                                                          \
        cudaError_t _err = (expr);                                                                \
        if (_err != cudaSuccess) {                                                                \
            result = int(_err);                                                                   \
            goto cleanup;                                                                         \
        }                                                                                         \
    } while (0)
#define CUDA_ALLOC(ptr, bytes) CUDA_TRY(cudaMallocAsync(reinterpret_cast<void **>(&(ptr)), bytes, stream))

    CUDA_ALLOC(ts_a, event_count * sizeof(uint32_t));
    CUDA_ALLOC(ts_b, event_count * sizeof(uint32_t));
    CUDA_ALLOC(events_a, event_count * sizeof(EventPayload));
    CUDA_ALLOC(events_b, event_count * sizeof(EventPayload));
    CUDA_ALLOC(addr_a, event_count * sizeof(uint64_t));
    CUDA_ALLOC(addr_b, event_count * sizeof(uint64_t));
    CUDA_ALLOC(event_indices, event_count * sizeof(uint32_t));
    CUDA_ALLOC(group_starts, event_count * sizeof(uint32_t));
    CUDA_ALLOC(group_flags, event_count * sizeof(uint8_t));
    CUDA_TRY(cudaMemsetAsync(d_touched_count, 0, sizeof(uint32_t), stream));
    if (delta_count != 0) {
        CUDA_ALLOC(prev, delta_count * 4 * sizeof(uint32_t));
        CUDA_ALLOC(prev_values, delta_count * sizeof(uint64_t));
        CUDA_ALLOC(air_a, delta_count * sizeof(uint8_t));
        CUDA_ALLOC(air_b, delta_count * sizeof(uint8_t));
        CUDA_ALLOC(indices_a, delta_count * sizeof(uint32_t));
        CUDA_ALLOC(indices_b, delta_count * sizeof(uint32_t));
        CUDA_TRY(cudaMemsetAsync(prev, 0, delta_count * 4 * sizeof(uint32_t), stream));
        CUDA_TRY(cudaMemsetAsync(prev_values, 0, delta_count * sizeof(uint64_t), stream));
    }
    if (memory_count != 0) {
        CUDA_TRY(cudaMemsetAsync(
            d_memory_prev_timestamps, 0, memory_count * sizeof(uint32_t), stream
        ));
        CUDA_TRY(cudaMemsetAsync(
            d_memory_prev_values, 0, memory_count * sizeof(uint64_t), stream
        ));
    }

    {
        dim3 block(256);
        dim3 grid((event_count + block.x - 1) / block.x);
        build_events<<<grid, block, 0, stream>>>(
            reinterpret_cast<DeltaRecord const *>(d_delta_bytes.ptr),
            delta_count,
            d_memory_bytes.ptr,
            memory_count,
            memory_stride,
            reinterpret_cast<ProgramLogEntry const *>(d_program_bytes.ptr),
            program_count,
            d_operand_table,
            operand_count,
            pc_base,
            d_arena_native_flags,
            d_outputs,
            num_airs,
            ts_a,
            events_a,
            d_error
        );
        CUDA_TRY(cudaGetLastError());
    }

    CUDA_TRY(cub::DeviceRadixSort::SortPairs(
        nullptr, ts_temp, ts_a, ts_b, events_a, events_b, event_count, 0, 32, stream
    ));
    CUDA_TRY(cub::DeviceRadixSort::SortPairs(
        nullptr, addr_temp, addr_a, addr_b, events_a, events_b, event_count, 0, 64, stream
    ));
    CUDA_TRY(cub::DeviceSelect::Flagged(
        nullptr,
        select_temp,
        event_indices,
        group_flags,
        group_starts,
        d_touched_count,
        event_count,
        stream
    ));
    if (delta_count != 0) {
        CUDA_TRY(cub::DeviceRadixSort::SortPairs(
            nullptr, air_temp, air_a, air_b, indices_a, indices_b, delta_count, 0, 8, stream
        ));
    }
    {
        size_t temp_bytes = ts_temp;
        if (addr_temp > temp_bytes) temp_bytes = addr_temp;
        if (air_temp > temp_bytes) temp_bytes = air_temp;
        if (select_temp > temp_bytes) temp_bytes = select_temp;
        CUDA_ALLOC(temp, temp_bytes);
    }
    CUDA_TRY(cub::DeviceRadixSort::SortPairs(
        temp, ts_temp, ts_a, ts_b, events_a, events_b, event_count, 0, 32, stream
    ));
    {
        dim3 block(256);
        dim3 grid((event_count + block.x - 1) / block.x);
        extract_address_keys<<<grid, block, 0, stream>>>(
            events_b, addr_a, events_a, event_count
        );
        CUDA_TRY(cudaGetLastError());
    }
    CUDA_TRY(cub::DeviceRadixSort::SortPairs(
        temp, addr_temp, addr_a, addr_b, events_a, events_b, event_count, 0, 64, stream
    ));
    {
        dim3 block(256);
        dim3 grid((event_count + block.x - 1) / block.x);
        mark_address_group_starts<<<grid, block, 0, stream>>>(
            addr_b, event_indices, group_flags, event_count
        );
        CUDA_TRY(cudaGetLastError());
    }
    CUDA_TRY(cub::DeviceSelect::Flagged(
        temp,
        select_temp,
        event_indices,
        group_flags,
        group_starts,
        d_touched_count,
        event_count,
        stream
    ));
    validate_group_count<<<1, 1, 0, stream>>>(d_touched_count, event_count, d_error);
    CUDA_TRY(cudaGetLastError());
    if (event_count != 0) {
        dim3 block(256);
        dim3 grid((event_count + block.x - 1) / block.x);
        replay_address_groups<<<grid, block, 0, stream>>>(
            addr_b,
            events_b,
            event_count,
            group_starts,
            d_touched_count,
            event_count,
            reinterpret_cast<DeviceInitialMemory const *>(d_initial_memory_bytes.ptr),
            initial_memory_count,
            reinterpret_cast<TouchedMemoryRecord *>(d_touched_output_bytes.ptr),
            prev,
            prev_values,
            d_memory_prev_timestamps,
            d_memory_prev_values,
            d_expected_blocks.ptr,
            d_expected_modes.ptr,
            d_expected_modes.len(),
            d_error
        );
        CUDA_TRY(cudaGetLastError());
    }
    if (delta_count != 0) {
        dim3 block(256);
        dim3 grid((delta_count + block.x - 1) / block.x);
        build_air_keys<<<grid, block, 0, stream>>>(
            reinterpret_cast<DeltaRecord const *>(d_delta_bytes.ptr),
            delta_count,
            d_operand_table,
            operand_count,
            pc_base,
            air_a,
            indices_a,
            d_error
        );
        CUDA_TRY(cudaGetLastError());
    }
    if (delta_count != 0) {
        CUDA_TRY(cub::DeviceRadixSort::SortPairs(
            temp, air_temp, air_a, air_b, indices_a, indices_b, delta_count, 0, 8, stream
        ));
        dim3 block(256);
        dim3 grid((delta_count + block.x - 1) / block.x);
        partition_records<<<grid, block, 0, stream>>>(
            reinterpret_cast<DeltaRecord const *>(d_delta_bytes.ptr),
            prev,
            prev_values,
            d_memory_bytes.ptr,
            memory_count,
            memory_stride,
            d_memory_prev_timestamps,
            d_memory_prev_values,
            air_b,
            indices_b,
            delta_count,
            d_operand_table,
            operand_count,
            pc_base,
            d_outputs,
            num_airs,
            d_error
        );
        CUDA_TRY(cudaGetLastError());
    }

cleanup:
    if (temp) cudaFreeAsync(temp, stream);
    if (indices_b) cudaFreeAsync(indices_b, stream);
    if (indices_a) cudaFreeAsync(indices_a, stream);
    if (group_starts) cudaFreeAsync(group_starts, stream);
    if (event_indices) cudaFreeAsync(event_indices, stream);
    if (group_flags) cudaFreeAsync(group_flags, stream);
    if (air_b) cudaFreeAsync(air_b, stream);
    if (air_a) cudaFreeAsync(air_a, stream);
    if (prev) cudaFreeAsync(prev, stream);
    if (prev_values) cudaFreeAsync(prev_values, stream);
    if (events_b) cudaFreeAsync(events_b, stream);
    if (events_a) cudaFreeAsync(events_a, stream);
    if (addr_b) cudaFreeAsync(addr_b, stream);
    if (addr_a) cudaFreeAsync(addr_a, stream);
    if (ts_b) cudaFreeAsync(ts_b, stream);
    if (ts_a) cudaFreeAsync(ts_a, stream);
    return result;

#undef CUDA_ALLOC
#undef CUDA_TRY
}
