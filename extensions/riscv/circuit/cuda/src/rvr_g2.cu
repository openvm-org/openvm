#include "fp.h"
#include "primitives/buffer_view.cuh"
#include "riscv/cores/load.cuh"
#include "riscv/cores/load_sign_extend.cuh"
#include "riscv/cores/store.cuh"
#include "riscv/rvr_compact.cuh"
#include "riscv/rvr_g2_trace.cuh"

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdint.h>

using namespace riscv;

namespace {

static constexpr uint16_t G2_FLAGS_COMMITTED_V1 = 0x000fu;
static constexpr uint16_t G2_RUN_BLOCK_ID = 0x0001u;
static constexpr uint16_t G2_RESIDUAL_CTRL = 0x0080u;
static constexpr uint16_t G2_RESIDUAL_TAG = 0x0081u;
static constexpr uint16_t G2_RESIDUAL_VALUE = 0x0082u;
static constexpr uint16_t G2_OPAQUE_EVENT_COUNT = 0x0083u;
static constexpr uint32_t G2_REQUIRED = 1u;
static constexpr uint32_t G2_REQUIRED_ATOMIC = 3u;
static constexpr uint32_t G2_LOAD_STORE_GROUP = 1u;
static constexpr uint32_t G2_RESIDUAL_GROUP = 2u;
static constexpr uint8_t G2_ADDI_PATTERN = 8u;
static constexpr uint8_t G2_LOAD_PATTERN = 2u;
static constexpr uint8_t G2_STORE_PATTERN = 3u;
static constexpr uint8_t G2_HINT_PATTERN = 9u;
static constexpr uint8_t G2_OPAQUE_PATTERN = 10u;
static constexpr uint8_t G2_OPAQUE_TIMESTAMP_PATTERN = 11u;
static constexpr uint8_t INVALID_AIR = UINT8_MAX;
static constexpr uint8_t EXPECT_NONE = 0u;
static constexpr uint8_t EXPECT_BEFORE_LOAD = 1u;
static constexpr uint8_t EXPECT_BEFORE_STORE = 2u;
static constexpr uint8_t EXPECT_AFTER_STORE = 3u;
static constexpr size_t CUDA_GRID_X_MAX = 0x7fffffffu;

struct G2SegmentHeaderV1 {
    uint8_t magic[8];
    uint16_t version;
    uint16_t header_bytes;
    uint16_t lane_count;
    uint16_t flags;
    uint32_t segment_id;
    uint32_t instruction_count;
    uint32_t run_count;
    uint32_t residual_event_count;
    uint8_t schema_fingerprint[32];
};
static_assert(sizeof(G2SegmentHeaderV1) == 64, "G2 header size drift");

struct G2LaneDescV1 {
    uint16_t kind;
    uint8_t elem_width;
    uint8_t encoding;
    uint32_t flags;
    uint32_t count;
    uint32_t payload_bytes;
    uint64_t offset;
    uint32_t group_id;
    uint32_t reserved;
};
static_assert(sizeof(G2LaneDescV1) == 32, "G2 lane descriptor size drift");

struct G2BlockEntryV1 {
    uint32_t program_slot;
    uint32_t instruction_count;
};
static_assert(sizeof(G2BlockEntryV1) == 8, "G2 block entry size drift");

struct G2ExpectedKindV1 {
    uint32_t kind;
    uint32_t air_idx;
    uint32_t count;
};
static_assert(sizeof(G2ExpectedKindV1) == 12, "G2 expected-kind size drift");

struct G2ExpectedOpaqueV1 {
    uint32_t lane_kind;
    uint32_t air_idx;
    uint32_t count;
    uint32_t payload_bytes;
    uint32_t stride;
};
static_assert(sizeof(G2ExpectedOpaqueV1) == 20, "G2 expected-opaque size drift");

struct DeltaAirOutputDesc {
    uint64_t base;
    uint32_t count;
    uint32_t stride;
    uint32_t sorted_start;
    uint32_t kind;
};
static_assert(sizeof(DeltaAirOutputDesc) == 24, "G2 output descriptor size drift");

enum G2TimelineAction : uint8_t {
    G2_EVENT_NONE = 0,
    G2_EVENT_READ_EXPECT = 1,
    G2_EVENT_READ_PATCH = 2,
    G2_EVENT_WRITE_SET = 3,
    G2_EVENT_STORE_PATCH = 4,
    G2_EVENT_READ_ANY = 5,
    G2_EVENT_WRITE_SET_EXPECT = 6,
};

static constexpr uint32_t G2_NO_RECORD = UINT32_MAX;
static constexpr uint64_t G2_INVALID_ADDRESS = UINT64_MAX;
static constexpr uint32_t G2_REGISTER_COUNT = 32;
static constexpr uint32_t G2_REGISTER_REPLAY_CHUNK = 4096;
static constexpr uint32_t G2_MAIN_MEMORY_ADDRESS_SPACE = 2;
static constexpr uint64_t G2_CONTINUATION_PAGE_BYTES = 4096;

struct DeviceInitialMemory {
    uint64_t base;
    uint64_t len;
    uint32_t cell_size;
    uint32_t reserved;
};
static_assert(sizeof(DeviceInitialMemory) == 24, "initial memory descriptor size drift");

struct DeltaRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint64_t v1;
    uint64_t v2;
};
static_assert(sizeof(DeltaRecord) == 24, "delta record size drift");

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
static_assert(sizeof(DeltaMemoryLogEntry) == 24, "delta memory-log size drift");

__device__ __forceinline__ void fail(uint32_t *error, uint32_t code) {
    if (*error == 0) *error = code;
}

__device__ __forceinline__ void fail_parallel(uint32_t *error, uint32_t code) {
    atomicCAS(error, 0u, code);
}

__device__ __forceinline__ uint64_t g2_address_key(
    uint8_t address_space, uint64_t address, uint32_t *error
) {
    if (address >> 56) {
        fail_parallel(error, 58);
        return G2_INVALID_ADDRESS;
    }
    return (uint64_t(address_space) << 56) | address;
}

__device__ __forceinline__ void g2_emit_timeline(
    G2TimelineEvent *timeline,
    size_t timeline_capacity,
    uint32_t initial_timestamp,
    uint32_t timestamp,
    uint8_t address_space,
    uint64_t address,
    uint64_t value,
    uint32_t instruction,
    uint32_t residual_index,
    uint8_t action,
    uint8_t width,
    uint8_t byte_offset,
    uint8_t expected_mode,
    uint32_t *error
) {
    if (timestamp < initial_timestamp ||
        size_t(timestamp - initial_timestamp) >= timeline_capacity || action == G2_EVENT_NONE) {
        fail_parallel(error, 60);
        return;
    }
    size_t index = timestamp - initial_timestamp;
    G2TimelineEvent event{
        g2_address_key(address_space, address, error),
        value,
        instruction,
        residual_index,
        0,
        action,
        width,
        byte_offset,
        expected_mode,
    };
    if (event.address_key == G2_INVALID_ADDRESS) return;
    // The execution timestamp discipline permits at most one memory-bus event
    // at each timestamp. A duplicate indicates a malformed chronology rather
    // than a benign race.
    if (atomicCAS(
            reinterpret_cast<unsigned long long *>(&timeline[index].address_key),
            G2_INVALID_ADDRESS,
            event.address_key
        ) != G2_INVALID_ADDRESS) {
        fail_parallel(error, 61);
        return;
    }
    timeline[index].value = event.value;
    timeline[index].instruction = event.instruction;
    timeline[index].residual_index = event.residual_index;
    timeline[index].previous_timestamp = 0;
    timeline[index].action = event.action;
    timeline[index].width = event.width;
    timeline[index].byte_offset = event.byte_offset;
    timeline[index].expected_mode = event.expected_mode;
}

__device__ __forceinline__ bool load_store_kind(uint32_t kind) {
    return kind == 8 || kind == 9 || (kind >= 20 && kind <= 28);
}

__device__ __forceinline__ void lane_consumption(
    uint32_t kind, RvrOperandEntry const &entry, bool &v0, bool &v1
) {
    v0 = false;
    v1 = false;
    if (kind <= 7 || (kind >= 15 && kind <= 19)) {
        v0 = true;
        v1 = (entry.flags & RVR_OPERAND_FLAG_RS2_IMM) == 0;
    } else if (kind == 10 || kind == 11) {
        v0 = true;
        v1 = true;
    } else if (kind == 13 || kind == 29) {
        v0 = true;
    } else if (load_store_kind(kind)) {
        v0 = true;
        v1 = true;
    }
}

__device__ __forceinline__ bool zero_arity_kind(uint32_t kind) {
    return kind == 12 || kind == 14;
}

__device__ __forceinline__ bool standard_v0_kind(uint32_t kind) {
    return kind < 30 && !zero_arity_kind(kind);
}

__device__ __forceinline__ bool standard_v1_kind(uint32_t kind) {
    return standard_v0_kind(kind) && kind != 13 && kind != 29;
}

__device__ __forceinline__ uint16_t lane_v0(uint32_t kind) {
    return uint16_t(0x0100u + 2u * kind);
}

__device__ __forceinline__ uint16_t lane_v1(uint32_t kind) {
    return uint16_t(lane_v0(kind) + 1u);
}

__device__ bool lane_spec(
    uint16_t kind, uint8_t &width, uint32_t &flags, uint32_t &group
) {
    if (kind == G2_RUN_BLOCK_ID) {
        width = 4;
        flags = G2_REQUIRED;
        group = 0;
        return true;
    }
    if (kind == G2_RESIDUAL_CTRL || kind == G2_RESIDUAL_TAG ||
        kind == G2_RESIDUAL_VALUE || kind == G2_OPAQUE_EVENT_COUNT) {
        width = kind == G2_RESIDUAL_TAG ? 1 : (kind == G2_OPAQUE_EVENT_COUNT ? 4 : 8);
        flags = G2_REQUIRED_ATOMIC;
        group = G2_RESIDUAL_GROUP;
        return true;
    }
    for (uint32_t delta_kind = 0; delta_kind < 30; ++delta_kind) {
        bool is_v0 = kind == lane_v0(delta_kind) && standard_v0_kind(delta_kind);
        bool is_v1 = kind == lane_v1(delta_kind) && standard_v1_kind(delta_kind);
        if (is_v0 || is_v1) {
            width = is_v0 && (load_store_kind(delta_kind) || delta_kind == 13) ? 4 : 8;
            bool atomic = load_store_kind(delta_kind);
            flags = atomic ? G2_REQUIRED_ATOMIC : G2_REQUIRED;
            group = atomic ? G2_LOAD_STORE_GROUP : 0;
            return true;
        }
    }
    return false;
}

__device__ G2LaneDescV1 const *find_lane(
    G2LaneDescV1 const *descs, size_t count, uint16_t kind
) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (descs[mid].kind < kind)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < count && descs[lo].kind == kind ? &descs[lo] : nullptr;
}

__device__ G2BlockEntryV1 const *find_block(
    G2BlockEntryV1 const *blocks, size_t count, uint32_t slot
) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (blocks[mid].program_slot < slot)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < count && blocks[lo].program_slot == slot ? &blocks[lo] : nullptr;
}

__device__ G2ExpectedOpaqueV1 const *find_expected_opaque(
    G2ExpectedOpaqueV1 const *expected, size_t count, uint32_t lane_kind
) {
    for (size_t i = 0; i < count; ++i) {
        if (expected[i].lane_kind == lane_kind) return &expected[i];
    }
    return nullptr;
}

__device__ __forceinline__ uint64_t lane_u64(
    uint8_t const *wire, G2LaneDescV1 const &lane, size_t index
) {
    return reinterpret_cast<uint64_t const *>(wire + lane.offset)[index];
}

__device__ __forceinline__ uint32_t lane_u32(
    uint8_t const *wire, G2LaneDescV1 const &lane, size_t index
) {
    return reinterpret_cast<uint32_t const *>(wire + lane.offset)[index];
}

__device__ __forceinline__ bool g2_prepared_lane_value(
    uint8_t const *wire,
    G2PreparedInstruction const &prepared,
    bool second,
    uint64_t &value,
    uint32_t *error
) {
    if (prepared.kind < 0) {
        fail_parallel(error, second ? 55 : 54);
        return false;
    }
    uint32_t index = second ? prepared.v1_index : prepared.v0_index;
    G2SegmentHeaderV1 const &header = *reinterpret_cast<G2SegmentHeaderV1 const *>(wire);
    G2LaneDescV1 const *descs = reinterpret_cast<G2LaneDescV1 const *>(wire + 64);
    G2LaneDescV1 const *lane =
        find_lane(descs, header.lane_count, second ? lane_v1(prepared.kind)
                                                  : lane_v0(prepared.kind));
    if (lane == nullptr || index == UINT32_MAX || index >= lane->count ||
        (lane->elem_width != 4 && lane->elem_width != 8)) {
        fail_parallel(error, second ? 55 : 54);
        return false;
    }
    value = lane->elem_width == 4 ? lane_u32(wire, *lane, index)
                                  : lane_u64(wire, *lane, index);
    return true;
}

__device__ __forceinline__ uint32_t effective_address(
    uint32_t pointer, RvrOperandEntry const &entry
) {
    int32_t offset = int32_t(entry.c & 0xffffu);
    if (entry.flags & RVR_OPERAND_FLAG_LS_IMM_SIGN) offset = int32_t(int16_t(offset));
    return pointer + uint32_t(offset);
}

__device__ __forceinline__ uint8_t g2_memory_width(
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
    default: fail_parallel(error, 62); return 0;
    }
}

__device__ __forceinline__ bool g2_crosses(
    uint32_t pointer, RvrOperandEntry const &entry, uint32_t *error
) {
    uint8_t width = g2_memory_width(entry, error);
    return (effective_address(pointer, entry) & 7u) + width > 8u;
}

__device__ bool load_value(
    RvrOperandEntry const &entry, uint32_t pointer, uint64_t block, uint64_t &value
) {
    uint32_t shift = (effective_address(pointer, entry) & 7u) * 8u;
    uint64_t shifted = block >> shift;
    switch (entry.local_opcode) {
    case 0: value = shifted; return true;
    case 1: value = uint8_t(shifted); return true;
    case 2: value = uint16_t(shifted); return true;
    case 3: value = uint32_t(shifted); return true;
    case 8: value = uint64_t(int64_t(int8_t(shifted))); return true;
    case 9: value = uint64_t(int64_t(int16_t(shifted))); return true;
    case 10: value = uint64_t(int64_t(int32_t(shifted))); return true;
    default: return false;
    }
}

__device__ bool load_value_two_blocks(
    RvrOperandEntry const &entry,
    uint32_t pointer,
    uint64_t block0,
    uint64_t block1,
    uint64_t &value
) {
    uint32_t shift = (effective_address(pointer, entry) & 7u) * 8u;
    uint64_t joined = shift == 0 ? block0 : (block0 >> shift) | (block1 << (64u - shift));
    switch (entry.local_opcode) {
    case 0: value = joined; return true;
    case 1: value = uint8_t(joined); return true;
    case 2: value = uint16_t(joined); return true;
    case 3: value = uint32_t(joined); return true;
    case 8: value = uint64_t(int64_t(int8_t(joined))); return true;
    case 9: value = uint64_t(int64_t(int16_t(joined))); return true;
    case 10: value = uint64_t(int64_t(int32_t(joined))); return true;
    default: return false;
    }
}

__device__ __forceinline__ uint64_t standard_immediate(RvrOperandEntry const &entry) {
    return (entry.flags & RVR_OPERAND_FLAG_RS2_IMM_SIGN)
               ? uint64_t(int64_t(int32_t(entry.c << 8) >> 8))
               : uint64_t(entry.c);
}

__device__ __forceinline__ uint64_t sign_extend_word(uint32_t value) {
    return uint64_t(int64_t(int32_t(value)));
}

__device__ bool divrem_result(
    uint8_t local_opcode, uint64_t lhs_u, uint64_t rhs_u, uint64_t &result
) {
    switch (local_opcode) {
    case 0: {
        int64_t lhs = int64_t(lhs_u), rhs = int64_t(rhs_u);
        if (rhs == 0)
            result = UINT64_MAX;
        else if (lhs == INT64_MIN && rhs == -1)
            result = uint64_t(lhs);
        else
            result = uint64_t(lhs / rhs);
        return true;
    }
    case 1: result = rhs_u == 0 ? UINT64_MAX : lhs_u / rhs_u; return true;
    case 2: {
        int64_t lhs = int64_t(lhs_u), rhs = int64_t(rhs_u);
        if (rhs == 0)
            result = lhs_u;
        else if (lhs == INT64_MIN && rhs == -1)
            result = 0;
        else
            result = uint64_t(lhs % rhs);
        return true;
    }
    case 3: result = rhs_u == 0 ? lhs_u : lhs_u % rhs_u; return true;
    default: return false;
    }
}

__device__ bool divrem_w_result(
    uint8_t local_opcode, uint64_t lhs_u, uint64_t rhs_u, uint64_t &result
) {
    uint32_t lhs = uint32_t(lhs_u), rhs = uint32_t(rhs_u), word;
    switch (local_opcode) {
    case 0: {
        int32_t lhs_s = int32_t(lhs), rhs_s = int32_t(rhs);
        if (rhs_s == 0)
            word = UINT32_MAX;
        else if (lhs_s == INT32_MIN && rhs_s == -1)
            word = uint32_t(lhs_s);
        else
            word = uint32_t(lhs_s / rhs_s);
        break;
    }
    case 1: word = rhs == 0 ? UINT32_MAX : lhs / rhs; break;
    case 2: {
        int32_t lhs_s = int32_t(lhs), rhs_s = int32_t(rhs);
        if (rhs_s == 0)
            word = lhs;
        else if (lhs_s == INT32_MIN && rhs_s == -1)
            word = 0;
        else
            word = uint32_t(lhs_s % rhs_s);
        break;
    }
    case 3: word = rhs == 0 ? lhs : lhs % rhs; break;
    default: return false;
    }
    result = sign_extend_word(word);
    return true;
}

__device__ bool standard_post_write(
    uint32_t kind,
    RvrOperandEntry const &entry,
    uint64_t v0,
    uint64_t v1,
    uint64_t &result
) {
    switch (kind) {
    case 0:
        if (entry.local_opcode == 0)
            result = v0 + v1;
        else if (entry.local_opcode == 1)
            result = v0 - v1;
        else
            return false;
        return true;
    case 1:
        if (entry.local_opcode == 2)
            result = v0 ^ v1;
        else if (entry.local_opcode == 3)
            result = v0 | v1;
        else if (entry.local_opcode == 4)
            result = v0 & v1;
        else
            return false;
        return true;
    case 2:
        if (entry.local_opcode == 0)
            result = int64_t(v0) < int64_t(v1);
        else if (entry.local_opcode == 1)
            result = v0 < v1;
        else
            return false;
        return true;
    case 3:
        if (entry.local_opcode == 0)
            result = v0 << uint32_t(v1 & 63u);
        else if (entry.local_opcode == 1)
            result = v0 >> uint32_t(v1 & 63u);
        else
            return false;
        return true;
    case 4:
        if (entry.local_opcode != 2) return false;
        result = uint64_t(int64_t(v0) >> uint32_t(v1 & 63u));
        return true;
    case 5:
        if (entry.local_opcode == 0)
            result = sign_extend_word(uint32_t(v0) + uint32_t(v1));
        else if (entry.local_opcode == 1)
            result = sign_extend_word(uint32_t(v0) - uint32_t(v1));
        else
            return false;
        return true;
    case 6:
        if (entry.local_opcode == 0)
            result = sign_extend_word(uint32_t(v0) << uint32_t(v1 & 31u));
        else if (entry.local_opcode == 1)
            result = sign_extend_word(uint32_t(v0) >> uint32_t(v1 & 31u));
        else
            return false;
        return true;
    case 7:
        if (entry.local_opcode != 0) return false;
        result = sign_extend_word(uint32_t(int32_t(uint32_t(v0)) >> uint32_t(v1 & 31u)));
        return true;
    case 15: result = v0 * v1; return true;
    case 16:
        if (entry.local_opcode == 0)
            result = uint64_t(__mul64hi((long long)v0, (long long)v1));
        else if (entry.local_opcode == 1)
            result = __umul64hi(v0, v1) - (int64_t(v0) < 0 ? v1 : 0);
        else if (entry.local_opcode == 2)
            result = __umul64hi(v0, v1);
        else
            return false;
        return true;
    case 17: result = sign_extend_word(uint32_t(v0) * uint32_t(v1)); return true;
    case 18: return divrem_result(entry.local_opcode, v0, v1, result);
    case 19: return divrem_w_result(entry.local_opcode, v0, v1, result);
    default: return false;
    }
}

__device__ bool residual_event(
    uint8_t const *wire,
    G2LaneDescV1 const &ctrl,
    G2LaneDescV1 const &tag,
    G2LaneDescV1 const &value,
    size_t index,
    uint32_t expected_timestamp,
    uint32_t expected_address,
    uint8_t expected_tag,
    uint64_t expected_value,
    bool check_value,
    uint64_t &actual_value
) {
    uint64_t control = lane_u64(wire, ctrl, index);
    uint8_t actual_tag = (wire + tag.offset)[index];
    actual_value = lane_u64(wire, value, index);
    return uint32_t(control) == expected_timestamp && uint32_t(control >> 32) == expected_address &&
           actual_tag == expected_tag && (!check_value || actual_value == expected_value);
}

__device__ bool append_opaque_residual(
    uint8_t const *wire,
    G2LaneDescV1 const &ctrl,
    G2LaneDescV1 const &tag,
    G2LaneDescV1 const &value,
    size_t index,
    DeltaMemoryLogEntry *output,
    size_t output_capacity,
    size_t &output_count,
    uint64_t *registers
) {
    if (output_count >= output_capacity) return false;
    uint64_t control = lane_u64(wire, ctrl, index);
    uint8_t encoded = (wire + tag.offset)[index];
    uint64_t event_value = lane_u64(wire, value, index);
    uint8_t kind = encoded & 3u;
    uint8_t address_space_code = (encoded >> 2) & 3u;
    uint8_t width_code = (encoded >> 4) & 7u;
    static constexpr uint8_t widths[5] = {0, 1, 2, 4, 8};
    if ((encoded & 0x80u) || kind == 3 || address_space_code == 3 || width_code > 4 ||
        (kind != 2 && width_code == 0)) {
        return false;
    }
    uint32_t address = uint32_t(control >> 32);
    uint8_t address_space = address_space_code + 1u;
    if (address_space == 1u && registers != nullptr) {
        if (width_code != 4 || (address & 7u) != 0 || address >= 32u * 8u) return false;
        uint32_t reg = address / 8u;
        if (kind == 0 && registers[reg] != event_value) return false;
        if (kind == 1 && reg != 0) registers[reg] = event_value;
    }
    output[output_count++] = {
        uint32_t(control),
        address,
        event_value,
        kind,
        address_space,
        widths[width_code],
        1u,
        0u,
    };
    return true;
}

__global__ void g2_run_lengths(
    uint8_t const *wire,
    size_t wire_storage_bytes,
    size_t run_count,
    G2BlockEntryV1 const *blocks,
    size_t block_count,
    uint32_t *lengths,
    uint32_t *error
) {
    if (*error != 0) return;
    size_t run_index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (run_index >= run_count) return;
    G2SegmentHeaderV1 const &header = *reinterpret_cast<G2SegmentHeaderV1 const *>(wire);
    if (wire_storage_bytes < 128 || header.header_bytes != 64 + 32 * header.lane_count ||
        header.header_bytes > wire_storage_bytes || header.run_count != run_count) {
        fail_parallel(error, 49);
        return;
    }
    G2LaneDescV1 const *descs = reinterpret_cast<G2LaneDescV1 const *>(wire + 64);
    G2LaneDescV1 const *run = find_lane(descs, header.lane_count, G2_RUN_BLOCK_ID);
    if (run == nullptr || run->count != run_count || run->elem_width != 4 ||
        run->payload_bytes != run_count * sizeof(uint32_t) ||
        run->offset > wire_storage_bytes ||
        run->payload_bytes > wire_storage_bytes - run->offset) {
        fail_parallel(error, 50);
        return;
    }
    uint32_t run_slot = lane_u32(wire, *run, run_index);
    G2BlockEntryV1 const *block = find_block(blocks, block_count, run_slot);
    if (block == nullptr || block->instruction_count == 0) {
        fail_parallel(error, 10);
        return;
    }
    lengths[run_index] = block->instruction_count;
}

__global__ void g2_expand_runs(
    uint8_t const *wire,
    size_t run_count,
    G2BlockEntryV1 const *blocks,
    size_t block_count,
    uint32_t const *offsets,
    size_t instruction_count,
    RvrOperandEntry const *operands,
    size_t operand_count,
    uint32_t *program_slots,
    uint32_t *program_frequencies,
    size_t frequency_count,
    uint32_t *error
) {
    if (*error != 0) return;
    if (offsets[run_count] != instruction_count) {
        fail_parallel(error, 53);
        return;
    }
    size_t run_index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (run_index >= run_count) return;
    G2SegmentHeaderV1 const &header = *reinterpret_cast<G2SegmentHeaderV1 const *>(wire);
    G2LaneDescV1 const *descs = reinterpret_cast<G2LaneDescV1 const *>(wire + 64);
    G2LaneDescV1 const *run = find_lane(descs, header.lane_count, G2_RUN_BLOCK_ID);
    if (run == nullptr) {
        fail_parallel(error, 50);
        return;
    }
    uint32_t run_slot = lane_u32(wire, *run, run_index);
    G2BlockEntryV1 const *block = find_block(blocks, block_count, run_slot);
    uint32_t begin = offsets[run_index], end = offsets[run_index + 1];
    if (block == nullptr || end < begin || end - begin != block->instruction_count ||
        end > instruction_count) {
        fail_parallel(error, 51);
        return;
    }
    for (uint32_t local = 0; local < block->instruction_count; ++local) {
        uint32_t slot = run_slot + local;
        if (slot < run_slot || slot >= operand_count) {
            fail_parallel(error, 11);
            return;
        }
        RvrOperandEntry const entry = operands[slot];
        if (entry.filtered_index >= frequency_count) {
            fail_parallel(error, 12);
            return;
        }
        program_slots[begin + local] = slot;
        atomicAdd(program_frequencies + entry.filtered_index, 1u);
    }
}

__global__ void g2_build_kind_map(
    G2ExpectedKindV1 const *expected,
    size_t expected_count,
    int8_t *kind_by_air,
    uint32_t *error
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (size_t i = 0; i < 256; ++i) kind_by_air[i] = -1;
    for (size_t i = 0; i < expected_count; ++i) {
        G2ExpectedKindV1 entry = expected[i];
        if (entry.air_idx >= 256 || entry.kind >= 31 || kind_by_air[entry.air_idx] != -1) {
            fail_parallel(error, 48);
            return;
        }
        kind_by_air[entry.air_idx] = int8_t(entry.kind);
    }
}

__global__ void g2_count_lane_blocks(
    uint32_t const *program_slots,
    size_t instruction_count,
    RvrOperandEntry const *operands,
    size_t operand_count,
    int8_t const *kind_by_air,
    uint32_t *row_counts,
    uint32_t *v0_counts,
    uint32_t *v1_counts,
    size_t count_stride,
    uint32_t *error
) {
    if (*error != 0) return;
    __shared__ uint32_t counts[93];
    for (uint32_t i = threadIdx.x; i < 93; i += blockDim.x) counts[i] = 0;
    __syncthreads();

    size_t instruction = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (instruction < instruction_count) {
        uint32_t slot = program_slots[instruction];
        if (slot >= operand_count) {
            fail_parallel(error, 11);
        } else {
            RvrOperandEntry entry = operands[slot];
            if (entry.air_idx >= 256) {
                fail_parallel(error, 48);
            } else {
                int32_t kind = kind_by_air[entry.air_idx];
                if (kind >= 0) {
                    bool v0, v1;
                    lane_consumption(uint32_t(kind), entry, v0, v1);
                    if (kind < 30) atomicAdd(&counts[kind], 1u);
                    if (v0) atomicAdd(&counts[31 + kind], 1u);
                    if (v1) atomicAdd(&counts[62 + kind], 1u);
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t kind = threadIdx.x; kind < 31; kind += blockDim.x) {
        row_counts[size_t(kind) * count_stride + blockIdx.x] = counts[kind];
        v0_counts[size_t(kind) * count_stride + blockIdx.x] = counts[31 + kind];
        v1_counts[size_t(kind) * count_stride + blockIdx.x] = counts[62 + kind];
    }
}

__global__ void g2_prepare_instructions(
    uint8_t const *wire,
    uint32_t const *program_slots,
    size_t instruction_count,
    RvrOperandEntry const *operands,
    size_t operand_count,
    int8_t const *kind_by_air,
    uint32_t const *row_offsets,
    uint32_t const *v0_offsets,
    uint32_t const *v1_offsets,
    size_t count_stride,
    G2PreparedInstruction *prepared,
    uint32_t *error
) {
    if (*error != 0) return;
    __shared__ int8_t kinds[256];
    __shared__ uint8_t consumption[256];
    size_t instruction = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t slot = UINT32_MAX;
    RvrOperandEntry entry{};
    int32_t kind = -1;
    bool v0 = false, v1 = false;
    if (instruction < instruction_count) {
        slot = program_slots[instruction];
        if (slot >= operand_count) {
            fail_parallel(error, 11);
        } else {
            entry = operands[slot];
            if (entry.air_idx >= 256) {
                fail_parallel(error, 48);
            } else {
                kind = kind_by_air[entry.air_idx];
                if (kind >= 0) lane_consumption(uint32_t(kind), entry, v0, v1);
            }
        }
    }
    kinds[threadIdx.x] = int8_t(kind);
    consumption[threadIdx.x] = uint8_t(v0) | (uint8_t(v1) << 1);
    __syncthreads();
    if (instruction >= instruction_count || slot >= operand_count) return;

    uint32_t local_row = 0, local_v0 = 0, local_v1 = 0;
    for (uint32_t local = 0; local < threadIdx.x; ++local) {
        if (kinds[local] != kind) continue;
        local_row += kind >= 0 && kind < 30;
        local_v0 += consumption[local] & 1u;
        local_v1 += consumption[local] >> 1;
    }
    G2PreparedInstruction output{};
    output.slot = slot;
    output.kind = kind;
    output.row_index = UINT32_MAX;
    output.v0_index = UINT32_MAX;
    output.v1_index = UINT32_MAX;
    if (kind >= 0) {
        if (kind < 30) {
            output.row_index =
                row_offsets[size_t(kind) * count_stride + blockIdx.x] + local_row;
        }
        G2SegmentHeaderV1 const &header = *reinterpret_cast<G2SegmentHeaderV1 const *>(wire);
        G2LaneDescV1 const *descs = reinterpret_cast<G2LaneDescV1 const *>(wire + 64);
        if (v0) {
            uint32_t index =
                v0_offsets[size_t(kind) * count_stride + blockIdx.x] + local_v0;
            G2LaneDescV1 const *lane = find_lane(descs, header.lane_count, lane_v0(kind));
            if (lane == nullptr || index >= lane->count) {
                fail_parallel(error, 54);
            } else {
                output.v0_index = index;
            }
        }
        if (v1) {
            uint32_t index =
                v1_offsets[size_t(kind) * count_stride + blockIdx.x] + local_v1;
            G2LaneDescV1 const *lane = find_lane(descs, header.lane_count, lane_v1(kind));
            if (lane == nullptr || index >= lane->count) {
                fail_parallel(error, 55);
            } else {
                output.v1_index = index;
            }
        }
    }
    prepared[instruction] = output;
}

__global__ void g2_classify_instructions(
    uint8_t const *wire,
    G2PreparedInstruction const *prepared,
    size_t instruction_count,
    RvrOperandEntry const *operands,
    size_t operand_count,
    int8_t const *kind_by_air,
    uint32_t *timestamp_counts,
    uint32_t *output_counts,
    uint32_t *residual_counts,
    uint32_t *opaque_counts,
    uint32_t *opaque_markers,
    uint8_t *hint_flags,
    uint32_t *instruction_indices,
    uint32_t *error
) {
    if (*error != 0) return;
    size_t instruction = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (instruction >= instruction_count) return;
    G2PreparedInstruction decoded = prepared[instruction];
    if (decoded.slot >= operand_count) {
        fail_parallel(error, 11);
        return;
    }
    RvrOperandEntry entry = operands[decoded.slot];
    instruction_indices[instruction] = uint32_t(instruction);
    if (entry.air_idx == INVALID_AIR) {
        if (decoded.kind != -1) fail_parallel(error, 56);
        return;
    }
    if (entry.air_idx >= 256) {
        fail_parallel(error, 48);
        return;
    }
    int32_t kind_code = kind_by_air[entry.air_idx];
    if (kind_code != decoded.kind) {
        fail_parallel(error, 56);
        return;
    }
    if (kind_code < 0) {
        if (entry.access_pattern == G2_OPAQUE_TIMESTAMP_PATTERN) {
            timestamp_counts[instruction] = 1;
        } else if (entry.access_pattern == G2_OPAQUE_PATTERN) {
            opaque_markers[instruction] = 1;
        } else {
            fail_parallel(error, 13);
        }
        return;
    }
    uint32_t kind = uint32_t(kind_code);
    if (kind == 30 && entry.access_pattern == G2_HINT_PATTERN) {
        if (entry.local_opcode > 1) {
            fail_parallel(error, 42);
            return;
        }
        hint_flags[instruction] = 1;
        return;
    }
    output_counts[instruction] = 1;
    if (kind == 29 && entry.access_pattern == G2_ADDI_PATTERN) {
        timestamp_counts[instruction] = 2;
    } else if ((kind <= 7 || (kind >= 15 && kind <= 19)) &&
               (entry.access_pattern == 0 || entry.access_pattern == 1)) {
        timestamp_counts[instruction] = 3;
    } else if ((kind == 10 || kind == 11) && entry.access_pattern == 4) {
        timestamp_counts[instruction] = 2;
    } else if ((kind == 12 || kind == 14) &&
               (entry.access_pattern == 5 || entry.access_pattern == 6)) {
        timestamp_counts[instruction] = 1;
    } else if (kind == 13 && entry.access_pattern == 7) {
        timestamp_counts[instruction] = 2;
    } else if (load_store_kind(kind) &&
               (entry.access_pattern == G2_LOAD_PATTERN ||
                entry.access_pattern == G2_STORE_PATTERN)) {
        uint64_t pointer;
        if (!g2_prepared_lane_value(wire, decoded, false, pointer, error) ||
            pointer > UINT32_MAX) {
            fail_parallel(error, 27);
            return;
        }
        uint8_t width = g2_memory_width(entry, error);
        bool crossing = g2_crosses(uint32_t(pointer), entry, error);
        timestamp_counts[instruction] = width > 1 ? 4 : 3;
        residual_counts[instruction] = crossing ? 2 : 0;
    } else {
        fail_parallel(error, 22);
    }
}

__global__ void g2_fill_opaque_shapes(
    uint8_t const *wire,
    uint32_t const *opaque_occurrences,
    size_t instruction_count,
    uint32_t *timestamp_counts,
    uint32_t *residual_counts,
    uint32_t *opaque_counts,
    uint32_t *error
) {
    if (*error != 0) return;
    size_t instruction = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (instruction >= instruction_count ||
        opaque_occurrences[instruction] == opaque_occurrences[instruction + 1])
        return;
    G2SegmentHeaderV1 const &header = *reinterpret_cast<G2SegmentHeaderV1 const *>(wire);
    G2LaneDescV1 const *descs = reinterpret_cast<G2LaneDescV1 const *>(wire + 64);
    G2LaneDescV1 const *events = find_lane(descs, header.lane_count, G2_OPAQUE_EVENT_COUNT);
    uint32_t occurrence = opaque_occurrences[instruction];
    if (events == nullptr || occurrence >= events->count) {
        fail_parallel(error, 40);
        return;
    }
    uint32_t count = lane_u32(wire, *events, occurrence);
    if (count == 0) {
        fail_parallel(error, 47);
        return;
    }
    timestamp_counts[instruction] = count;
    residual_counts[instruction] = count;
    opaque_counts[instruction] = count;
}

__global__ void g2_plan_hints(
    uint8_t const *wire,
    G2PreparedInstruction *prepared,
    RvrOperandEntry const *operands,
    size_t operand_count,
    uint32_t const *hint_indices,
    uint32_t const *hint_count,
    uint32_t const *static_timestamp_offsets,
    uint32_t const *static_residual_offsets,
    uint32_t initial_timestamp,
    uint32_t *timestamp_counts,
    uint32_t *output_counts,
    uint32_t *residual_counts,
    uint32_t *hint_row_offsets,
    uint32_t *error
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *error != 0) return;
    G2SegmentHeaderV1 const &header = *reinterpret_cast<G2SegmentHeaderV1 const *>(wire);
    G2LaneDescV1 const *descs = reinterpret_cast<G2LaneDescV1 const *>(wire + 64);
    G2LaneDescV1 const *ctrl = find_lane(descs, header.lane_count, G2_RESIDUAL_CTRL);
    G2LaneDescV1 const *tag = find_lane(descs, header.lane_count, G2_RESIDUAL_TAG);
    G2LaneDescV1 const *value = find_lane(descs, header.lane_count, G2_RESIDUAL_VALUE);
    uint32_t dynamic_timestamp = 0;
    uint32_t dynamic_residual = 0;
    uint32_t hint_row_cursor = 0;
    for (uint32_t hint = 0; hint < *hint_count; ++hint) {
        uint32_t instruction = hint_indices[hint];
        if (prepared[instruction].slot >= operand_count) {
            fail(error, 11);
            return;
        }
        RvrOperandEntry entry = operands[prepared[instruction].slot];
        if (ctrl == nullptr || tag == nullptr || value == nullptr ||
            entry.access_pattern != G2_HINT_PATTERN || (entry.b & 7u) != 0 ||
            entry.b >= 32u * 8u ||
            (entry.local_opcode != 0 && ((entry.a & 7u) != 0 || entry.a >= 32u * 8u))) {
            fail(error, 42);
            return;
        }
        uint32_t timestamp =
            initial_timestamp + static_timestamp_offsets[instruction] + dynamic_timestamp;
        size_t residual =
            size_t(static_residual_offsets[instruction]) + dynamic_residual;
        if (residual >= ctrl->count) {
            fail(error, 43);
            return;
        }
        uint64_t memory_pointer;
        if (!residual_event(
                wire,
                *ctrl,
                *tag,
                *value,
                residual,
                timestamp,
                entry.b,
                0x40u,
                0,
                false,
                memory_pointer
            ) ||
            memory_pointer > UINT32_MAX) {
            fail(error, 44);
            return;
        }
        uint32_t num_words = 1;
        if (entry.local_opcode != 0) {
            uint64_t words;
            if (residual + 1 >= ctrl->count ||
                !residual_event(
                    wire,
                    *ctrl,
                    *tag,
                    *value,
                    residual + 1,
                    timestamp + 1,
                    entry.a,
                    0x40u,
                    0,
                    false,
                    words
                ) ||
                words == 0 || words > 1023) {
                fail(error, 45);
                return;
            }
            num_words = uint32_t(words);
        }
        uint32_t residual_count = num_words + 1u + entry.local_opcode;
        if (residual_count < num_words || residual + residual_count > ctrl->count ||
            dynamic_timestamp > UINT32_MAX - 3u * num_words ||
            dynamic_residual > UINT32_MAX - residual_count) {
            fail(error, 43);
            return;
        }
        timestamp_counts[instruction] = 3u * num_words;
        output_counts[instruction] = num_words;
        residual_counts[instruction] = residual_count;
        hint_row_offsets[instruction] = hint_row_cursor;
        if (hint_row_cursor > UINT32_MAX - num_words) {
            fail(error, 43);
            return;
        }
        prepared[instruction].row_index = hint_row_cursor;
        prepared[instruction].v0_index = num_words;
        prepared[instruction].v1_index = UINT32_MAX;
        hint_row_cursor += num_words;
        dynamic_timestamp += 3u * num_words;
        dynamic_residual += residual_count;
    }
}

__device__ __forceinline__ uint8_t g2_store_width(
    RvrOperandEntry const &entry, uint32_t *error
) {
    switch (entry.local_opcode) {
    case 4: return 8;
    case 5: return 4;
    case 6: return 2;
    case 7: return 1;
    default: fail_parallel(error, 62); return 0;
    }
}

__device__ bool g2_emit_standard_direct(
    DeltaRecord const &record,
    uint64_t expected_block,
    uint8_t expected_mode,
    bool crossing,
    uint32_t crossing_residual_start,
    uint64_t crossing_value0,
    uint64_t crossing_value1,
    RvrOperandEntry const &entry,
    uint32_t kind,
    uint32_t instruction,
    G2TimelineEvent *timeline,
    size_t timeline_capacity,
    uint32_t initial_timestamp,
    uint32_t *error
) {
    auto reg_event = [&](uint32_t timestamp, uint32_t pointer, uint64_t value, uint8_t action) {
        g2_emit_timeline(
            timeline,
            timeline_capacity,
            initial_timestamp,
            timestamp,
            1,
            uint64_t(pointer) & ~uint64_t(7),
            value,
            instruction,
            G2_NO_RECORD,
            action,
            8,
            0,
            EXPECT_NONE,
            error
        );
    };
    auto memory_event = [&](uint32_t timestamp,
                            uint64_t value,
                            uint8_t action,
                            uint8_t width,
                            uint8_t byte_offset,
                            uint8_t mode) {
        uint8_t address_space =
            (entry.flags & RVR_OPERAND_FLAG_LS_PUBLIC_VALUES) ? 3 : 2;
        uint32_t effective = effective_address(uint32_t(record.v1), entry);
        g2_emit_timeline(
            timeline,
            timeline_capacity,
            initial_timestamp,
            timestamp,
            address_space,
            uint64_t(effective) & ~uint64_t(7),
            value,
            instruction,
            G2_NO_RECORD,
            action,
            width,
            byte_offset,
            mode,
            error
        );
    };

    if (entry.access_pattern == G2_ADDI_PATTERN) {
        reg_event(record.from_timestamp, entry.b, record.v1, G2_EVENT_READ_EXPECT);
        reg_event(record.from_timestamp + 1, entry.a, record.v1 + record.v2,
                  G2_EVENT_WRITE_SET);
        return true;
    }
    if (entry.access_pattern == 0 || entry.access_pattern == 1) {
        reg_event(record.from_timestamp, entry.b, record.v1, G2_EVENT_READ_EXPECT);
        if ((entry.flags & RVR_OPERAND_FLAG_RS2_IMM) == 0) {
            reg_event(record.from_timestamp + 1, entry.c, record.v2, G2_EVENT_READ_EXPECT);
        }
        uint64_t result;
        if (!standard_post_write(kind, entry, record.v1, record.v2, result)) {
            fail_parallel(error, 64);
            return false;
        }
        reg_event(record.from_timestamp + 2, entry.a, result, G2_EVENT_WRITE_SET);
        return true;
    }
    if (entry.access_pattern == 4) {
        reg_event(record.from_timestamp, entry.a, record.v1, G2_EVENT_READ_EXPECT);
        reg_event(record.from_timestamp + 1, entry.b, record.v2, G2_EVENT_READ_EXPECT);
        return true;
    }
    if (entry.access_pattern == 5 || entry.access_pattern == 6) {
        if (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED) {
            uint64_t result = kind == 14
                                  ? uint64_t(record.from_pc) +
                                        uint64_t(int64_t(int32_t(entry.c << 8)))
                                  : ((entry.flags & RVR_OPERAND_FLAG_IS_JAL)
                                         ? uint64_t(record.from_pc + 4u)
                                         : uint64_t(int64_t(int32_t(entry.c << 12))));
            reg_event(record.from_timestamp, entry.a, result, G2_EVENT_WRITE_SET);
        }
        return true;
    }
    if (entry.access_pattern == 7) {
        reg_event(record.from_timestamp, entry.b, record.v1, G2_EVENT_READ_EXPECT);
        if (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED) {
            reg_event(record.from_timestamp + 1, entry.a, uint64_t(record.from_pc + 4u),
                      G2_EVENT_WRITE_SET);
        }
        return true;
    }
    if (entry.access_pattern == G2_LOAD_PATTERN ||
        entry.access_pattern == G2_STORE_PATTERN) {
        uint8_t width = g2_memory_width(entry, error);
        uint32_t effective = effective_address(uint32_t(record.v1), entry);
        uint8_t address_space =
            (entry.flags & RVR_OPERAND_FLAG_LS_PUBLIC_VALUES) ? 3 : 2;
        if (width > 1) {
            if (entry.access_pattern == G2_LOAD_PATTERN) {
                uint64_t block0 = crossing ? crossing_value0 : expected_block;
                uint64_t block1 = crossing ? crossing_value1 : 0;
                reg_event(record.from_timestamp, entry.b, record.v1, G2_EVENT_READ_EXPECT);
                if (crossing) {
                    g2_emit_timeline(
                        timeline, timeline_capacity, initial_timestamp,
                        record.from_timestamp + 1, address_space,
                        uint64_t(effective) & ~uint64_t(7), block0, instruction,
                        crossing_residual_start, G2_EVENT_READ_EXPECT, 8, 0,
                        EXPECT_NONE, error
                    );
                    g2_emit_timeline(
                        timeline, timeline_capacity, initial_timestamp,
                        record.from_timestamp + 2, address_space,
                        (uint64_t(effective) & ~uint64_t(7)) + 8, block1,
                        instruction, crossing_residual_start + 1u,
                        G2_EVENT_READ_EXPECT, 8, 0, EXPECT_NONE, error
                    );
                } else {
                    memory_event(record.from_timestamp + 1, expected_block,
                                 G2_EVENT_READ_EXPECT, 8, 0, expected_mode);
                }
                if (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED) {
                    uint64_t result;
                    if (!load_value_two_blocks(
                            entry, uint32_t(record.v1), block0, block1, result
                        )) {
                        fail_parallel(error, 65);
                        return false;
                    }
                    reg_event(record.from_timestamp + 3, entry.a, result, G2_EVENT_WRITE_SET);
                }
            } else {
                reg_event(record.from_timestamp, entry.b, record.v1, G2_EVENT_READ_EXPECT);
                reg_event(record.from_timestamp + 1, entry.a, record.v2,
                          G2_EVENT_READ_PATCH);
                if (crossing) {
                    g2_emit_timeline(
                        timeline, timeline_capacity, initial_timestamp,
                        record.from_timestamp + 2, address_space,
                        uint64_t(effective) & ~uint64_t(7), crossing_value0,
                        instruction, crossing_residual_start,
                        G2_EVENT_WRITE_SET_EXPECT, 8, 0, EXPECT_BEFORE_STORE, error
                    );
                    g2_emit_timeline(
                        timeline, timeline_capacity, initial_timestamp,
                        record.from_timestamp + 3, address_space,
                        (uint64_t(effective) & ~uint64_t(7)) + 8,
                        crossing_value1, instruction, crossing_residual_start + 1u,
                        G2_EVENT_WRITE_SET, 8, 0, EXPECT_NONE, error
                    );
                } else {
                    memory_event(record.from_timestamp + 2, expected_block,
                                 G2_EVENT_STORE_PATCH, width,
                                 uint8_t(effective & 7u), expected_mode);
                }
            }
            return true;
        }
        reg_event(record.from_timestamp, entry.b, record.v1, G2_EVENT_READ_EXPECT);
        if (entry.access_pattern == G2_LOAD_PATTERN) {
            memory_event(record.from_timestamp + 1, expected_block, G2_EVENT_READ_EXPECT,
                         8, 0, expected_mode);
            if (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED) {
                uint64_t result;
                if (!load_value(entry, uint32_t(record.v1), record.v2, result)) {
                    fail_parallel(error, 65);
                    return false;
                }
                reg_event(record.from_timestamp + 2, entry.a, result, G2_EVENT_WRITE_SET);
            }
        } else {
            uint8_t source_action = expected_mode == EXPECT_AFTER_STORE
                                        ? G2_EVENT_READ_EXPECT
                                        : G2_EVENT_READ_PATCH;
            reg_event(record.from_timestamp + 1, entry.a, record.v2, source_action);
            uint8_t width = g2_store_width(entry, error);
            uint8_t byte_offset = uint8_t(effective_address(uint32_t(record.v1), entry) & 7u);
            memory_event(record.from_timestamp + 2, expected_block, G2_EVENT_STORE_PATCH,
                         width, byte_offset, expected_mode);
        }
        return true;
    }
    fail_parallel(error, 63);
    return false;
}

__global__ void g2_emit_parallel(
    uint8_t const *wire,
    G2PreparedInstruction *prepared,
    size_t instruction_count,
    RvrOperandEntry const *operands,
    size_t operand_count,
    int8_t const *kind_by_air,
    uint32_t const *timestamp_offsets,
    uint32_t const *output_offsets,
    uint32_t const *residual_offsets,
    uint32_t const *opaque_offsets,
    uint32_t const *hint_row_offsets,
    uint32_t initial_timestamp,
    uint32_t pc_base,
    size_t delta_count,
    DeltaMemoryLogEntry *opaque_output,
    size_t opaque_capacity,
    DeltaAirOutputDesc const *outputs,
    size_t num_airs,
    uint32_t *row_instructions,
    G2TimelineEvent *timeline,
    size_t timeline_capacity,
    uint32_t *kind_counts,
    uint32_t *error
) {
    size_t instruction = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (instruction >= instruction_count || *error != 0) return;
    G2PreparedInstruction decoded = prepared[instruction];
    if (decoded.slot >= operand_count) {
        fail_parallel(error, 11);
        return;
    }
    RvrOperandEntry entry = operands[decoded.slot];
    if (entry.air_idx == INVALID_AIR) return;
    if (entry.air_idx >= 256 || decoded.kind != kind_by_air[entry.air_idx]) {
        fail_parallel(error, 56);
        return;
    }
    uint32_t timestamp = initial_timestamp + timestamp_offsets[instruction];
    size_t output_cursor = output_offsets[instruction];
    size_t output_end = output_offsets[instruction + 1];
    size_t residual_cursor = residual_offsets[instruction];
    size_t residual_end = residual_offsets[instruction + 1];
    size_t opaque_cursor = opaque_offsets[instruction];
    size_t opaque_end = opaque_offsets[instruction + 1];
    G2SegmentHeaderV1 const &header = *reinterpret_cast<G2SegmentHeaderV1 const *>(wire);
    G2LaneDescV1 const *descs = reinterpret_cast<G2LaneDescV1 const *>(wire + 64);
    // The scans use u32 cursors and dynamic HintStore/opaque counts.  Guard
    // every assigned interval before dereferencing residual lanes or writing
    // an output; the aggregate validator below then enforces exact exhaustion.
    if (timestamp_offsets[instruction + 1] < timestamp_offsets[instruction] ||
        initial_timestamp > UINT32_MAX - timestamp_offsets[instruction + 1] ||
        output_end < output_cursor || output_end > delta_count ||
        residual_end < residual_cursor || residual_end > header.residual_event_count ||
        opaque_end < opaque_cursor || opaque_end > opaque_capacity) {
        fail_parallel(error, 57);
        return;
    }

    if (decoded.kind < 0) {
        if (entry.access_pattern == G2_OPAQUE_TIMESTAMP_PATTERN) return;
        G2LaneDescV1 const *ctrl = find_lane(descs, header.lane_count, G2_RESIDUAL_CTRL);
        G2LaneDescV1 const *tag = find_lane(descs, header.lane_count, G2_RESIDUAL_TAG);
        G2LaneDescV1 const *value = find_lane(descs, header.lane_count, G2_RESIDUAL_VALUE);
        if (entry.access_pattern != G2_OPAQUE_PATTERN || ctrl == nullptr || tag == nullptr ||
            value == nullptr || residual_end - residual_cursor != opaque_end - opaque_cursor) {
            fail_parallel(error, 40);
            return;
        }
        for (size_t event = 0; residual_cursor + event < residual_end; ++event) {
            size_t output_count = opaque_cursor + event;
            if (uint32_t(lane_u64(wire, *ctrl, residual_cursor + event)) != timestamp + event ||
                !append_opaque_residual(
                    wire,
                    *ctrl,
                    *tag,
                    *value,
                    residual_cursor + event,
                    opaque_output,
                    opaque_capacity,
                    output_count,
                    nullptr
                ) ||
                output_count != opaque_cursor + event + 1) {
                fail_parallel(error, 41);
                return;
            }
            DeltaMemoryLogEntry emitted = opaque_output[opaque_cursor + event];
            uint8_t action = emitted.kind == 1
                                 ? G2_EVENT_WRITE_SET
                                 : (emitted.kind == 0 ? G2_EVENT_READ_EXPECT
                                                      : G2_EVENT_READ_ANY);
            g2_emit_timeline(
                timeline,
                timeline_capacity,
                initial_timestamp,
                emitted.timestamp,
                emitted.addr_space,
                uint64_t(emitted.address) & ~uint64_t(7),
                emitted.value,
                G2_NO_RECORD,
                uint32_t(opaque_cursor + event),
                action,
                emitted.width,
                uint8_t(emitted.address & 7u),
                EXPECT_NONE,
                error
            );
        }
        return;
    }

    uint32_t kind = uint32_t(decoded.kind);
    bool consumes_v0, consumes_v1;
    lane_consumption(kind, entry, consumes_v0, consumes_v1);
    if (kind != 30 && ((consumes_v0 && decoded.v0_index == UINT32_MAX) ||
                       (!consumes_v0 && decoded.v0_index != UINT32_MAX) ||
                       (consumes_v1 && decoded.v1_index == UINT32_MAX) ||
                       (!consumes_v1 && decoded.v1_index != UINT32_MAX))) {
        fail_parallel(error, 54);
        return;
    }
    if (kind == 30 && entry.access_pattern == G2_HINT_PATTERN) {
        G2LaneDescV1 const *ctrl = find_lane(descs, header.lane_count, G2_RESIDUAL_CTRL);
        G2LaneDescV1 const *tag = find_lane(descs, header.lane_count, G2_RESIDUAL_TAG);
        G2LaneDescV1 const *value = find_lane(descs, header.lane_count, G2_RESIDUAL_VALUE);
        uint32_t num_words = uint32_t(output_end - output_cursor);
        size_t expected_residual = size_t(num_words) + 1u + entry.local_opcode;
        if (entry.air_idx >= num_airs || ctrl == nullptr || tag == nullptr || value == nullptr ||
            num_words == 0 || num_words > 1023 ||
            residual_end - residual_cursor != expected_residual) {
            fail_parallel(error, 43);
            return;
        }
        DeltaAirOutputDesc output = outputs[entry.air_idx];
        uint32_t hint_row_start = hint_row_offsets[instruction];
        if (output.kind != kind || output.sorted_start > delta_count ||
            hint_row_start > output.count || num_words > output.count - hint_row_start ||
            hint_row_start > delta_count - output.sorted_start ||
            num_words > delta_count - output.sorted_start - hint_row_start ||
            decoded.row_index != hint_row_start || decoded.v0_index != num_words) {
            fail_parallel(error, 59);
            return;
        }
        uint32_t const hint_residual_start = uint32_t(residual_cursor);
        uint64_t memory_pointer_value;
        if (!residual_event(
                wire,
                *ctrl,
                *tag,
                *value,
                residual_cursor++,
                timestamp,
                entry.b,
                0x40u,
                0,
                false,
                memory_pointer_value
            ) ||
            memory_pointer_value > UINT32_MAX) {
            fail_parallel(error, 44);
            return;
        }
        if (entry.local_opcode != 0) {
            uint64_t words;
            if (!residual_event(
                    wire,
                    *ctrl,
                    *tag,
                    *value,
                    residual_cursor++,
                    timestamp + 1,
                    entry.a,
                    0x40u,
                    num_words,
                    true,
                    words
                )) {
                fail_parallel(error, 45);
                return;
            }
        }
        uint32_t memory_pointer = uint32_t(memory_pointer_value);
        for (uint32_t row = 0; row < num_words; ++row) {
            uint64_t word;
            uint32_t word_residual = uint32_t(residual_cursor);
            uint32_t row_timestamp = timestamp + 3u * row;
            uint32_t address = memory_pointer + 8u * row;
            if (address < memory_pointer ||
                !residual_event(
                    wire,
                    *ctrl,
                    *tag,
                    *value,
                    residual_cursor++,
                    row_timestamp + 2,
                    address,
                    0x45u,
                    0,
                    false,
                    word
                )) {
                fail_parallel(error, 46);
                return;
            }
            row_instructions[output.sorted_start + hint_row_start + row] =
                uint32_t(instruction);
            if (row == 0) {
                g2_emit_timeline(
                    timeline, timeline_capacity, initial_timestamp, row_timestamp, 1,
                    uint64_t(entry.b) & ~uint64_t(7), memory_pointer, uint32_t(instruction),
                    hint_residual_start, G2_EVENT_READ_EXPECT, 8, 0, EXPECT_NONE, error
                );
                if (entry.local_opcode != 0) {
                    g2_emit_timeline(
                        timeline, timeline_capacity, initial_timestamp, row_timestamp + 1, 1,
                        uint64_t(entry.a) & ~uint64_t(7), num_words, uint32_t(instruction),
                        hint_residual_start + 1u, G2_EVENT_READ_EXPECT, 8, 0,
                        EXPECT_NONE, error
                    );
                }
            }
            g2_emit_timeline(
                timeline, timeline_capacity, initial_timestamp, row_timestamp + 2, 2,
                uint64_t(address) & ~uint64_t(7), word, uint32_t(instruction),
                word_residual, G2_EVENT_WRITE_SET, 8, 0, EXPECT_NONE, error
            );
        }
        if (residual_cursor != residual_end) {
            fail_parallel(error, 43);
            return;
        }
        atomicAdd(&kind_counts[kind], num_words);
        return;
    }
    if (output_end != output_cursor + 1 || output_cursor >= delta_count) {
        fail_parallel(error, 13);
        return;
    }
    uint64_t lane_v0_value = 0;
    uint64_t lane_v1_value = 0;
    if ((consumes_v0 &&
         !g2_prepared_lane_value(wire, decoded, false, lane_v0_value, error)) ||
        (consumes_v1 &&
         !g2_prepared_lane_value(wire, decoded, true, lane_v1_value, error))) {
        fail_parallel(error, 54);
        return;
    }
    DeltaRecord record{pc_base + decoded.slot * 4u, timestamp, 0, 0};
    uint64_t expected_block = 0;
    uint8_t expected_mode = EXPECT_NONE;
    bool crossing = false;
    uint64_t crossing_value0 = 0;
    uint64_t crossing_value1 = 0;
    if (kind == 29 && entry.access_pattern == G2_ADDI_PATTERN) {
        record.v1 = lane_v0_value;
        record.v2 = uint64_t(int64_t(int32_t(entry.c << 20) >> 20));
    } else if ((kind <= 7 || (kind >= 15 && kind <= 19)) &&
               (entry.access_pattern == 0 || entry.access_pattern == 1)) {
        record.v1 = lane_v0_value;
        record.v2 = (entry.flags & RVR_OPERAND_FLAG_RS2_IMM)
                        ? standard_immediate(entry)
                        : lane_v1_value;
    } else if ((kind == 10 || kind == 11) && entry.access_pattern == 4) {
        record.v1 = lane_v0_value;
        record.v2 = lane_v1_value;
    } else if ((kind == 12 || kind == 14) &&
               (entry.access_pattern == 5 || entry.access_pattern == 6)) {
    } else if (kind == 13 && entry.access_pattern == 7) {
        record.v1 = lane_v0_value;
    } else if (load_store_kind(kind) &&
               (entry.access_pattern == G2_LOAD_PATTERN ||
                entry.access_pattern == G2_STORE_PATTERN)) {
        if (lane_v0_value > UINT32_MAX) {
            fail_parallel(error, 27);
            return;
        }
        record.v1 = uint32_t(lane_v0_value);
        expected_block = lane_v1_value;
        expected_mode = entry.access_pattern == G2_LOAD_PATTERN
                            ? EXPECT_BEFORE_LOAD
                            : EXPECT_BEFORE_STORE;
        record.v2 = entry.access_pattern == G2_LOAD_PATTERN ? lane_v1_value : 0;
        crossing = g2_crosses(uint32_t(record.v1), entry, error);
        if (crossing) {
            G2LaneDescV1 const *ctrl = find_lane(descs, header.lane_count, G2_RESIDUAL_CTRL);
            G2LaneDescV1 const *tag = find_lane(descs, header.lane_count, G2_RESIDUAL_TAG);
            G2LaneDescV1 const *value = find_lane(descs, header.lane_count, G2_RESIDUAL_VALUE);
            uint32_t effective = effective_address(uint32_t(record.v1), entry);
            uint32_t first_timestamp = timestamp +
                (entry.access_pattern == G2_LOAD_PATTERN ? 1u : 2u);
            uint8_t address_space_code =
                (entry.flags & RVR_OPERAND_FLAG_LS_PUBLIC_VALUES) ? 2u : 1u;
            uint8_t residual_tag =
                uint8_t(entry.access_pattern == G2_STORE_PATTERN) |
                uint8_t(address_space_code << 2u) | uint8_t(4u << 4u);
            if (ctrl == nullptr || tag == nullptr || value == nullptr ||
                residual_end - residual_cursor != 2 ||
                !residual_event(
                    wire, *ctrl, *tag, *value, residual_cursor, first_timestamp,
                    effective & ~7u, residual_tag, 0, false, crossing_value0
                ) ||
                !residual_event(
                    wire, *ctrl, *tag, *value, residual_cursor + 1,
                    first_timestamp + 1, (effective & ~7u) + 8u, residual_tag,
                    0, false, crossing_value1
                ) ||
                (entry.access_pattern == G2_LOAD_PATTERN &&
                 crossing_value0 != expected_block)) {
                fail_parallel(error, 18);
                return;
            }
        }
    } else {
        fail_parallel(error, 22);
        return;
    }
    if (decoded.row_index == UINT32_MAX || !g2_emit_standard_direct(
            record,
            expected_block,
            expected_mode,
            crossing,
            uint32_t(residual_cursor),
            crossing_value0,
            crossing_value1,
            entry,
            kind,
            uint32_t(instruction),
            timeline,
            timeline_capacity,
            initial_timestamp,
            error
        )) {
        fail_parallel(error, 66);
        return;
    }
    if (entry.air_idx >= num_airs) {
        fail_parallel(error, 59);
        return;
    }
    DeltaAirOutputDesc output = outputs[entry.air_idx];
    if (output.kind != kind || decoded.row_index >= output.count ||
        output.sorted_start > delta_count ||
        decoded.row_index >= delta_count - output.sorted_start) {
        fail_parallel(error, 59);
        return;
    }
    row_instructions[output.sorted_start + decoded.row_index] = uint32_t(instruction);
    atomicAdd(&kind_counts[kind], 1u);
}

__global__ void g2_validate_parallel_output(
    uint8_t const *wire,
    size_t instruction_count,
    uint32_t const *timestamp_offsets,
    uint32_t const *output_offsets,
    uint32_t const *residual_offsets,
    uint32_t const *opaque_offsets,
    uint32_t const *opaque_occurrences,
    uint32_t const *row_offsets,
    uint32_t const *v0_offsets,
    uint32_t const *v1_offsets,
    size_t count_stride,
    G2ExpectedKindV1 const *expected_kinds,
    size_t expected_kind_count,
    uint32_t const *kind_counts,
    size_t delta_count,
    size_t opaque_capacity,
    uint32_t *opaque_residual_count,
    uint32_t *error
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *error != 0) return;
    G2SegmentHeaderV1 const &header = *reinterpret_cast<G2SegmentHeaderV1 const *>(wire);
    G2LaneDescV1 const *descs = reinterpret_cast<G2LaneDescV1 const *>(wire + 64);
    G2LaneDescV1 const *opaque_events =
        find_lane(descs, header.lane_count, G2_OPAQUE_EVENT_COUNT);
    if (output_offsets[instruction_count] != delta_count ||
        residual_offsets[instruction_count] != header.residual_event_count ||
        opaque_offsets[instruction_count] > opaque_capacity ||
        opaque_occurrences[instruction_count] !=
            (opaque_events == nullptr ? 0 : opaque_events->count) ||
        timestamp_offsets[instruction_count] < timestamp_offsets[0]) {
        fail(error, 23);
        return;
    }
    for (size_t i = 0; i < expected_kind_count; ++i) {
        G2ExpectedKindV1 expected = expected_kinds[i];
        if (expected.kind >= 31 || kind_counts[expected.kind] != expected.count) {
            fail(error, 24);
            return;
        }
        G2LaneDescV1 const *v0 = find_lane(descs, header.lane_count, lane_v0(expected.kind));
        G2LaneDescV1 const *v1 = find_lane(descs, header.lane_count, lane_v1(expected.kind));
        size_t offset = size_t(expected.kind) * count_stride + count_stride - 1;
        if ((expected.kind < 30 && expected.count != row_offsets[offset]) ||
            (v0 == nullptr ? 0 : v0->count) != v0_offsets[offset] ||
            (v1 == nullptr ? 0 : v1->count) != v1_offsets[offset]) {
            fail(error, 26);
            return;
        }
    }
    *opaque_residual_count = opaque_offsets[instruction_count];
}

__global__ void g2_validate_trace_rows(
    G2PreparedInstruction const *prepared,
    size_t instruction_count,
    RvrOperandEntry const *operands,
    size_t operand_count,
    uint32_t const *row_instructions,
    DeltaAirOutputDesc const *outputs,
    size_t num_airs,
    size_t row_count,
    uint32_t *error
) {
    size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= row_count || *error != 0) return;
    uint32_t instruction = row_instructions[row];
    if (instruction >= instruction_count) {
        fail_parallel(error, 81);
        return;
    }
    G2PreparedInstruction decoded = prepared[instruction];
    if (decoded.kind < 0 || decoded.kind >= 31 || decoded.slot >= operand_count) {
        fail_parallel(error, 81);
        return;
    }
    RvrOperandEntry entry = operands[decoded.slot];
    if (entry.air_idx >= num_airs) {
        fail_parallel(error, 81);
        return;
    }
    DeltaAirOutputDesc output = outputs[entry.air_idx];
    if (output.sorted_start > row) {
        fail_parallel(error, 81);
        return;
    }
    size_t local_row = row - output.sorted_start;
    bool row_matches = decoded.kind == 30
                           ? decoded.v0_index != UINT32_MAX &&
                                 local_row >= decoded.row_index &&
                                 local_row - decoded.row_index < decoded.v0_index
                           : local_row == decoded.row_index;
    if (output.kind != uint32_t(decoded.kind) || local_row >= output.count ||
        !row_matches) {
        fail_parallel(error, 81);
    }
}

struct G2RegisterSummary {
    uint32_t last_timestamp;
    uint32_t flags;
    uint64_t last_write_value;
};
static_assert(sizeof(G2RegisterSummary) == 16, "G2 register-summary size drift");

struct G2RegisterState {
    uint32_t timestamp;
    uint32_t reserved;
    uint64_t value;
};
static_assert(sizeof(G2RegisterState) == 16, "G2 register-state size drift");

struct G2TouchedMemoryRecord {
    uint32_t addr_space;
    uint32_t block_ptr;
    uint32_t timestamp;
    uint32_t values[4];
};
static_assert(sizeof(G2TouchedMemoryRecord) == 28, "G2 touched-record size drift");

static constexpr uint32_t G2_SUMMARY_ACCESSED = 1u;
static constexpr uint32_t G2_SUMMARY_WRITTEN = 2u;

__global__ void g2_summarize_register_chunks(
    G2TimelineEvent const *timeline,
    size_t timeline_capacity,
    uint32_t const *timeline_length,
    uint32_t initial_timestamp,
    G2RegisterSummary *summaries,
    size_t chunk_capacity,
    uint32_t *error
) {
    size_t chunk = blockIdx.x;
    if (threadIdx.x != 0 || chunk >= chunk_capacity || *error != 0) return;
    uint32_t length = *timeline_length;
    size_t begin = chunk * G2_REGISTER_REPLAY_CHUNK;
    if (begin >= length || length > timeline_capacity) {
        if (length > timeline_capacity) fail_parallel(error, 67);
        return;
    }
    size_t end = begin + G2_REGISTER_REPLAY_CHUNK;
    if (end > length) end = length;
    G2RegisterSummary local[G2_REGISTER_COUNT]{};
    for (size_t index = begin; index < end; ++index) {
        G2TimelineEvent event = timeline[index];
        if (event.address_key == G2_INVALID_ADDRESS || uint8_t(event.address_key >> 56) != 1)
            continue;
        uint64_t address = event.address_key & ((uint64_t(1) << 56) - 1);
        if ((address & 7u) != 0 || address >= G2_REGISTER_COUNT * sizeof(uint64_t)) {
            fail_parallel(error, 68);
            return;
        }
        uint32_t reg = uint32_t(address / sizeof(uint64_t));
        G2RegisterSummary &summary = local[reg];
        summary.last_timestamp = initial_timestamp + uint32_t(index);
        summary.flags |= G2_SUMMARY_ACCESSED;
        if (event.action == G2_EVENT_WRITE_SET) {
            summary.last_write_value = event.value;
            summary.flags |= G2_SUMMARY_WRITTEN;
        } else if (event.action != G2_EVENT_READ_EXPECT &&
                   event.action != G2_EVENT_READ_PATCH &&
                   event.action != G2_EVENT_READ_ANY) {
            fail_parallel(error, 69);
            return;
        }
    }
    for (uint32_t reg = 0; reg < G2_REGISTER_COUNT; ++reg) {
        summaries[chunk * G2_REGISTER_COUNT + reg] = local[reg];
    }
}

__global__ void g2_scan_register_chunks(
    G2RegisterSummary const *summaries,
    G2RegisterState *incoming,
    size_t chunk_capacity,
    uint32_t const *timeline_length,
    DeviceInitialMemory const *initial_memory,
    size_t initial_memory_count,
    G2RegisterState *final_states,
    uint32_t *error
) {
    uint32_t reg = threadIdx.x;
    if (blockIdx.x != 0 || reg >= G2_REGISTER_COUNT || *error != 0) return;
    if (initial_memory_count <= 1 || initial_memory[1].reserved != 0 ||
        initial_memory[1].cell_size != 2 || initial_memory[1].base == 0 ||
        initial_memory[1].len < G2_REGISTER_COUNT * sizeof(uint64_t)) {
        fail_parallel(error, 70);
        return;
    }
    G2RegisterState state{
        0,
        0,
        *reinterpret_cast<uint64_t const *>(
            initial_memory[1].base + size_t(reg) * sizeof(uint64_t)
        ),
    };
    if (reg == 0) state.value = 0;
    size_t chunks = (size_t(*timeline_length) + G2_REGISTER_REPLAY_CHUNK - 1) /
                    G2_REGISTER_REPLAY_CHUNK;
    if (chunks > chunk_capacity) {
        fail_parallel(error, 67);
        return;
    }
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
        incoming[chunk * G2_REGISTER_COUNT + reg] = state;
        G2RegisterSummary summary = summaries[chunk * G2_REGISTER_COUNT + reg];
        if (summary.flags & G2_SUMMARY_ACCESSED) state.timestamp = summary.last_timestamp;
        if (summary.flags & G2_SUMMARY_WRITTEN) state.value = summary.last_write_value;
    }
    final_states[reg] = state;
}

__global__ void g2_fill_register_chunks(
    G2TimelineEvent *timeline,
    size_t timeline_capacity,
    uint32_t const *timeline_length,
    uint32_t initial_timestamp,
    size_t instruction_count,
    G2RegisterState const *incoming,
    size_t chunk_capacity,
    uint32_t *residual_prev_timestamps,
    uint64_t *residual_prev_values,
    size_t residual_count,
    uint32_t *error
) {
    size_t chunk = blockIdx.x;
    if (threadIdx.x != 0 || chunk >= chunk_capacity || *error != 0) return;
    uint32_t length = *timeline_length;
    size_t begin = chunk * G2_REGISTER_REPLAY_CHUNK;
    if (begin >= length || length > timeline_capacity) return;
    size_t end = begin + G2_REGISTER_REPLAY_CHUNK;
    if (end > length) end = length;
    G2RegisterState state[G2_REGISTER_COUNT];
    for (uint32_t reg = 0; reg < G2_REGISTER_COUNT; ++reg) {
        state[reg] = incoming[chunk * G2_REGISTER_COUNT + reg];
    }
    for (size_t index = begin; index < end; ++index) {
        G2TimelineEvent event = timeline[index];
        if (event.address_key == G2_INVALID_ADDRESS || uint8_t(event.address_key >> 56) != 1)
            continue;
        uint64_t address = event.address_key & ((uint64_t(1) << 56) - 1);
        if ((address & 7u) != 0 || address >= G2_REGISTER_COUNT * sizeof(uint64_t)) {
            fail_parallel(error, 68);
            return;
        }
        uint32_t reg = uint32_t(address / sizeof(uint64_t));
        uint64_t predecessor_value = state[reg].value;
        if (event.instruction != G2_NO_RECORD && event.instruction >= instruction_count) {
            fail_parallel(error, 71);
            return;
        }
        if (event.residual_index != G2_NO_RECORD) {
            if (event.residual_index >= residual_count) {
                fail_parallel(error, 72);
                return;
            }
            residual_prev_timestamps[event.residual_index] = state[reg].timestamp;
            residual_prev_values[event.residual_index] = state[reg].value;
        }
        if (event.action == G2_EVENT_READ_EXPECT) {
            if (state[reg].value != event.value) {
                fail_parallel(error, 73);
                return;
            }
        } else if (event.action == G2_EVENT_READ_PATCH) {
        } else if (event.action == G2_EVENT_WRITE_SET) {
            state[reg].value = event.value;
        } else if (event.action != G2_EVENT_READ_ANY) {
            fail_parallel(error, 69);
            return;
        }
        timeline[index].previous_timestamp = state[reg].timestamp;
        timeline[index].value = predecessor_value;
        state[reg].timestamp = initial_timestamp + uint32_t(index);
    }
}

__global__ void g2_pack_register_touched(
    G2RegisterState const *states,
    G2TouchedMemoryRecord *touched,
    size_t touched_capacity,
    uint32_t *register_touched_count,
    uint32_t *error
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *error != 0) return;
    uint32_t count = 0;
    for (uint32_t reg = 0; reg < G2_REGISTER_COUNT; ++reg) {
        G2RegisterState state = states[reg];
        if (state.timestamp == 0) continue;
        if (count >= touched_capacity) {
            fail_parallel(error, 75);
            return;
        }
        G2TouchedMemoryRecord record{};
        record.addr_space = 1;
        record.block_ptr = reg * 4;
        record.timestamp = state.timestamp;
        record.values[0] = Fp(uint16_t(state.value)).asRaw();
        record.values[1] = Fp(uint16_t(state.value >> 16)).asRaw();
        record.values[2] = Fp(uint16_t(state.value >> 32)).asRaw();
        record.values[3] = Fp(uint16_t(state.value >> 48)).asRaw();
        touched[count++] = record;
    }
    *register_touched_count = count;
}

__global__ void g2_mark_memory_events(
    G2TimelineEvent const *timeline,
    uint32_t *indices,
    uint8_t *flags,
    size_t count
) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    indices[index] = uint32_t(index);
    uint64_t key = timeline[index].address_key;
    flags[index] = key != G2_INVALID_ADDRESS && uint8_t(key >> 56) != 1;
}

__global__ void g2_extract_memory_keys(
    G2TimelineEvent const *timeline,
    uint32_t const *event_indices,
    uint64_t *keys,
    size_t count
) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) keys[index] = timeline[event_indices[index]].address_key;
}

__global__ void g2_mark_memory_group_starts(
    uint64_t const *keys, uint32_t *indices, uint8_t *flags, size_t count
) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    indices[index] = uint32_t(index);
    flags[index] = index == 0 || keys[index - 1] != keys[index];
}

struct G2MemoryTransform {
    uint64_t keep;
    uint64_t bits;
};
static_assert(sizeof(G2MemoryTransform) == 16, "G2 memory-transform size drift");

struct G2ComposeMemoryTransform {
    __host__ __device__ __forceinline__ G2MemoryTransform operator()(
        G2MemoryTransform before, G2MemoryTransform after
    ) const {
        return {
            before.keep & after.keep,
            (before.bits & after.keep) | after.bits,
        };
    }
};

struct G2EqualAddressKey {
    __host__ __device__ __forceinline__ bool operator()(uint64_t lhs, uint64_t rhs) const {
        return lhs == rhs;
    }
};

__global__ void g2_prepare_memory_transforms(
    uint32_t const *event_indices,
    G2TimelineEvent const *timeline,
    size_t event_count,
    size_t instruction_count,
    G2MemoryTransform *transforms,
    uint32_t *error
) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= event_count || *error != 0) return;
    G2TimelineEvent event = timeline[event_indices[index]];
    G2MemoryTransform transform{UINT64_MAX, 0};
    if (event.action == G2_EVENT_WRITE_SET ||
        event.action == G2_EVENT_WRITE_SET_EXPECT) {
        transform = {0, event.value};
    } else if (event.action == G2_EVENT_STORE_PATCH) {
        uint32_t timeline_index = event_indices[index];
        if (event.instruction == G2_NO_RECORD ||
            event.instruction >= instruction_count || timeline_index == 0 ||
            event.width == 0 || event.width > 8 || event.byte_offset > 7 ||
            uint32_t(event.byte_offset) + event.width > 8) {
            fail_parallel(error, 78);
            return;
        }
        G2TimelineEvent source_event = timeline[timeline_index - 1];
        if (source_event.instruction != event.instruction ||
            uint8_t(source_event.address_key >> 56) != 1 ||
            (source_event.action != G2_EVENT_READ_PATCH &&
             source_event.action != G2_EVENT_READ_EXPECT)) {
            fail_parallel(error, 78);
            return;
        }
        uint64_t source = source_event.value;
        uint32_t shift = uint32_t(event.byte_offset) * 8u;
        uint64_t mask = event.width == 8
                            ? UINT64_MAX
                            : (uint64_t(1) << (uint32_t(event.width) * 8u)) - 1u;
        mask <<= shift;
        transform = {~mask, (source << shift) & mask};
    } else if (event.action != G2_EVENT_READ_EXPECT &&
               event.action != G2_EVENT_READ_ANY) {
        fail_parallel(error, 79);
        return;
    }
    transforms[index] = transform;
}

__device__ __forceinline__ bool g2_load_initial_memory_value(
    uint64_t key,
    DeviceInitialMemory const *initial_memory,
    size_t initial_memory_count,
    uint64_t &value,
    uint32_t *error
) {
    uint32_t address_space = uint32_t(key >> 56);
    uint64_t byte_address = key & ((uint64_t(1) << 56) - 1);
    if (address_space == 0 || address_space == 1 || address_space >= initial_memory_count ||
        initial_memory[address_space].reserved != 0 ||
        initial_memory[address_space].cell_size != 2 ||
        initial_memory[address_space].base == 0 ||
        byte_address > initial_memory[address_space].len ||
        initial_memory[address_space].len - byte_address < sizeof(uint64_t)) {
        fail_parallel(error, 76);
        return false;
    }
    value = *reinterpret_cast<uint64_t const *>(
        initial_memory[address_space].base + byte_address
    );
    return true;
}

__global__ void g2_replay_memory_events(
    uint8_t const *wire,
    uint64_t const *keys,
    uint32_t const *event_indices,
    G2TimelineEvent *timeline,
    size_t event_count,
    G2MemoryTransform const *inclusive_transforms,
    G2PreparedInstruction const *prepared,
    size_t instruction_count,
    uint32_t initial_timestamp,
    DeviceInitialMemory const *initial_memory,
    size_t initial_memory_count,
    uint32_t *residual_prev_timestamps,
    uint64_t *residual_prev_values,
    size_t residual_count,
    uint32_t *error
) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= event_count || *error != 0) return;
    uint64_t key = keys[index];
    uint64_t initial_value;
    if (!g2_load_initial_memory_value(
            key, initial_memory, initial_memory_count, initial_value, error
        ))
        return;
    G2MemoryTransform prefix{UINT64_MAX, 0};
    uint32_t previous_timestamp = 0;
    if (index != 0 && keys[index - 1] == key) {
        prefix = inclusive_transforms[index - 1];
        previous_timestamp = initial_timestamp + event_indices[index - 1];
    }
    uint64_t current_value = (initial_value & prefix.keep) | prefix.bits;
    G2TimelineEvent event = timeline[event_indices[index]];
    if (event.instruction != G2_NO_RECORD && event.instruction >= instruction_count) {
        fail_parallel(error, 71);
        return;
    }
    if (event.expected_mode == EXPECT_BEFORE_LOAD ||
        event.expected_mode == EXPECT_BEFORE_STORE) {
        uint64_t expected_value = event.value;
        if (event.action == G2_EVENT_WRITE_SET_EXPECT) {
            if (event.instruction == G2_NO_RECORD ||
                !g2_prepared_lane_value(
                    wire, prepared[event.instruction], true, expected_value, error
                )) {
                fail_parallel(error, 77);
                return;
            }
        }
        if (current_value != expected_value) {
            fail_parallel(error, 77);
            return;
        }
    }
    if (event.action == G2_EVENT_READ_EXPECT && event.expected_mode == EXPECT_NONE &&
        current_value != event.value) {
        fail_parallel(error, 77);
        return;
    }
    if (event.residual_index != G2_NO_RECORD) {
        if (event.residual_index >= residual_count) {
            fail_parallel(error, 72);
            return;
        }
        residual_prev_timestamps[event.residual_index] = previous_timestamp;
        residual_prev_values[event.residual_index] = current_value;
    }
    G2MemoryTransform inclusive = inclusive_transforms[index];
    uint64_t next_value = (initial_value & inclusive.keep) | inclusive.bits;
    if (event.expected_mode == EXPECT_AFTER_STORE && next_value != event.value) {
        fail_parallel(error, 80);
    }
    timeline[event_indices[index]].previous_timestamp = previous_timestamp;
    timeline[event_indices[index]].value = current_value;
}

__global__ void g2_pack_memory_touched(
    uint64_t const *keys,
    uint32_t const *event_indices,
    G2TimelineEvent const *timeline,
    size_t event_count,
    G2MemoryTransform const *inclusive_transforms,
    uint32_t const *group_starts,
    uint32_t const *group_count,
    DeviceInitialMemory const *initial_memory,
    size_t initial_memory_count,
    G2TouchedMemoryRecord *touched,
    size_t touched_capacity,
    uint64_t *dirty_pages,
    size_t dirty_page_words,
    uint32_t const *register_touched_count,
    uint32_t initial_timestamp,
    uint32_t *error
) {
    size_t group = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t groups = *group_count;
    if (group >= groups || *error != 0) return;
    size_t begin = group_starts[group];
    size_t end = group + 1 < groups ? group_starts[group + 1] : event_count;
    if (begin >= end || end > event_count) {
        fail_parallel(error, 81);
        return;
    }
    size_t last = end - 1;
    uint64_t key = keys[begin];
    uint64_t initial_value;
    if (!g2_load_initial_memory_value(
            key, initial_memory, initial_memory_count, initial_value, error
        ))
        return;
    G2MemoryTransform inclusive = inclusive_transforms[last];
    uint64_t current_value = (initial_value & inclusive.keep) | inclusive.bits;
    uint32_t address_space = uint32_t(key >> 56);
    uint64_t byte_address = key & ((uint64_t(1) << 56) - 1);
    if (address_space == G2_MAIN_MEMORY_ADDRESS_SPACE &&
        (inclusive.keep != UINT64_MAX || inclusive.bits != 0)) {
        uint64_t page = byte_address / G2_CONTINUATION_PAGE_BYTES;
        uint64_t word = page / 64;
        if (dirty_pages == nullptr || word >= dirty_page_words) {
            fail_parallel(error, 82);
            return;
        }
        atomicOr(
            reinterpret_cast<unsigned long long *>(&dirty_pages[word]),
            1ull << (page % 64)
        );
    }
    uint32_t previous_timestamp = initial_timestamp + event_indices[last];
    uint32_t output = *register_touched_count + uint32_t(group);
    if (output >= touched_capacity || byte_address > uint64_t(UINT32_MAX) * 2u + 1u) {
        fail_parallel(error, 75);
        return;
    }
    G2TouchedMemoryRecord record{};
    record.addr_space = address_space;
    record.block_ptr = uint32_t(byte_address / 2u);
    record.timestamp = previous_timestamp;
    record.values[0] = Fp(uint16_t(current_value)).asRaw();
    record.values[1] = Fp(uint16_t(current_value >> 16)).asRaw();
    record.values[2] = Fp(uint16_t(current_value >> 32)).asRaw();
    record.values[3] = Fp(uint16_t(current_value >> 48)).asRaw();
    touched[output] = record;
}

__global__ void g2_finish_touched_count(
    uint32_t const *register_count,
    uint32_t const *memory_count,
    size_t capacity,
    uint32_t *total,
    uint32_t *error
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *error != 0) return;
    uint64_t count = uint64_t(*register_count) + *memory_count;
    if (count > capacity) {
        fail_parallel(error, 75);
        return;
    }
    *total = uint32_t(count);
}

__global__ void g2_predecode(
    uint8_t const *wire,
    size_t wire_storage_bytes,
    size_t logical_wire_bytes,
    uint8_t const *expected_fingerprint,
    G2BlockEntryV1 const *blocks,
    size_t block_count,
    RvrOperandEntry const *operands,
    size_t operand_count,
    G2PreparedInstruction const *prepared,
    size_t program_instruction_count,
    bool validate_only,
    uint32_t pc_base,
    DeviceInitialMemory const *initial_memory,
    size_t initial_memory_count,
    uint32_t initial_timestamp,
    G2ExpectedKindV1 const *expected_kinds,
    size_t expected_kind_count,
    G2ExpectedOpaqueV1 const *expected_opaque,
    size_t expected_opaque_count,
    uint32_t *program_frequencies,
    size_t frequency_count,
    DeltaRecord *delta_output,
    size_t delta_count,
    uint64_t *expected_blocks,
    uint8_t *expected_modes,
    DeltaMemoryLogEntry *opaque_residual_output,
    size_t opaque_residual_capacity,
    uint32_t *opaque_residual_count,
    uint32_t *error
) {
    if (*error != 0) return;
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (wire_storage_bytes < 128 || (wire_storage_bytes & 127u) != 0 ||
        logical_wire_bytes < wire_storage_bytes || (logical_wire_bytes & 127u) != 0 ||
        block_count == 0 || initial_memory_count <= 1 || expected_kind_count == 0 ||
        program_instruction_count == 0 || prepared == nullptr) {
        fail(error, 1);
        return;
    }
    G2SegmentHeaderV1 const &header = *reinterpret_cast<G2SegmentHeaderV1 const *>(wire);
    uint8_t const magic[8] = {'O', 'V', 'M', 'G', '2', 'W', '1', '\0'};
    for (size_t i = 0; i < 8; ++i) {
        if (header.magic[i] != magic[i]) {
            fail(error, 2);
            return;
        }
    }
    if (header.version != 1 || header.lane_count == 0 ||
        header.lane_count > 59 + expected_opaque_count ||
        header.header_bytes != 64 + 32 * header.lane_count ||
        header.header_bytes > wire_storage_bytes || header.flags != G2_FLAGS_COMMITTED_V1) {
        fail(error, 3);
        return;
    }
    for (size_t i = 0; i < 32; ++i) {
        if (header.schema_fingerprint[i] != expected_fingerprint[i]) {
            fail(error, 4);
            return;
        }
    }
    G2LaneDescV1 const *descs = reinterpret_cast<G2LaneDescV1 const *>(wire + 64);
    for (size_t i = 0; i < header.lane_count; ++i) {
        G2LaneDescV1 const &desc = descs[i];
        uint8_t width = 0;
        uint32_t flags = 0, group = 0;
        G2ExpectedOpaqueV1 const *opaque =
            find_expected_opaque(expected_opaque, expected_opaque_count, desc.kind);
        bool standard = lane_spec(desc.kind, width, flags, group);
        uint64_t end = desc.offset + desc.payload_bytes;
        if ((i != 0 && descs[i - 1].kind >= desc.kind) ||
            (!standard && opaque == nullptr) ||
            (standard &&
             (desc.elem_width != width || desc.encoding != 0 || desc.flags != flags ||
              desc.group_id != group ||
              desc.payload_bytes != uint64_t(desc.count) * width)) ||
            (opaque != nullptr &&
             (standard || desc.elem_width != 0 || desc.encoding != 1 || desc.flags != 4 ||
              desc.group_id != 0 || desc.count != opaque->count ||
              desc.payload_bytes != opaque->payload_bytes ||
              uint64_t(desc.count) * opaque->stride != desc.payload_bytes)) ||
            desc.reserved != 0 ||
            desc.offset < header.header_bytes || (desc.offset & 127u) != 0 ||
            end < desc.offset ||
            (opaque == nullptr ? end > wire_storage_bytes : end > logical_wire_bytes)) {
            fail(error, 5);
            return;
        }
        for (size_t j = 0; j < i; ++j) {
            uint64_t other_end = descs[j].offset + descs[j].payload_bytes;
            if (desc.offset < other_end && descs[j].offset < end) {
                fail(error, 6);
                return;
            }
        }
    }
    for (size_t i = 0; i < expected_opaque_count; ++i) {
        if (find_lane(descs, header.lane_count, uint16_t(expected_opaque[i].lane_kind)) ==
            nullptr) {
            fail(error, 39);
            return;
        }
    }
    G2LaneDescV1 const *run = find_lane(descs, header.lane_count, G2_RUN_BLOCK_ID);
    G2LaneDescV1 const *residual_ctrl = find_lane(descs, header.lane_count, G2_RESIDUAL_CTRL);
    G2LaneDescV1 const *residual_tag = find_lane(descs, header.lane_count, G2_RESIDUAL_TAG);
    G2LaneDescV1 const *residual_value = find_lane(descs, header.lane_count, G2_RESIDUAL_VALUE);
    G2LaneDescV1 const *opaque_event_counts =
        find_lane(descs, header.lane_count, G2_OPAQUE_EVENT_COUNT);
    if (run == nullptr || run->count != header.run_count ||
        (header.instruction_count != 0 && run->count == 0) ||
        (header.residual_event_count != 0) != (residual_ctrl != nullptr) ||
        (residual_ctrl != nullptr &&
         (residual_tag == nullptr || residual_value == nullptr ||
          residual_ctrl->count != header.residual_event_count ||
          residual_tag->count != header.residual_event_count ||
          residual_value->count != header.residual_event_count))) {
        fail(error, 7);
        return;
    }
    G2LaneDescV1 const *v0_lanes[31]{};
    G2LaneDescV1 const *v1_lanes[31]{};
    for (uint32_t kind = 0; kind < 30; ++kind) {
        v0_lanes[kind] = find_lane(descs, header.lane_count, lane_v0(kind));
        v1_lanes[kind] = find_lane(descs, header.lane_count, lane_v1(kind));
        if (!load_store_kind(kind)) continue;
        G2LaneDescV1 const *v0 = v0_lanes[kind];
        G2LaneDescV1 const *v1 = v1_lanes[kind];
        if ((v0 == nullptr) != (v1 == nullptr) ||
            (v0 != nullptr && v0->count != v1->count)) {
            fail(error, 8);
            return;
        }
    }
    int8_t kind_by_air[256];
    for (size_t i = 0; i < 256; ++i) kind_by_air[i] = -1;
    for (size_t i = 0; i < expected_kind_count; ++i) {
        G2ExpectedKindV1 const &expected = expected_kinds[i];
        if (expected.air_idx >= 256 || expected.kind >= 31 ||
            kind_by_air[expected.air_idx] != -1) {
            fail(error, 48);
            return;
        }
        kind_by_air[expected.air_idx] = int8_t(expected.kind);
    }
    if (program_instruction_count != header.instruction_count) {
        fail(error, 52);
        return;
    }
    if (validate_only) return;

    DeviceInitialMemory registers_image = initial_memory[1];
    if (registers_image.reserved != 0 || registers_image.cell_size != 2 ||
        registers_image.base == 0 || registers_image.len < 32 * sizeof(uint64_t)) {
        fail(error, 9);
        return;
    }
    uint64_t registers[32];
    for (size_t i = 0; i < 32; ++i)
        registers[i] = *reinterpret_cast<uint64_t const *>(registers_image.base + i * 8);
    registers[0] = 0;
    size_t kind_v0_cursors[31]{};
    size_t kind_v1_cursors[31]{};
    size_t kind_counts[31]{};
    size_t residual_cursor = 0;
    size_t opaque_residual_cursor = 0;
    size_t opaque_event_cursor = 0;
    size_t instruction_cursor = 0;
    size_t output_cursor = 0;
    uint32_t timestamp = initial_timestamp;

    for (; instruction_cursor < program_instruction_count; ++instruction_cursor) {
            if (instruction_cursor + 64 < program_instruction_count) {
                G2PreparedInstruction const *future = prepared + instruction_cursor + 64;
                asm volatile("prefetch.global.L2 [%0];" : : "l"(future));
            }
            G2PreparedInstruction const decoded = prepared[instruction_cursor];
            uint32_t slot = decoded.slot;
            if (slot >= operand_count) {
                fail(error, 11);
                return;
            }
            RvrOperandEntry const entry = operands[slot];
            if (entry.filtered_index >= frequency_count) {
                fail(error, 12);
                return;
            }
            if (entry.air_idx == INVALID_AIR) continue;
            if (entry.air_idx >= 256) {
                fail(error, 48);
                return;
            }
            int32_t kind_code = kind_by_air[entry.air_idx];
            if (kind_code != decoded.kind) {
                fail(error, 56);
                return;
            }
            if (kind_code < 0 && entry.access_pattern == G2_OPAQUE_TIMESTAMP_PATTERN) {
                ++timestamp;
                continue;
            }
            if (kind_code < 0 && entry.access_pattern == G2_OPAQUE_PATTERN) {
                if (residual_ctrl == nullptr || opaque_event_counts == nullptr ||
                    opaque_event_cursor >= opaque_event_counts->count) {
                    fail(error, 40);
                    return;
                }
                uint32_t event_count =
                    lane_u32(wire, *opaque_event_counts, opaque_event_cursor++);
                if (event_count == 0 || residual_cursor + size_t(event_count) < residual_cursor ||
                    residual_cursor + size_t(event_count) > residual_ctrl->count) {
                    fail(error, 47);
                    return;
                }
                for (uint32_t event = 0; event < event_count; ++event) {
                    uint64_t control = lane_u64(wire, *residual_ctrl, residual_cursor);
                    if (uint32_t(control) != timestamp || !append_opaque_residual(
                            wire,
                            *residual_ctrl,
                            *residual_tag,
                            *residual_value,
                            residual_cursor,
                            opaque_residual_output,
                            opaque_residual_capacity,
                            opaque_residual_cursor,
                            registers
                        )) {
                        fail(error, 41);
                        return;
                    }
                    ++residual_cursor;
                    ++timestamp;
                }
                continue;
            }
            if (kind_code < 0 || output_cursor >= delta_count) {
                fail(error, 13);
                return;
            }
            uint32_t kind = uint32_t(kind_code);
            bool consumes_v0, consumes_v1;
            lane_consumption(kind, entry, consumes_v0, consumes_v1);
            uint64_t prepared_v0 = 0, prepared_v1 = 0;
            if (consumes_v0) {
                size_t cursor = kind_v0_cursors[kind]++;
                if (cursor != decoded.v0_index) {
                    fail(error, 54);
                    return;
                }
                if (!g2_prepared_lane_value(
                        wire, decoded, false, prepared_v0, error
                    ))
                    return;
            } else if (decoded.v0_index != UINT32_MAX) {
                fail(error, 54);
                return;
            }
            if (consumes_v1) {
                size_t cursor = kind_v1_cursors[kind]++;
                if (cursor != decoded.v1_index) {
                    fail(error, 55);
                    return;
                }
                if (!g2_prepared_lane_value(
                        wire, decoded, true, prepared_v1, error
                    ))
                    return;
            } else if (decoded.v1_index != UINT32_MAX) {
                fail(error, 55);
                return;
            }
            DeltaRecord record{pc_base + slot * 4u, timestamp, 0, 0};
            uint64_t expected_block = 0;
            uint8_t expected_mode = EXPECT_NONE;
            if (kind == 30 && entry.access_pattern == G2_HINT_PATTERN) {
                if (residual_ctrl == nullptr || (entry.b & 7u) != 0 || entry.b >= 32u * 8u ||
                    (entry.local_opcode != 0 &&
                     ((entry.a & 7u) != 0 || entry.a >= 32u * 8u))) {
                    fail(error, 42);
                    return;
                }
                uint32_t num_words =
                    entry.local_opcode == 0 ? 1u : uint32_t(registers[entry.a / 8u]);
                if (num_words == 0 || num_words > 1023 ||
                    residual_cursor + size_t(num_words) + 1u + entry.local_opcode >
                        residual_ctrl->count ||
                    output_cursor + num_words > delta_count || registers[entry.b / 8u] > UINT32_MAX) {
                    fail(error, 43);
                    return;
                }
                uint64_t actual;
                uint32_t memory_pointer = uint32_t(registers[entry.b / 8u]);
                if (!residual_event(
                        wire,
                        *residual_ctrl,
                        *residual_tag,
                        *residual_value,
                        residual_cursor++,
                        timestamp,
                        entry.b,
                        0x40u,
                        memory_pointer,
                        true,
                        actual
                    )) {
                    fail(error, 44);
                    return;
                }
                if (entry.local_opcode != 0 &&
                    !residual_event(
                        wire,
                        *residual_ctrl,
                        *residual_tag,
                        *residual_value,
                        residual_cursor++,
                        timestamp + 1u,
                        entry.a,
                        0x40u,
                        num_words,
                        true,
                        actual
                    )) {
                    fail(error, 45);
                    return;
                }
                for (uint32_t row = 0; row < num_words; ++row) {
                    uint64_t word;
                    uint32_t row_timestamp = timestamp + 3u * row;
                    uint32_t address = memory_pointer + 8u * row;
                    if (address < memory_pointer ||
                        !residual_event(
                            wire,
                            *residual_ctrl,
                            *residual_tag,
                            *residual_value,
                            residual_cursor++,
                            row_timestamp + 2u,
                            address,
                            0x45u,
                            0,
                            false,
                            word
                        )) {
                        fail(error, 46);
                        return;
                    }
                    DeltaRecord hint_record{
                        pc_base + slot * 4u,
                        row_timestamp,
                        uint64_t(address) | (uint64_t(row) << 32) |
                            (uint64_t(num_words) << 42),
                        word,
                    };
                    delta_output[output_cursor] = hint_record;
                    expected_blocks[output_cursor] = 0;
                    expected_modes[output_cursor] = EXPECT_NONE;
                    ++output_cursor;
                    ++kind_counts[kind];
                }
                timestamp += 3u * num_words;
                continue;
            } else if (kind == 29 && entry.access_pattern == G2_ADDI_PATTERN) {
                if ((entry.a & 7u) != 0 || (entry.b & 7u) != 0 ||
                    entry.a >= 32 * 8 || entry.b >= 32 * 8) {
                    fail(error, 14);
                    return;
                }
                uint32_t rd = entry.a / 8, rs1 = entry.b / 8;
                record.v1 = prepared_v0;
                if (record.v1 != registers[rs1]) {
                    fail(error, 15);
                    return;
                }
                record.v2 = uint64_t(int64_t(int32_t(entry.c << 20) >> 20));
                if (rd != 0) registers[rd] = record.v1 + record.v2;
                timestamp += 2;
            } else if ((kind <= 7 || (kind >= 15 && kind <= 19)) &&
                       (entry.access_pattern == 0 || entry.access_pattern == 1)) {
                if ((entry.a & 7u) != 0 || (entry.b & 7u) != 0 ||
                    entry.a >= 32 * 8 || entry.b >= 32 * 8) {
                    fail(error, 28);
                    return;
                }
                uint32_t rd = entry.a / 8, rs1 = entry.b / 8;
                record.v1 = prepared_v0;
                if (record.v1 != registers[rs1]) {
                    fail(error, 29);
                    return;
                }
                if (entry.flags & RVR_OPERAND_FLAG_RS2_IMM) {
                    record.v2 = standard_immediate(entry);
                } else {
                    if ((entry.c & 7u) != 0 || entry.c >= 32 * 8) {
                        fail(error, 30);
                        return;
                    }
                    record.v2 = prepared_v1;
                    if (record.v2 != registers[entry.c / 8]) {
                        fail(error, 31);
                        return;
                    }
                }
                uint64_t result;
                if (!standard_post_write(kind, entry, record.v1, record.v2, result)) {
                    fail(error, 32);
                    return;
                }
                if (rd != 0) registers[rd] = result;
                timestamp += 3;
            } else if ((kind == 10 || kind == 11) && entry.access_pattern == 4) {
                if ((entry.a & 7u) != 0 || (entry.b & 7u) != 0 ||
                    entry.a >= 32 * 8 || entry.b >= 32 * 8) {
                    fail(error, 33);
                    return;
                }
                record.v1 = prepared_v0;
                record.v2 = prepared_v1;
                if (record.v1 != registers[entry.a / 8] || record.v2 != registers[entry.b / 8]) {
                    fail(error, 34);
                    return;
                }
                timestamp += 2;
            } else if ((kind == 12 || kind == 14) &&
                       (entry.access_pattern == 5 || entry.access_pattern == 6)) {
                bool write_enabled = entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED;
                if (write_enabled) {
                    if ((entry.a & 7u) != 0 || entry.a >= 32 * 8) {
                        fail(error, 35);
                        return;
                    }
                    uint64_t result = kind == 14
                                          ? uint64_t(record.from_pc) +
                                                uint64_t(int64_t(int32_t(entry.c << 8)))
                                          : ((entry.flags & RVR_OPERAND_FLAG_IS_JAL)
                                                 ? uint64_t(record.from_pc + 4u)
                                                 : uint64_t(int64_t(int32_t(entry.c << 12))));
                    uint32_t rd = entry.a / 8;
                    if (rd != 0) registers[rd] = result;
                } else if (kind == 14) {
                    fail(error, 36);
                    return;
                }
                timestamp += 1;
            } else if (kind == 13 && entry.access_pattern == 7) {
                if ((entry.a & 7u) != 0 || (entry.b & 7u) != 0 ||
                    entry.a >= 32 * 8 || entry.b >= 32 * 8) {
                    fail(error, 37);
                    return;
                }
                record.v1 = prepared_v0;
                if (record.v1 != registers[entry.b / 8]) {
                    fail(error, 38);
                    return;
                }
                if (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED) {
                    uint32_t rd = entry.a / 8;
                    if (rd != 0) registers[rd] = uint64_t(record.from_pc + 4u);
                }
                timestamp += 2;
            } else if (load_store_kind(kind) &&
                       (entry.access_pattern == G2_LOAD_PATTERN ||
                        entry.access_pattern == G2_STORE_PATTERN)) {
                if ((entry.a & 7u) != 0 || (entry.b & 7u) != 0 ||
                    entry.a >= 32 * 8 || entry.b >= 32 * 8) {
                    fail(error, 16);
                    return;
                }
                uint32_t operand_a = entry.a / 8, base_reg = entry.b / 8;
                bool narrow_reveal =
                    (entry.flags & RVR_OPERAND_FLAG_LS_PUBLIC_VALUES) &&
                    entry.access_pattern == G2_STORE_PATTERN && entry.local_opcode != 4;
                uint32_t pointer;
                uint64_t block_value;
                if (narrow_reveal) {
                    if (residual_ctrl == nullptr || residual_cursor + 3 > residual_ctrl->count) {
                        fail(error, 17);
                        return;
                    }
                    uint64_t pointer_value;
                    uint64_t source_value;
                    uint64_t post_block;
                    if (registers[base_reg] > UINT32_MAX) {
                        fail(error, 27);
                        return;
                    }
                    uint32_t pointer_u32 = uint32_t(registers[base_reg]);
                    uint32_t effective = effective_address(pointer_u32, entry);
                    if (!residual_event(wire, *residual_ctrl, *residual_tag, *residual_value,
                                        residual_cursor, timestamp, entry.b, 0x40u,
                                        registers[base_reg], true, pointer_value) ||
                        !residual_event(wire, *residual_ctrl, *residual_tag, *residual_value,
                                        residual_cursor + 1, timestamp + 1, entry.a, 0x40u,
                                        registers[operand_a], true, source_value) ||
                        !residual_event(wire, *residual_ctrl, *residual_tag, *residual_value,
                                        residual_cursor + 2, timestamp + 2, effective & ~7u,
                                        0x49u, 0, false, post_block)) {
                        fail(error, 18);
                        return;
                    }
                    if (pointer_value > UINT32_MAX) {
                        fail(error, 27);
                        return;
                    }
                    pointer = pointer_u32;
                    record.v2 = source_value;
                    expected_block = post_block;
                    expected_mode = EXPECT_AFTER_STORE;
                    residual_cursor += 3;
                } else {
                    pointer = uint32_t(prepared_v0);
                    block_value = prepared_v1;
                    if (uint64_t(pointer) != registers[base_reg]) {
                        fail(error, 20);
                        return;
                    }
                    expected_block = block_value;
                    expected_mode = entry.access_pattern == G2_LOAD_PATTERN
                                        ? EXPECT_BEFORE_LOAD
                                        : EXPECT_BEFORE_STORE;
                    record.v2 = entry.access_pattern == G2_LOAD_PATTERN
                                    ? block_value
                                    : registers[operand_a];
                }
                record.v1 = pointer;
                if (entry.access_pattern == G2_LOAD_PATTERN) {
                    uint64_t result;
                    if (!load_value(entry, pointer, record.v2, result)) {
                        fail(error, 21);
                        return;
                    }
                    if (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED) registers[operand_a] = result;
                }
                timestamp += 3;
            } else {
                fail(error, 22);
                return;
            }
            ++kind_counts[kind];
            delta_output[output_cursor] = record;
            expected_blocks[output_cursor] = expected_block;
            expected_modes[output_cursor] = expected_mode;
            ++output_cursor;
    }
    if (instruction_cursor != header.instruction_count || output_cursor != delta_count ||
        residual_cursor != header.residual_event_count ||
        opaque_event_cursor != (opaque_event_counts == nullptr ? 0 : opaque_event_counts->count)) {
        fail(error, 23);
        return;
    }
    *opaque_residual_count = uint32_t(opaque_residual_cursor);
    for (size_t i = 0; i < expected_kind_count; ++i) {
        G2ExpectedKindV1 const &expected = expected_kinds[i];
        if (expected.kind >= 31 || kind_counts[expected.kind] != expected.count) {
            fail(error, 24);
            return;
        }
    }
    G2LaneDescV1 const *addi = v0_lanes[29];
    if ((addi == nullptr ? 0 : addi->count) != kind_v0_cursors[29]) {
        fail(error, 25);
        return;
    }
    for (uint32_t kind = 0; kind < 30; ++kind) {
        G2LaneDescV1 const *v0 = v0_lanes[kind];
        G2LaneDescV1 const *v1 = v1_lanes[kind];
        if ((v0 == nullptr ? 0 : v0->count) != kind_v0_cursors[kind] ||
            (v1 == nullptr ? 0 : v1->count) != kind_v1_cursors[kind]) {
            fail(error, 26);
            return;
        }
    }
}

} // namespace

extern "C" int _rvr_g2_tracegen(uint32_t kind, RVR_G2_TRACEGEN_PARAMETERS) {
#define G2_DISPATCH(name)                                                                \
    return name(                                                                         \
        trace, height, width, source, operand_table, pc_base, pointer_max_bits,           \
        range_checker, range_checker_num_bins, bitwise_lookup, range_tuple_checker,       \
        range_tuple_checker_sizes, timestamp_max_bits, stream                             \
    )
    switch (kind) {
    case 0: G2_DISPATCH(_add_sub_tracegen_g2);
    case 1: G2_DISPATCH(_bitwise_logic_tracegen_g2);
    case 2: G2_DISPATCH(_rv64_less_than_tracegen_g2);
    case 3: G2_DISPATCH(_rv64_shift_logical_tracegen_g2);
    case 4: G2_DISPATCH(_rv64_shift_right_arithmetic_tracegen_g2);
    case 5: G2_DISPATCH(_rv64_add_sub_w_tracegen_g2);
    case 6: G2_DISPATCH(_rv64_shift_w_logical_tracegen_g2);
    case 7: G2_DISPATCH(_rv64_shift_w_right_arithmetic_tracegen_g2);
    case 8: G2_DISPATCH(_rv64_load_byte_tracegen_g2);
    case 9: G2_DISPATCH(_rv64_load_sign_extend_byte_tracegen_g2);
    case 10: G2_DISPATCH(_beq_tracegen_g2);
    case 11: G2_DISPATCH(_blt_tracegen_g2);
    case 12: G2_DISPATCH(_jal_lui_tracegen_g2);
    case 13: G2_DISPATCH(_jalr_tracegen_g2);
    case 14: G2_DISPATCH(_auipc_tracegen_g2);
    case 15: G2_DISPATCH(_mul_tracegen_g2);
    case 16: G2_DISPATCH(_mulh_tracegen_g2);
    case 17: G2_DISPATCH(_rv64_mul_w_tracegen_g2);
    case 18: G2_DISPATCH(_rv64_div_rem_tracegen_g2);
    case 19: G2_DISPATCH(_rv64_div_rem_w_tracegen_g2);
    case 20: G2_DISPATCH(_rv64_load_halfword_tracegen_g2);
    case 21: G2_DISPATCH(_rv64_load_word_tracegen_g2);
    case 22: G2_DISPATCH(_rv64_load_doubleword_tracegen_g2);
    case 23: G2_DISPATCH(_rv64_store_byte_tracegen_g2);
    case 24: G2_DISPATCH(_rv64_store_halfword_tracegen_g2);
    case 25: G2_DISPATCH(_rv64_store_word_tracegen_g2);
    case 26: G2_DISPATCH(_rv64_store_doubleword_tracegen_g2);
    case 27: G2_DISPATCH(_rv64_load_sign_extend_halfword_tracegen_g2);
    case 28: G2_DISPATCH(_rv64_load_sign_extend_word_tracegen_g2);
    case 29: G2_DISPATCH(_addi_tracegen_g2_common);
    case 30: G2_DISPATCH(_hintstore_tracegen_g2);
    default: return int(cudaErrorInvalidValue);
    }
#undef G2_DISPATCH
}

extern "C" int _rvr_g2_tracegen_reference(
    uint32_t kind, RVR_G2_REFERENCE_PARAMETERS
) {
#define G2_REFERENCE_DISPATCH(name)                                                      \
    return name(                                                                         \
        trace, height, width, records, operand_table, pc_base, pointer_max_bits,          \
        range_checker, range_checker_num_bins, bitwise_lookup, range_tuple_checker,       \
        range_tuple_checker_sizes, timestamp_max_bits, stream                             \
    )
    switch (kind) {
    case 0: G2_REFERENCE_DISPATCH(_add_sub_tracegen_g2_reference);
    case 1: G2_REFERENCE_DISPATCH(_bitwise_logic_tracegen_g2_reference);
    case 2: G2_REFERENCE_DISPATCH(_rv64_less_than_tracegen_g2_reference);
    case 3: G2_REFERENCE_DISPATCH(_rv64_shift_logical_tracegen_g2_reference);
    case 4: G2_REFERENCE_DISPATCH(_rv64_shift_right_arithmetic_tracegen_g2_reference);
    case 5: G2_REFERENCE_DISPATCH(_rv64_add_sub_w_tracegen_g2_reference);
    case 6: G2_REFERENCE_DISPATCH(_rv64_shift_w_logical_tracegen_g2_reference);
    case 7: G2_REFERENCE_DISPATCH(_rv64_shift_w_right_arithmetic_tracegen_g2_reference);
    case 8: G2_REFERENCE_DISPATCH(_rv64_load_byte_tracegen_g2_reference);
    case 9: G2_REFERENCE_DISPATCH(_rv64_load_sign_extend_byte_tracegen_g2_reference);
    case 10: G2_REFERENCE_DISPATCH(_beq_tracegen_g2_reference);
    case 11: G2_REFERENCE_DISPATCH(_blt_tracegen_g2_reference);
    case 12: G2_REFERENCE_DISPATCH(_jal_lui_tracegen_g2_reference);
    case 13: G2_REFERENCE_DISPATCH(_jalr_tracegen_g2_reference);
    case 14: G2_REFERENCE_DISPATCH(_auipc_tracegen_g2_reference);
    case 15: G2_REFERENCE_DISPATCH(_mul_tracegen_g2_reference);
    case 16: G2_REFERENCE_DISPATCH(_mulh_tracegen_g2_reference);
    case 17: G2_REFERENCE_DISPATCH(_rv64_mul_w_tracegen_g2_reference);
    case 18: G2_REFERENCE_DISPATCH(_rv64_div_rem_tracegen_g2_reference);
    case 19: G2_REFERENCE_DISPATCH(_rv64_div_rem_w_tracegen_g2_reference);
    case 20: G2_REFERENCE_DISPATCH(_rv64_load_halfword_tracegen_g2_reference);
    case 21: G2_REFERENCE_DISPATCH(_rv64_load_word_tracegen_g2_reference);
    case 22: G2_REFERENCE_DISPATCH(_rv64_load_doubleword_tracegen_g2_reference);
    case 23: G2_REFERENCE_DISPATCH(_rv64_store_byte_tracegen_g2_reference);
    case 24: G2_REFERENCE_DISPATCH(_rv64_store_halfword_tracegen_g2_reference);
    case 25: G2_REFERENCE_DISPATCH(_rv64_store_word_tracegen_g2_reference);
    case 26: G2_REFERENCE_DISPATCH(_rv64_store_doubleword_tracegen_g2_reference);
    case 27: G2_REFERENCE_DISPATCH(_rv64_load_sign_extend_halfword_tracegen_g2_reference);
    case 28: G2_REFERENCE_DISPATCH(_rv64_load_sign_extend_word_tracegen_g2_reference);
    case 29: G2_REFERENCE_DISPATCH(_addi_tracegen_g2_common_reference);
    case 30: G2_REFERENCE_DISPATCH(_hintstore_tracegen_g2_reference);
    default: return int(cudaErrorInvalidValue);
    }
#undef G2_REFERENCE_DISPATCH
}

extern "C" int _rvr_g2_predecode(
    DeviceBufferConstView<uint8_t> d_wire,
    size_t logical_wire_bytes,
    size_t run_count,
    size_t instruction_count,
    uint8_t const *d_expected_fingerprint,
    G2BlockEntryV1 const *d_blocks,
    size_t block_count,
    RvrOperandEntry const *d_operands,
    size_t operand_count,
    uint32_t pc_base,
    DeviceBufferConstView<uint8_t> d_initial_memory_bytes,
    uint32_t initial_timestamp,
    G2ExpectedKindV1 const *d_expected_kinds,
    size_t expected_kind_count,
    G2ExpectedOpaqueV1 const *d_expected_opaque,
    size_t expected_opaque_count,
    uint32_t *d_program_frequencies,
    size_t frequency_count,
    size_t total_record_count,
    DeviceRawBufferConstView d_prepared_output,
    DeviceRawBufferConstView d_row_instruction_output,
    DeviceRawBufferConstView d_timestamp_offset_output,
    DeviceRawBufferConstView d_timeline_output,
    DeviceRawBufferConstView d_opaque_residual_output,
    uint32_t *d_opaque_residual_count,
    DeltaAirOutputDesc const *d_outputs,
    size_t num_airs,
    DeviceRawBufferConstView d_touched_output,
    uint32_t *d_touched_count,
    DeviceRawBufferConstView d_dirty_pages,
    uint32_t *d_opaque_prev_timestamps,
    uint64_t *d_opaque_prev_values,
    uint32_t *d_error,
    cudaStream_t stream
) {
    size_t delta_count = total_record_count;
    size_t opaque_capacity =
        d_opaque_residual_output.size / sizeof(DeltaMemoryLogEntry);
    size_t touched_capacity = d_touched_output.size / sizeof(G2TouchedMemoryRecord);
    size_t dirty_page_words = d_dirty_pages.size / sizeof(uint64_t);
    bool timestamp_budget_overflow = delta_count > UINT32_MAX / 4;
    size_t timestamp_budget = timestamp_budget_overflow ? 0 : delta_count * 4;
    if (!timestamp_budget_overflow) {
        timestamp_budget_overflow = opaque_capacity > UINT32_MAX - timestamp_budget;
        if (!timestamp_budget_overflow) timestamp_budget += opaque_capacity;
    }
    if (!timestamp_budget_overflow) {
        timestamp_budget_overflow = instruction_count > UINT32_MAX - timestamp_budget;
        if (!timestamp_budget_overflow) timestamp_budget += instruction_count;
    }
    if (d_wire.ptr == nullptr || d_wire.size < 128 ||
        d_expected_fingerprint == nullptr || d_blocks == nullptr || block_count == 0 ||
        d_operands == nullptr || operand_count == 0 ||
        d_initial_memory_bytes.ptr == nullptr ||
        d_initial_memory_bytes.size % sizeof(DeviceInitialMemory) != 0 ||
        d_dirty_pages.size % sizeof(uint64_t) != 0 ||
        (d_dirty_pages.size != 0 && d_dirty_pages.ptr == 0) ||
        d_expected_kinds == nullptr || expected_kind_count == 0 ||
        (expected_opaque_count != 0 && d_expected_opaque == nullptr) ||
        d_program_frequencies == nullptr || frequency_count == 0 ||
        d_prepared_output.ptr == 0 ||
        d_prepared_output.size != instruction_count * sizeof(G2PreparedInstruction) ||
        (delta_count != 0 && d_row_instruction_output.ptr == 0) ||
        d_row_instruction_output.size != delta_count * sizeof(uint32_t) ||
        d_timestamp_offset_output.ptr == 0 ||
        d_timestamp_offset_output.size != (instruction_count + 1) * sizeof(uint32_t) ||
        d_timeline_output.ptr == 0 ||
        d_timeline_output.size != timestamp_budget * sizeof(G2TimelineEvent) ||
        (d_opaque_residual_output.size != 0 && d_opaque_residual_output.ptr == 0) ||
        (opaque_capacity != 0 &&
         (d_opaque_prev_timestamps == nullptr || d_opaque_prev_values == nullptr)) ||
        d_outputs == nullptr || num_airs == 0 ||
        d_touched_output.size % sizeof(G2TouchedMemoryRecord) != 0 ||
        (touched_capacity != 0 && d_touched_output.ptr == 0) ||
        d_touched_count == nullptr || d_opaque_residual_count == nullptr ||
        d_error == nullptr || run_count == 0 ||
        run_count > CUDA_GRID_X_MAX || run_count > UINT32_MAX || run_count == SIZE_MAX ||
        instruction_count == 0 || instruction_count > UINT32_MAX ||
        delta_count > UINT32_MAX || opaque_capacity > UINT32_MAX ||
        timestamp_budget_overflow) {
        std::fprintf(stderr, "OPENVM_RVR_G2_CUDA_ERROR call=argument_validation code=%d\n", int(cudaErrorInvalidValue));
        return int(cudaErrorInvalidValue);
    }
    size_t instruction_blocks = 1 + (instruction_count - 1) / 256;
    size_t count_stride = instruction_blocks + 1;
    size_t chunk_capacity =
        (timestamp_budget + G2_REGISTER_REPLAY_CHUNK - 1) / G2_REGISTER_REPLAY_CHUNK;
    if (instruction_blocks > CUDA_GRID_X_MAX || instruction_blocks == SIZE_MAX ||
        count_stride > SIZE_MAX / 31 ||
        instruction_count > SIZE_MAX / sizeof(G2PreparedInstruction) ||
        timestamp_budget > SIZE_MAX / sizeof(G2TimelineEvent) ||
        chunk_capacity > SIZE_MAX / G2_REGISTER_COUNT ||
        chunk_capacity * G2_REGISTER_COUNT > SIZE_MAX / sizeof(G2RegisterSummary)) {
        std::fprintf(stderr, "OPENVM_RVR_G2_CUDA_ERROR call=launch_dimensions code=%d\n", int(cudaErrorInvalidValue));
        return int(cudaErrorInvalidValue);
    }
    size_t lane_count_entries = 31 * count_stride;
    size_t instruction_scan_entries = instruction_count + 1;
    uint32_t *d_run_lengths = nullptr, *d_run_offsets = nullptr, *d_program_slots = nullptr;
    uint32_t *d_row_counts = nullptr, *d_row_offsets = nullptr;
    uint32_t *d_v0_counts = nullptr, *d_v0_offsets = nullptr;
    uint32_t *d_v1_counts = nullptr, *d_v1_offsets = nullptr;
    uint32_t *d_timestamp_counts = nullptr;
    uint32_t *d_timestamp_offsets =
        reinterpret_cast<uint32_t *>(d_timestamp_offset_output.ptr);
    uint32_t *d_output_counts = nullptr, *d_output_offsets = nullptr;
    uint32_t *d_residual_counts = nullptr, *d_residual_offsets = nullptr;
    uint32_t *d_opaque_counts = nullptr, *d_opaque_offsets = nullptr;
    uint32_t *d_opaque_markers = nullptr, *d_opaque_occurrences = nullptr;
    uint32_t *d_instruction_indices = nullptr, *d_hint_indices = nullptr;
    uint32_t *d_hint_count = nullptr, *d_kind_counts = nullptr;
    uint32_t *d_hint_row_offsets = nullptr;
    uint8_t *d_hint_flags = nullptr;
    int8_t *d_kind_by_air = nullptr;
    G2PreparedInstruction *d_prepared =
        reinterpret_cast<G2PreparedInstruction *>(d_prepared_output.ptr);
    uint32_t *d_row_instructions =
        reinterpret_cast<uint32_t *>(d_row_instruction_output.ptr);
    G2TimelineEvent *d_timeline =
        reinterpret_cast<G2TimelineEvent *>(d_timeline_output.ptr);
    G2RegisterSummary *d_register_summaries = nullptr;
    G2RegisterState *d_register_incoming = nullptr, *d_register_final = nullptr;
    uint32_t *d_register_touched_count = nullptr;
    uint32_t *d_memory_indices = nullptr, *d_memory_selected_a = nullptr;
    uint32_t *d_memory_selected_b = nullptr, *d_memory_group_starts = nullptr;
    uint32_t *d_memory_event_count = nullptr, *d_memory_group_count = nullptr;
    uint8_t *d_memory_flags = nullptr, *d_memory_group_flags = nullptr;
    uint64_t *d_memory_keys_a = nullptr, *d_memory_keys_b = nullptr;
    G2MemoryTransform *d_memory_transforms_a = nullptr;
    G2MemoryTransform *d_memory_transforms_b = nullptr;
    void *d_memory_temp = nullptr;
    void *d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0, lane_scan_temp_bytes = 0;
    size_t instruction_scan_temp_bytes = 0, select_temp_bytes = 0;
    size_t memory_count_temp_bytes = 0, memory_select_temp_bytes = 0;
    size_t memory_sort_temp_bytes = 0;
    size_t memory_group_select_temp_bytes = 0, memory_scan_temp_bytes = 0;
    size_t register_entries = chunk_capacity * G2_REGISTER_COUNT;
    uint32_t memory_event_count = 0;
    int result = 0;

#define G2_TRY(expr)                                                                               \
    do {                                                                                           \
        cudaError_t _error = (expr);                                                               \
        if (_error != cudaSuccess) {                                                               \
            std::fprintf(                                                                          \
                stderr,                                                                            \
                "OPENVM_RVR_G2_CUDA_ERROR call=%s code=%d\n",                                    \
                #expr,                                                                             \
                int(_error)                                                                        \
            );                                                                                     \
            result = int(_error);                                                                  \
            goto cleanup;                                                                          \
        }                                                                                          \
    } while (0)

    G2_TRY(cudaMallocAsync(
        reinterpret_cast<void **>(&d_run_lengths), (run_count + 1) * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMallocAsync(
        reinterpret_cast<void **>(&d_run_offsets), (run_count + 1) * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMallocAsync(
        reinterpret_cast<void **>(&d_program_slots), instruction_count * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMallocAsync(
        reinterpret_cast<void **>(&d_row_counts), lane_count_entries * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMallocAsync(
        reinterpret_cast<void **>(&d_row_offsets), lane_count_entries * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMallocAsync(
        reinterpret_cast<void **>(&d_v0_counts), lane_count_entries * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMallocAsync(
        reinterpret_cast<void **>(&d_v0_offsets), lane_count_entries * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMallocAsync(
        reinterpret_cast<void **>(&d_v1_counts), lane_count_entries * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMallocAsync(
        reinterpret_cast<void **>(&d_v1_offsets), lane_count_entries * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMallocAsync(reinterpret_cast<void **>(&d_kind_by_air), 256, stream));
    G2_TRY(cudaMallocAsync(
        reinterpret_cast<void **>(&d_register_summaries),
        register_entries * sizeof(G2RegisterSummary),
        stream
    ));
    G2_TRY(cudaMallocAsync(
        reinterpret_cast<void **>(&d_register_incoming),
        register_entries * sizeof(G2RegisterState),
        stream
    ));
    G2_TRY(cudaMallocAsync(
        reinterpret_cast<void **>(&d_register_final),
        G2_REGISTER_COUNT * sizeof(G2RegisterState),
        stream
    ));
    G2_TRY(cudaMallocAsync(
        reinterpret_cast<void **>(&d_memory_indices), timestamp_budget * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMallocAsync(
        reinterpret_cast<void **>(&d_memory_flags), timestamp_budget, stream
    ));
#define G2_ALLOC_U32(ptr, count)                                                                  \
    G2_TRY(cudaMallocAsync(                                                                       \
        reinterpret_cast<void **>(&(ptr)), (count) * sizeof(uint32_t), stream                    \
    ))
    G2_ALLOC_U32(d_timestamp_counts, instruction_scan_entries);
    G2_ALLOC_U32(d_output_counts, instruction_scan_entries);
    G2_ALLOC_U32(d_output_offsets, instruction_scan_entries);
    G2_ALLOC_U32(d_residual_counts, instruction_scan_entries);
    G2_ALLOC_U32(d_residual_offsets, instruction_scan_entries);
    G2_ALLOC_U32(d_opaque_counts, instruction_scan_entries);
    G2_ALLOC_U32(d_opaque_offsets, instruction_scan_entries);
    G2_ALLOC_U32(d_opaque_markers, instruction_scan_entries);
    G2_ALLOC_U32(d_opaque_occurrences, instruction_scan_entries);
    G2_ALLOC_U32(d_instruction_indices, instruction_count);
    G2_ALLOC_U32(d_hint_indices, instruction_count);
    G2_ALLOC_U32(d_hint_count, 1);
    G2_ALLOC_U32(d_kind_counts, 31);
    G2_ALLOC_U32(d_hint_row_offsets, instruction_count);
    G2_ALLOC_U32(d_register_touched_count, 1);
    G2_ALLOC_U32(d_memory_event_count, 1);
    G2_ALLOC_U32(d_memory_group_count, 1);
    G2_TRY(cudaMallocAsync(
        reinterpret_cast<void **>(&d_hint_flags), instruction_count, stream
    ));
#undef G2_ALLOC_U32
    G2_TRY(cudaMemsetAsync(d_run_lengths, 0, (run_count + 1) * sizeof(uint32_t), stream));
    G2_TRY(cudaMemsetAsync(
        d_row_counts, 0, lane_count_entries * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMemsetAsync(
        d_v0_counts, 0, lane_count_entries * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMemsetAsync(
        d_v1_counts, 0, lane_count_entries * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMemsetAsync(
        d_timestamp_counts, 0, instruction_scan_entries * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMemsetAsync(
        d_output_counts, 0, instruction_scan_entries * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMemsetAsync(
        d_residual_counts, 0, instruction_scan_entries * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMemsetAsync(
        d_opaque_counts, 0, instruction_scan_entries * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMemsetAsync(
        d_opaque_markers, 0, instruction_scan_entries * sizeof(uint32_t), stream
    ));
    G2_TRY(cudaMemsetAsync(d_hint_flags, 0, instruction_count, stream));
    G2_TRY(cudaMemsetAsync(d_hint_count, 0, sizeof(uint32_t), stream));
    G2_TRY(cudaMemsetAsync(d_kind_counts, 0, 31 * sizeof(uint32_t), stream));
    G2_TRY(cudaMemsetAsync(d_hint_row_offsets, 0xff, instruction_count * sizeof(uint32_t), stream));
    if (delta_count != 0) {
        G2_TRY(cudaMemsetAsync(
            d_row_instructions, 0xff, delta_count * sizeof(uint32_t), stream
        ));
    }
    G2_TRY(cudaMemsetAsync(
        d_timeline, 0xff, timestamp_budget * sizeof(G2TimelineEvent), stream
    ));
    G2_TRY(cudaMemsetAsync(
        d_register_summaries, 0, register_entries * sizeof(G2RegisterSummary), stream
    ));
    G2_TRY(cudaMemsetAsync(d_register_touched_count, 0, sizeof(uint32_t), stream));
    G2_TRY(cudaMemsetAsync(d_memory_event_count, 0, sizeof(uint32_t), stream));
    G2_TRY(cudaMemsetAsync(d_memory_group_count, 0, sizeof(uint32_t), stream));
    G2_TRY(cudaMemsetAsync(d_touched_count, 0, sizeof(uint32_t), stream));
    if (opaque_capacity != 0) {
        G2_TRY(cudaMemsetAsync(
            d_opaque_prev_timestamps, 0, opaque_capacity * sizeof(uint32_t), stream
        ));
        G2_TRY(cudaMemsetAsync(
            d_opaque_prev_values, 0, opaque_capacity * sizeof(uint64_t), stream
        ));
    }
    g2_build_kind_map<<<1, 1, 0, stream>>>(
        d_expected_kinds, expected_kind_count, d_kind_by_air, d_error
    );
    G2_TRY(cudaGetLastError());
    // Validate every descriptor and device-visible schema binding before any
    // parallel kernel dereferences a wire lane.  The host finalizer already
    // enforces this contract, but the device decoder remains independently
    // fail-closed for malformed or stale transport bytes.
    g2_predecode<<<1, 1, 0, stream>>>(
        d_wire.ptr,
        d_wire.size,
        logical_wire_bytes,
        d_expected_fingerprint,
        d_blocks,
        block_count,
        d_operands,
        operand_count,
        d_prepared,
        instruction_count,
        true,
        pc_base,
        reinterpret_cast<DeviceInitialMemory const *>(d_initial_memory_bytes.ptr),
        d_initial_memory_bytes.size / sizeof(DeviceInitialMemory),
        initial_timestamp,
        d_expected_kinds,
        expected_kind_count,
        d_expected_opaque,
        expected_opaque_count,
        d_program_frequencies,
        frequency_count,
        nullptr,
        delta_count,
        nullptr,
        nullptr,
        reinterpret_cast<DeltaMemoryLogEntry *>(d_opaque_residual_output.ptr),
        d_opaque_residual_output.size / sizeof(DeltaMemoryLogEntry),
        d_opaque_residual_count,
        d_error
    );
    G2_TRY(cudaGetLastError());
    {
        dim3 block(256);
        dim3 grid((run_count + block.x - 1) / block.x);
        g2_run_lengths<<<grid, block, 0, stream>>>(
            d_wire.ptr,
            d_wire.size,
            run_count,
            d_blocks,
            block_count,
            d_run_lengths,
            d_error
        );
        G2_TRY(cudaGetLastError());
    }
    G2_TRY(cub::DeviceScan::ExclusiveSum(
        nullptr,
        scan_temp_bytes,
        d_run_lengths,
        d_run_offsets,
        run_count + 1,
        stream
    ));
    G2_TRY(cub::DeviceScan::ExclusiveSum(
        nullptr,
        lane_scan_temp_bytes,
        d_v0_counts,
        d_v0_offsets,
        count_stride,
        stream
    ));
    G2_TRY(cub::DeviceScan::ExclusiveSum(
        nullptr,
        instruction_scan_temp_bytes,
        d_timestamp_counts,
        d_timestamp_offsets,
        instruction_scan_entries,
        stream
    ));
    G2_TRY(cub::DeviceSelect::Flagged(
        nullptr,
        select_temp_bytes,
        d_instruction_indices,
        d_hint_flags,
        d_hint_indices,
        d_hint_count,
        instruction_count,
        stream
    ));
    if (lane_scan_temp_bytes > scan_temp_bytes) scan_temp_bytes = lane_scan_temp_bytes;
    if (instruction_scan_temp_bytes > scan_temp_bytes)
        scan_temp_bytes = instruction_scan_temp_bytes;
    if (select_temp_bytes > scan_temp_bytes) scan_temp_bytes = select_temp_bytes;
    G2_TRY(cudaMallocAsync(&d_scan_temp, scan_temp_bytes, stream));
    G2_TRY(cub::DeviceScan::ExclusiveSum(
        d_scan_temp,
        scan_temp_bytes,
        d_run_lengths,
        d_run_offsets,
        run_count + 1,
        stream
    ));
    {
        dim3 block(256);
        dim3 grid((run_count + block.x - 1) / block.x);
        g2_expand_runs<<<grid, block, 0, stream>>>(
            d_wire.ptr,
            run_count,
            d_blocks,
            block_count,
            d_run_offsets,
            instruction_count,
            d_operands,
            operand_count,
            d_program_slots,
            d_program_frequencies,
            frequency_count,
            d_error
        );
        G2_TRY(cudaGetLastError());
    }
    {
        dim3 block(256);
        dim3 grid(instruction_blocks);
        g2_count_lane_blocks<<<grid, block, 0, stream>>>(
            d_program_slots,
            instruction_count,
            d_operands,
            operand_count,
            d_kind_by_air,
            d_row_counts,
            d_v0_counts,
            d_v1_counts,
            count_stride,
            d_error
        );
        G2_TRY(cudaGetLastError());
    }
    for (uint32_t kind = 0; kind < 31; ++kind) {
        size_t offset = size_t(kind) * count_stride;
        G2_TRY(cub::DeviceScan::ExclusiveSum(
            d_scan_temp,
            scan_temp_bytes,
            d_row_counts + offset,
            d_row_offsets + offset,
            count_stride,
            stream
        ));
        G2_TRY(cub::DeviceScan::ExclusiveSum(
            d_scan_temp,
            scan_temp_bytes,
            d_v0_counts + offset,
            d_v0_offsets + offset,
            count_stride,
            stream
        ));
        G2_TRY(cub::DeviceScan::ExclusiveSum(
            d_scan_temp,
            scan_temp_bytes,
            d_v1_counts + offset,
            d_v1_offsets + offset,
            count_stride,
            stream
        ));
    }
    {
        dim3 block(256);
        dim3 grid(instruction_blocks);
        g2_prepare_instructions<<<grid, block, 0, stream>>>(
            d_wire.ptr,
            d_program_slots,
            instruction_count,
            d_operands,
            operand_count,
            d_kind_by_air,
            d_row_offsets,
            d_v0_offsets,
            d_v1_offsets,
            count_stride,
            d_prepared,
            d_error
        );
        G2_TRY(cudaGetLastError());
    }
    {
        dim3 block(256);
        dim3 grid(instruction_blocks);
        g2_classify_instructions<<<grid, block, 0, stream>>>(
            d_wire.ptr,
            d_prepared,
            instruction_count,
            d_operands,
            operand_count,
            d_kind_by_air,
            d_timestamp_counts,
            d_output_counts,
            d_residual_counts,
            d_opaque_counts,
            d_opaque_markers,
            d_hint_flags,
            d_instruction_indices,
            d_error
        );
        G2_TRY(cudaGetLastError());
    }
#define G2_SCAN(input, output)                                                                    \
    G2_TRY(cub::DeviceScan::ExclusiveSum(                                                         \
        d_scan_temp,                                                                              \
        scan_temp_bytes,                                                                          \
        (input),                                                                                   \
        (output),                                                                                  \
        instruction_scan_entries,                                                                 \
        stream                                                                                     \
    ))
    G2_SCAN(d_opaque_markers, d_opaque_occurrences);
    {
        dim3 block(256);
        dim3 grid(instruction_blocks);
        g2_fill_opaque_shapes<<<grid, block, 0, stream>>>(
            d_wire.ptr,
            d_opaque_occurrences,
            instruction_count,
            d_timestamp_counts,
            d_residual_counts,
            d_opaque_counts,
            d_error
        );
        G2_TRY(cudaGetLastError());
    }
    G2_SCAN(d_timestamp_counts, d_timestamp_offsets);
    G2_SCAN(d_residual_counts, d_residual_offsets);
    G2_TRY(cub::DeviceSelect::Flagged(
        d_scan_temp,
        scan_temp_bytes,
        d_instruction_indices,
        d_hint_flags,
        d_hint_indices,
        d_hint_count,
        instruction_count,
        stream
    ));
    g2_plan_hints<<<1, 1, 0, stream>>>(
        d_wire.ptr,
        d_prepared,
        d_operands,
        operand_count,
        d_hint_indices,
        d_hint_count,
        d_timestamp_offsets,
        d_residual_offsets,
        initial_timestamp,
        d_timestamp_counts,
        d_output_counts,
        d_residual_counts,
        d_hint_row_offsets,
        d_error
    );
    G2_TRY(cudaGetLastError());
    G2_SCAN(d_timestamp_counts, d_timestamp_offsets);
    G2_SCAN(d_output_counts, d_output_offsets);
    G2_SCAN(d_residual_counts, d_residual_offsets);
    G2_SCAN(d_opaque_counts, d_opaque_offsets);
#undef G2_SCAN
    {
        dim3 block(256);
        dim3 grid(instruction_blocks);
        g2_emit_parallel<<<grid, block, 0, stream>>>(
            d_wire.ptr,
            d_prepared,
            instruction_count,
            d_operands,
            operand_count,
            d_kind_by_air,
            d_timestamp_offsets,
            d_output_offsets,
            d_residual_offsets,
            d_opaque_offsets,
            d_hint_row_offsets,
            initial_timestamp,
            pc_base,
            delta_count,
            reinterpret_cast<DeltaMemoryLogEntry *>(d_opaque_residual_output.ptr),
            d_opaque_residual_output.size / sizeof(DeltaMemoryLogEntry),
            d_outputs,
            num_airs,
            d_row_instructions,
            d_timeline,
            timestamp_budget,
            d_kind_counts,
            d_error
        );
        G2_TRY(cudaGetLastError());
    }
    g2_validate_parallel_output<<<1, 1, 0, stream>>>(
        d_wire.ptr,
        instruction_count,
        d_timestamp_offsets,
        d_output_offsets,
        d_residual_offsets,
        d_opaque_offsets,
        d_opaque_occurrences,
        d_row_offsets,
        d_v0_offsets,
        d_v1_offsets,
        count_stride,
        d_expected_kinds,
        expected_kind_count,
        d_kind_counts,
        delta_count,
        d_opaque_residual_output.size / sizeof(DeltaMemoryLogEntry),
        d_opaque_residual_count,
        d_error
    );
    G2_TRY(cudaGetLastError());
    if (delta_count != 0) {
        dim3 block(256);
        dim3 grid((delta_count + block.x - 1) / block.x);
        g2_validate_trace_rows<<<grid, block, 0, stream>>>(
            d_prepared,
            instruction_count,
            d_operands,
            operand_count,
            d_row_instructions,
            d_outputs,
            num_airs,
            delta_count,
            d_error
        );
        G2_TRY(cudaGetLastError());
    }

    if (chunk_capacity != 0) {
        g2_summarize_register_chunks<<<chunk_capacity, 1, 0, stream>>>(
            d_timeline,
            timestamp_budget,
            d_timestamp_offsets + instruction_count,
            initial_timestamp,
            d_register_summaries,
            chunk_capacity,
            d_error
        );
        G2_TRY(cudaGetLastError());
        g2_scan_register_chunks<<<1, G2_REGISTER_COUNT, 0, stream>>>(
            d_register_summaries,
            d_register_incoming,
            chunk_capacity,
            d_timestamp_offsets + instruction_count,
            reinterpret_cast<DeviceInitialMemory const *>(d_initial_memory_bytes.ptr),
            d_initial_memory_bytes.size / sizeof(DeviceInitialMemory),
            d_register_final,
            d_error
        );
        G2_TRY(cudaGetLastError());
        g2_fill_register_chunks<<<chunk_capacity, 1, 0, stream>>>(
            d_timeline,
            timestamp_budget,
            d_timestamp_offsets + instruction_count,
            initial_timestamp,
            instruction_count,
            d_register_incoming,
            chunk_capacity,
            d_opaque_prev_timestamps,
            d_opaque_prev_values,
            opaque_capacity,
            d_error
        );
        G2_TRY(cudaGetLastError());
    }
    g2_pack_register_touched<<<1, 1, 0, stream>>>(
        d_register_final,
        reinterpret_cast<G2TouchedMemoryRecord *>(d_touched_output.ptr),
        touched_capacity,
        d_register_touched_count,
        d_error
    );
    G2_TRY(cudaGetLastError());

    {
        dim3 block(256);
        dim3 grid((timestamp_budget + block.x - 1) / block.x);
        g2_mark_memory_events<<<grid, block, 0, stream>>>(
            d_timeline, d_memory_indices, d_memory_flags, timestamp_budget
        );
        G2_TRY(cudaGetLastError());
    }
    G2_TRY(cub::DeviceReduce::Sum(
        nullptr,
        memory_count_temp_bytes,
        d_memory_flags,
        d_memory_event_count,
        timestamp_budget,
        stream
    ));
    G2_TRY(cudaMallocAsync(&d_memory_temp, memory_count_temp_bytes, stream));
    G2_TRY(cub::DeviceReduce::Sum(
        d_memory_temp,
        memory_count_temp_bytes,
        d_memory_flags,
        d_memory_event_count,
        timestamp_budget,
        stream
    ));
    G2_TRY(cudaMemcpyAsync(
        &memory_event_count,
        d_memory_event_count,
        sizeof(memory_event_count),
        cudaMemcpyDeviceToHost,
        stream
    ));
    G2_TRY(cudaStreamSynchronize(stream));
    if (memory_event_count != 0) {
        G2_TRY(cudaMallocAsync(
            reinterpret_cast<void **>(&d_memory_selected_a),
            size_t(memory_event_count) * sizeof(uint32_t),
            stream
        ));
        G2_TRY(cudaMallocAsync(
            reinterpret_cast<void **>(&d_memory_selected_b),
            size_t(memory_event_count) * sizeof(uint32_t),
            stream
        ));
        G2_TRY(cudaMallocAsync(
            reinterpret_cast<void **>(&d_memory_keys_a),
            size_t(memory_event_count) * sizeof(uint64_t),
            stream
        ));
        G2_TRY(cudaMallocAsync(
            reinterpret_cast<void **>(&d_memory_keys_b),
            size_t(memory_event_count) * sizeof(uint64_t),
            stream
        ));
        G2_TRY(cudaMallocAsync(
            reinterpret_cast<void **>(&d_memory_group_starts),
            size_t(memory_event_count) * sizeof(uint32_t),
            stream
        ));
        G2_TRY(cudaMallocAsync(
            reinterpret_cast<void **>(&d_memory_group_flags), memory_event_count, stream
        ));
        G2_TRY(cudaMallocAsync(
            reinterpret_cast<void **>(&d_memory_transforms_a),
            size_t(memory_event_count) * sizeof(G2MemoryTransform),
            stream
        ));
        G2_TRY(cudaMallocAsync(
            reinterpret_cast<void **>(&d_memory_transforms_b),
            size_t(memory_event_count) * sizeof(G2MemoryTransform),
            stream
        ));
        G2_TRY(cub::DeviceSelect::Flagged(
            nullptr,
            memory_select_temp_bytes,
            d_memory_indices,
            d_memory_flags,
            d_memory_selected_a,
            d_memory_event_count,
            timestamp_budget,
            stream
        ));
        G2_TRY(cub::DeviceRadixSort::SortPairs(
            nullptr,
            memory_sort_temp_bytes,
            d_memory_keys_a,
            d_memory_keys_b,
            d_memory_selected_a,
            d_memory_selected_b,
            memory_event_count,
            0,
            64,
            stream
        ));
        G2_TRY(cub::DeviceSelect::Flagged(
            nullptr,
            memory_group_select_temp_bytes,
            d_memory_indices,
            d_memory_group_flags,
            d_memory_group_starts,
            d_memory_group_count,
            memory_event_count,
            stream
        ));
        G2_TRY(cub::DeviceScan::InclusiveScanByKey(
            nullptr,
            memory_scan_temp_bytes,
            d_memory_keys_b,
            d_memory_transforms_a,
            d_memory_transforms_b,
            G2ComposeMemoryTransform{},
            memory_event_count,
            G2EqualAddressKey{},
            stream
        ));
        size_t memory_temp_bytes = memory_select_temp_bytes;
        if (memory_sort_temp_bytes > memory_temp_bytes)
            memory_temp_bytes = memory_sort_temp_bytes;
        if (memory_group_select_temp_bytes > memory_temp_bytes)
            memory_temp_bytes = memory_group_select_temp_bytes;
        if (memory_scan_temp_bytes > memory_temp_bytes)
            memory_temp_bytes = memory_scan_temp_bytes;
        G2_TRY(cudaFreeAsync(d_memory_temp, stream));
        d_memory_temp = nullptr;
        G2_TRY(cudaMallocAsync(&d_memory_temp, memory_temp_bytes, stream));
        G2_TRY(cub::DeviceSelect::Flagged(
            d_memory_temp,
            memory_select_temp_bytes,
            d_memory_indices,
            d_memory_flags,
            d_memory_selected_a,
            d_memory_event_count,
            timestamp_budget,
            stream
        ));
        {
            dim3 block(256);
            dim3 grid((memory_event_count + block.x - 1) / block.x);
            g2_extract_memory_keys<<<grid, block, 0, stream>>>(
                d_timeline, d_memory_selected_a, d_memory_keys_a, memory_event_count
            );
            G2_TRY(cudaGetLastError());
        }
        G2_TRY(cub::DeviceRadixSort::SortPairs(
            d_memory_temp,
            memory_sort_temp_bytes,
            d_memory_keys_a,
            d_memory_keys_b,
            d_memory_selected_a,
            d_memory_selected_b,
            memory_event_count,
            0,
            64,
            stream
        ));
        {
            dim3 block(256);
            dim3 grid((memory_event_count + block.x - 1) / block.x);
            g2_mark_memory_group_starts<<<grid, block, 0, stream>>>(
                d_memory_keys_b, d_memory_indices, d_memory_group_flags, memory_event_count
            );
            G2_TRY(cudaGetLastError());
        }
        G2_TRY(cub::DeviceSelect::Flagged(
            d_memory_temp,
            memory_group_select_temp_bytes,
            d_memory_indices,
            d_memory_group_flags,
            d_memory_group_starts,
            d_memory_group_count,
            memory_event_count,
            stream
        ));
        {
            dim3 block(256);
            dim3 grid((memory_event_count + block.x - 1) / block.x);
            g2_prepare_memory_transforms<<<grid, block, 0, stream>>>(
                d_memory_selected_b,
                d_timeline,
                memory_event_count,
                instruction_count,
                d_memory_transforms_a,
                d_error
            );
            G2_TRY(cudaGetLastError());
        }
        G2_TRY(cub::DeviceScan::InclusiveScanByKey(
            d_memory_temp,
            memory_scan_temp_bytes,
            d_memory_keys_b,
            d_memory_transforms_a,
            d_memory_transforms_b,
            G2ComposeMemoryTransform{},
            memory_event_count,
            G2EqualAddressKey{},
            stream
        ));
        {
            dim3 block(256);
            dim3 grid((memory_event_count + block.x - 1) / block.x);
            g2_replay_memory_events<<<grid, block, 0, stream>>>(
                d_wire.ptr,
                d_memory_keys_b,
                d_memory_selected_b,
                d_timeline,
                memory_event_count,
                d_memory_transforms_b,
                d_prepared,
                instruction_count,
                initial_timestamp,
                reinterpret_cast<DeviceInitialMemory const *>(d_initial_memory_bytes.ptr),
                d_initial_memory_bytes.size / sizeof(DeviceInitialMemory),
                d_opaque_prev_timestamps,
                d_opaque_prev_values,
                opaque_capacity,
                d_error
            );
            G2_TRY(cudaGetLastError());
        }
        {
            dim3 block(256);
            dim3 grid((memory_event_count + block.x - 1) / block.x);
            g2_pack_memory_touched<<<grid, block, 0, stream>>>(
                d_memory_keys_b,
                d_memory_selected_b,
                d_timeline,
                memory_event_count,
                d_memory_transforms_b,
                d_memory_group_starts,
                d_memory_group_count,
                reinterpret_cast<DeviceInitialMemory const *>(d_initial_memory_bytes.ptr),
                d_initial_memory_bytes.size / sizeof(DeviceInitialMemory),
                reinterpret_cast<G2TouchedMemoryRecord *>(d_touched_output.ptr),
                touched_capacity,
                reinterpret_cast<uint64_t *>(d_dirty_pages.ptr),
                dirty_page_words,
                d_register_touched_count,
                initial_timestamp,
                d_error
            );
            G2_TRY(cudaGetLastError());
        }
    }
    g2_finish_touched_count<<<1, 1, 0, stream>>>(
        d_register_touched_count,
        d_memory_group_count,
        touched_capacity,
        d_touched_count,
        d_error
    );
    G2_TRY(cudaGetLastError());

cleanup:
    if (d_memory_temp) cudaFreeAsync(d_memory_temp, stream);
    if (d_memory_transforms_b) cudaFreeAsync(d_memory_transforms_b, stream);
    if (d_memory_transforms_a) cudaFreeAsync(d_memory_transforms_a, stream);
    if (d_memory_selected_b) cudaFreeAsync(d_memory_selected_b, stream);
    if (d_memory_selected_a) cudaFreeAsync(d_memory_selected_a, stream);
    if (d_memory_keys_b) cudaFreeAsync(d_memory_keys_b, stream);
    if (d_memory_keys_a) cudaFreeAsync(d_memory_keys_a, stream);
    if (d_memory_group_flags) cudaFreeAsync(d_memory_group_flags, stream);
    if (d_memory_flags) cudaFreeAsync(d_memory_flags, stream);
    if (d_memory_group_starts) cudaFreeAsync(d_memory_group_starts, stream);
    if (d_memory_indices) cudaFreeAsync(d_memory_indices, stream);
    if (d_memory_group_count) cudaFreeAsync(d_memory_group_count, stream);
    if (d_memory_event_count) cudaFreeAsync(d_memory_event_count, stream);
    if (d_register_touched_count) cudaFreeAsync(d_register_touched_count, stream);
    if (d_register_final) cudaFreeAsync(d_register_final, stream);
    if (d_register_incoming) cudaFreeAsync(d_register_incoming, stream);
    if (d_register_summaries) cudaFreeAsync(d_register_summaries, stream);
    if (d_scan_temp) cudaFreeAsync(d_scan_temp, stream);
    if (d_hint_flags) cudaFreeAsync(d_hint_flags, stream);
    if (d_kind_counts) cudaFreeAsync(d_kind_counts, stream);
    if (d_hint_row_offsets) cudaFreeAsync(d_hint_row_offsets, stream);
    if (d_hint_count) cudaFreeAsync(d_hint_count, stream);
    if (d_hint_indices) cudaFreeAsync(d_hint_indices, stream);
    if (d_instruction_indices) cudaFreeAsync(d_instruction_indices, stream);
    if (d_opaque_occurrences) cudaFreeAsync(d_opaque_occurrences, stream);
    if (d_opaque_markers) cudaFreeAsync(d_opaque_markers, stream);
    if (d_opaque_offsets) cudaFreeAsync(d_opaque_offsets, stream);
    if (d_opaque_counts) cudaFreeAsync(d_opaque_counts, stream);
    if (d_residual_offsets) cudaFreeAsync(d_residual_offsets, stream);
    if (d_residual_counts) cudaFreeAsync(d_residual_counts, stream);
    if (d_output_offsets) cudaFreeAsync(d_output_offsets, stream);
    if (d_output_counts) cudaFreeAsync(d_output_counts, stream);
    if (d_timestamp_counts) cudaFreeAsync(d_timestamp_counts, stream);
    if (d_kind_by_air) cudaFreeAsync(d_kind_by_air, stream);
    if (d_v1_offsets) cudaFreeAsync(d_v1_offsets, stream);
    if (d_v1_counts) cudaFreeAsync(d_v1_counts, stream);
    if (d_v0_offsets) cudaFreeAsync(d_v0_offsets, stream);
    if (d_v0_counts) cudaFreeAsync(d_v0_counts, stream);
    if (d_row_offsets) cudaFreeAsync(d_row_offsets, stream);
    if (d_row_counts) cudaFreeAsync(d_row_counts, stream);
    if (d_program_slots) cudaFreeAsync(d_program_slots, stream);
    if (d_run_offsets) cudaFreeAsync(d_run_offsets, stream);
    if (d_run_lengths) cudaFreeAsync(d_run_lengths, stream);
    return result;

#undef G2_TRY
}
