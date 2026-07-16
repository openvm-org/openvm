#include "fp.h"
#include "primitives/buffer_view.cuh"
#include "riscv/rvr_compact.cuh"

#include <cuda_runtime.h>
#include <stdint.h>

using namespace riscv;

namespace {

static constexpr uint16_t G2_FLAGS_COMMITTED_V1 = 0x000fu;
static constexpr uint16_t G2_RUN_BLOCK_ID = 0x0001u;
static constexpr uint16_t G2_RESIDUAL_CTRL = 0x0080u;
static constexpr uint16_t G2_RESIDUAL_TAG = 0x0081u;
static constexpr uint16_t G2_RESIDUAL_VALUE = 0x0082u;
static constexpr uint16_t G2_ADDI_V0 = 0x013au;
static constexpr uint32_t G2_REQUIRED = 1u;
static constexpr uint32_t G2_REQUIRED_ATOMIC = 3u;
static constexpr uint32_t G2_LOAD_STORE_GROUP = 1u;
static constexpr uint32_t G2_RESIDUAL_GROUP = 2u;
static constexpr uint8_t G2_ADDI_PATTERN = 8u;
static constexpr uint8_t G2_LOAD_PATTERN = 2u;
static constexpr uint8_t G2_STORE_PATTERN = 3u;
static constexpr uint8_t INVALID_AIR = UINT8_MAX;
static constexpr uint8_t EXPECT_NONE = 0u;
static constexpr uint8_t EXPECT_BEFORE_LOAD = 1u;
static constexpr uint8_t EXPECT_BEFORE_STORE = 2u;
static constexpr uint8_t EXPECT_AFTER_STORE = 3u;

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

__device__ __forceinline__ void fail(uint32_t *error, uint32_t code) {
    if (*error == 0) *error = code;
}

__device__ __forceinline__ bool load_store_kind(uint32_t kind) {
    return kind == 8 || kind == 9 || (kind >= 20 && kind <= 28);
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
        kind == G2_RESIDUAL_VALUE) {
        width = kind == G2_RESIDUAL_TAG ? 1 : 8;
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

__device__ G2ExpectedKindV1 const *find_expected(
    G2ExpectedKindV1 const *expected, size_t count, uint32_t air_idx
) {
    for (size_t i = 0; i < count; ++i) {
        if (expected[i].air_idx == air_idx) return &expected[i];
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

__device__ __forceinline__ uint32_t effective_address(
    uint32_t pointer, RvrOperandEntry const &entry
) {
    int32_t offset = int32_t(entry.c & 0xffffu);
    if (entry.flags & RVR_OPERAND_FLAG_LS_IMM_SIGN) offset = int32_t(int16_t(offset));
    return pointer + uint32_t(offset);
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

__global__ void g2_predecode(
    uint8_t const *wire,
    size_t wire_bytes,
    uint8_t const *expected_fingerprint,
    G2BlockEntryV1 const *blocks,
    size_t block_count,
    RvrOperandEntry const *operands,
    size_t operand_count,
    uint32_t pc_base,
    DeviceInitialMemory const *initial_memory,
    size_t initial_memory_count,
    uint32_t initial_timestamp,
    G2ExpectedKindV1 const *expected_kinds,
    size_t expected_kind_count,
    uint32_t *program_frequencies,
    size_t frequency_count,
    DeltaRecord *delta_output,
    size_t delta_count,
    uint64_t *expected_blocks,
    uint8_t *expected_modes,
    uint32_t *error
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (wire_bytes < 128 || (wire_bytes & 127u) != 0 || block_count == 0 ||
        initial_memory_count <= 1 || expected_kind_count == 0) {
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
    if (header.version != 1 || header.lane_count == 0 || header.lane_count > 58 ||
        header.header_bytes != 64 + 32 * header.lane_count ||
        header.header_bytes > wire_bytes || header.flags != G2_FLAGS_COMMITTED_V1) {
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
        uint8_t width;
        uint32_t flags, group;
        uint64_t end = desc.offset + desc.payload_bytes;
        if ((i != 0 && descs[i - 1].kind >= desc.kind) ||
            !lane_spec(desc.kind, width, flags, group) || desc.elem_width != width ||
            desc.encoding != 0 || desc.flags != flags || desc.group_id != group ||
            desc.reserved != 0 || desc.payload_bytes != uint64_t(desc.count) * width ||
            desc.offset < header.header_bytes || (desc.offset & 127u) != 0 ||
            end < desc.offset || end > wire_bytes) {
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
    G2LaneDescV1 const *run = find_lane(descs, header.lane_count, G2_RUN_BLOCK_ID);
    G2LaneDescV1 const *residual_ctrl = find_lane(descs, header.lane_count, G2_RESIDUAL_CTRL);
    G2LaneDescV1 const *residual_tag = find_lane(descs, header.lane_count, G2_RESIDUAL_TAG);
    G2LaneDescV1 const *residual_value = find_lane(descs, header.lane_count, G2_RESIDUAL_VALUE);
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
    for (uint32_t kind = 0; kind < 30; ++kind) {
        if (!load_store_kind(kind)) continue;
        G2LaneDescV1 const *v0 = find_lane(descs, header.lane_count, lane_v0(kind));
        G2LaneDescV1 const *v1 = find_lane(descs, header.lane_count, lane_v1(kind));
        if ((v0 == nullptr) != (v1 == nullptr) ||
            (v0 != nullptr && v0->count != v1->count)) {
            fail(error, 8);
            return;
        }
    }

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
    size_t kind_v0_cursors[30]{};
    size_t kind_v1_cursors[30]{};
    size_t kind_counts[30]{};
    size_t residual_cursor = 0;
    size_t instruction_cursor = 0;
    size_t output_cursor = 0;
    uint32_t timestamp = initial_timestamp;

    for (size_t run_index = 0; run_index < run->count; ++run_index) {
        uint32_t run_slot = lane_u32(wire, *run, run_index);
        G2BlockEntryV1 const *block = find_block(blocks, block_count, run_slot);
        if (block == nullptr || block->instruction_count == 0) {
            fail(error, 10);
            return;
        }
        for (uint32_t local = 0; local < block->instruction_count; ++local) {
            uint32_t slot = run_slot + local;
            if (slot < run_slot || slot >= operand_count) {
                fail(error, 11);
                return;
            }
            RvrOperandEntry const entry = operands[slot];
            if (entry.filtered_index >= frequency_count) {
                fail(error, 12);
                return;
            }
            ++program_frequencies[entry.filtered_index];
            ++instruction_cursor;
            if (entry.air_idx == INVALID_AIR) continue;
            G2ExpectedKindV1 const *expected =
                find_expected(expected_kinds, expected_kind_count, entry.air_idx);
            if (expected == nullptr || expected->kind >= 30 || output_cursor >= delta_count) {
                fail(error, 13);
                return;
            }
            uint32_t kind = expected->kind;
            DeltaRecord record{pc_base + slot * 4u, timestamp, 0, 0};
            uint64_t expected_block = 0;
            uint8_t expected_mode = EXPECT_NONE;
            if (kind == 29 && entry.access_pattern == G2_ADDI_PATTERN) {
                G2LaneDescV1 const *lane = find_lane(descs, header.lane_count, G2_ADDI_V0);
                size_t cursor = kind_v0_cursors[kind]++;
                if (lane == nullptr || cursor >= lane->count || (entry.a & 7u) != 0 ||
                    (entry.b & 7u) != 0 || entry.a >= 32 * 8 || entry.b >= 32 * 8) {
                    fail(error, 14);
                    return;
                }
                uint32_t rd = entry.a / 8, rs1 = entry.b / 8;
                record.v1 = lane_u64(wire, *lane, cursor);
                if (record.v1 != registers[rs1]) {
                    fail(error, 15);
                    return;
                }
                record.v2 = uint64_t(int64_t(int32_t(entry.c << 20) >> 20));
                if (rd != 0) registers[rd] = record.v1 + record.v2;
                timestamp += 2;
            } else if ((kind <= 7 || (kind >= 15 && kind <= 19)) &&
                       (entry.access_pattern == 0 || entry.access_pattern == 1)) {
                G2LaneDescV1 const *v0_lane = find_lane(descs, header.lane_count, lane_v0(kind));
                size_t v0_cursor = kind_v0_cursors[kind]++;
                if (v0_lane == nullptr || v0_cursor >= v0_lane->count ||
                    (entry.a & 7u) != 0 || (entry.b & 7u) != 0 ||
                    entry.a >= 32 * 8 || entry.b >= 32 * 8) {
                    fail(error, 28);
                    return;
                }
                uint32_t rd = entry.a / 8, rs1 = entry.b / 8;
                record.v1 = lane_u64(wire, *v0_lane, v0_cursor);
                if (record.v1 != registers[rs1]) {
                    fail(error, 29);
                    return;
                }
                if (entry.flags & RVR_OPERAND_FLAG_RS2_IMM) {
                    record.v2 = standard_immediate(entry);
                } else {
                    G2LaneDescV1 const *v1_lane =
                        find_lane(descs, header.lane_count, lane_v1(kind));
                    size_t v1_cursor = kind_v1_cursors[kind]++;
                    if (v1_lane == nullptr || v1_cursor >= v1_lane->count ||
                        (entry.c & 7u) != 0 || entry.c >= 32 * 8) {
                        fail(error, 30);
                        return;
                    }
                    record.v2 = lane_u64(wire, *v1_lane, v1_cursor);
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
                G2LaneDescV1 const *v0_lane = find_lane(descs, header.lane_count, lane_v0(kind));
                G2LaneDescV1 const *v1_lane = find_lane(descs, header.lane_count, lane_v1(kind));
                size_t v0_cursor = kind_v0_cursors[kind]++;
                size_t v1_cursor = kind_v1_cursors[kind]++;
                if (v0_lane == nullptr || v1_lane == nullptr || v0_cursor >= v0_lane->count ||
                    v1_cursor >= v1_lane->count || (entry.a & 7u) != 0 ||
                    (entry.b & 7u) != 0 || entry.a >= 32 * 8 || entry.b >= 32 * 8) {
                    fail(error, 33);
                    return;
                }
                record.v1 = lane_u64(wire, *v0_lane, v0_cursor);
                record.v2 = lane_u64(wire, *v1_lane, v1_cursor);
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
                G2LaneDescV1 const *v0_lane = find_lane(descs, header.lane_count, lane_v0(kind));
                size_t cursor = kind_v0_cursors[kind]++;
                if (v0_lane == nullptr || cursor >= v0_lane->count || (entry.a & 7u) != 0 ||
                    (entry.b & 7u) != 0 || entry.a >= 32 * 8 || entry.b >= 32 * 8) {
                    fail(error, 37);
                    return;
                }
                record.v1 = lane_u32(wire, *v0_lane, cursor);
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
                    G2LaneDescV1 const *v0 =
                        find_lane(descs, header.lane_count, lane_v0(kind));
                    G2LaneDescV1 const *v1 =
                        find_lane(descs, header.lane_count, lane_v1(kind));
                    size_t v0_cursor = kind_v0_cursors[kind]++;
                    size_t v1_cursor = kind_v1_cursors[kind]++;
                    if (v0 == nullptr || v1 == nullptr || v0_cursor >= v0->count ||
                        v1_cursor >= v1->count) {
                        fail(error, 19);
                        return;
                    }
                    pointer = lane_u32(wire, *v0, v0_cursor);
                    block_value = lane_u64(wire, *v1, v1_cursor);
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
    }
    if (instruction_cursor != header.instruction_count || output_cursor != delta_count ||
        residual_cursor != header.residual_event_count) {
        fail(error, 23);
        return;
    }
    for (size_t i = 0; i < expected_kind_count; ++i) {
        G2ExpectedKindV1 const &expected = expected_kinds[i];
        if (expected.kind >= 30 || kind_counts[expected.kind] != expected.count) {
            fail(error, 24);
            return;
        }
    }
    G2LaneDescV1 const *addi = find_lane(descs, header.lane_count, G2_ADDI_V0);
    if ((addi == nullptr ? 0 : addi->count) != kind_v0_cursors[29]) {
        fail(error, 25);
        return;
    }
    for (uint32_t kind = 0; kind < 30; ++kind) {
        G2LaneDescV1 const *v0 = find_lane(descs, header.lane_count, lane_v0(kind));
        G2LaneDescV1 const *v1 = find_lane(descs, header.lane_count, lane_v1(kind));
        if ((v0 == nullptr ? 0 : v0->count) != kind_v0_cursors[kind] ||
            (v1 == nullptr ? 0 : v1->count) != kind_v1_cursors[kind]) {
            fail(error, 26);
            return;
        }
    }
}

} // namespace

extern "C" int _rvr_g2_predecode(
    DeviceBufferConstView<uint8_t> d_wire,
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
    uint32_t *d_program_frequencies,
    size_t frequency_count,
    DeviceRawBufferConstView d_delta_output,
    DeviceRawBufferConstView d_expected_blocks,
    DeviceRawBufferConstView d_expected_modes,
    uint32_t *d_error,
    cudaStream_t stream
) {
    if (d_initial_memory_bytes.size % sizeof(DeviceInitialMemory) != 0 ||
        d_delta_output.size % sizeof(DeltaRecord) != 0 ||
        d_expected_blocks.size != d_delta_output.size / sizeof(DeltaRecord) * sizeof(uint64_t) ||
        d_expected_modes.size != d_delta_output.size / sizeof(DeltaRecord) ||
        expected_kind_count == 0 || d_expected_kinds == nullptr) {
        return int(cudaErrorInvalidValue);
    }
    g2_predecode<<<1, 1, 0, stream>>>(
        d_wire.ptr,
        d_wire.size,
        d_expected_fingerprint,
        d_blocks,
        block_count,
        d_operands,
        operand_count,
        pc_base,
        reinterpret_cast<DeviceInitialMemory const *>(d_initial_memory_bytes.ptr),
        d_initial_memory_bytes.size / sizeof(DeviceInitialMemory),
        initial_timestamp,
        d_expected_kinds,
        expected_kind_count,
        d_program_frequencies,
        frequency_count,
        reinterpret_cast<DeltaRecord *>(d_delta_output.ptr),
        d_delta_output.size / sizeof(DeltaRecord),
        reinterpret_cast<uint64_t *>(d_expected_blocks.ptr),
        reinterpret_cast<uint8_t *>(d_expected_modes.ptr),
        d_error
    );
    return CHECK_KERNEL();
}
