#include "fp.h"
#include "primitives/buffer_view.cuh"
#include "riscv/rvr_compact.cuh"

#include <cuda_runtime.h>
#include <stdint.h>

using namespace riscv;

namespace {

static constexpr uint16_t G2_FLAGS_COMMITTED_V1 = 0x000fu;
static constexpr uint16_t G2_RUN_BLOCK_ID = 0x0001u;
static constexpr uint16_t G2_ADDI_V0 = 0x013au;
static constexpr uint8_t G2_ADDI_PATTERN = 8u;
static constexpr uint8_t INVALID_AIR = UINT8_MAX;

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
static_assert(sizeof(TouchedMemoryRecord) == 28, "touched-memory record size drift");

__device__ __forceinline__ void fail(uint32_t *error, uint32_t code) {
    if (*error == 0) *error = code;
}

__device__ G2BlockEntryV1 const *find_block(
    G2BlockEntryV1 const *blocks, size_t count, uint32_t slot
) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        uint32_t candidate = blocks[mid].program_slot;
        if (candidate < slot)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < count && blocks[lo].program_slot == slot ? &blocks[lo] : nullptr;
}

__global__ void g2_addi_predecode(
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
    size_t expected_addi_count,
    uint32_t *program_frequencies,
    size_t frequency_count,
    RvrAlu3Compact *addi_output,
    size_t addi_output_count,
    TouchedMemoryRecord *touched_output,
    size_t touched_capacity,
    uint32_t *touched_count,
    uint32_t *error
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *touched_count = 0;
    if (wire_bytes < 128 || (wire_bytes & 127u) != 0 || block_count == 0 ||
        initial_memory_count <= 1 ||
        expected_addi_count != addi_output_count) {
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
    if (header.version != 1 || header.header_bytes != 128 || header.lane_count != 2 ||
        header.flags != G2_FLAGS_COMMITTED_V1 || header.residual_event_count != 0) {
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
    G2LaneDescV1 run = descs[0], addi = descs[1];
    auto valid_fixed = [&](G2LaneDescV1 const &desc, uint16_t kind, uint8_t width) {
        uint64_t end = desc.offset + desc.payload_bytes;
        return desc.kind == kind && desc.elem_width == width && desc.encoding == 0 &&
               desc.flags == 1 && desc.payload_bytes == uint64_t(desc.count) * width &&
               desc.offset >= 128 && (desc.offset & 127u) == 0 && desc.group_id == 0 &&
               desc.reserved == 0 && end >= desc.offset && end <= wire_bytes;
    };
    if (!valid_fixed(run, G2_RUN_BLOCK_ID, 4) || !valid_fixed(addi, G2_ADDI_V0, 8) ||
        run.count != header.run_count || addi.count != expected_addi_count ||
        (run.offset < addi.offset + addi.payload_bytes &&
         addi.offset < run.offset + run.payload_bytes)) {
        fail(error, 5);
        return;
    }

    DeviceInitialMemory registers_image = initial_memory[1];
    if (registers_image.reserved != 0 || registers_image.cell_size != 2 ||
        registers_image.base == 0 || registers_image.len < 32 * sizeof(uint64_t)) {
        fail(error, 6);
        return;
    }
    uint64_t registers[32];
    uint32_t timestamps[32];
    bool touched[32];
    for (size_t i = 0; i < 32; ++i) {
        registers[i] = *reinterpret_cast<uint64_t const *>(registers_image.base + i * 8);
        timestamps[i] = 0;
        touched[i] = false;
    }
    registers[0] = 0;

    uint32_t const *runs = reinterpret_cast<uint32_t const *>(wire + run.offset);
    uint64_t const *addi_values = reinterpret_cast<uint64_t const *>(wire + addi.offset);
    size_t addi_cursor = 0;
    size_t instruction_cursor = 0;
    uint32_t timestamp = initial_timestamp;
    for (size_t run_index = 0; run_index < run.count; ++run_index) {
        uint32_t run_slot = runs[run_index];
        G2BlockEntryV1 const *block = find_block(blocks, block_count, run_slot);
        if (block == nullptr || block->instruction_count == 0) {
            fail(error, 7);
            return;
        }
        for (uint32_t local = 0; local < block->instruction_count; ++local) {
            uint32_t slot = run_slot + local;
            if (slot < run_slot || slot >= operand_count) {
                fail(error, 8);
                return;
            }
            RvrOperandEntry const entry = operands[slot];
            if (entry.filtered_index >= frequency_count) {
                fail(error, 9);
                return;
            }
            ++program_frequencies[entry.filtered_index];
            ++instruction_cursor;
            if (entry.air_idx == INVALID_AIR) continue;
            if (entry.access_pattern != G2_ADDI_PATTERN || addi_cursor >= addi.count ||
                (entry.a & 7u) != 0 || (entry.b & 7u) != 0 || entry.a >= 32 * 8 ||
                entry.b >= 32 * 8) {
                fail(error, 10);
                return;
            }
            uint32_t rd = entry.a / 8, rs1 = entry.b / 8;
            uint64_t rs1_value = addi_values[addi_cursor];
            if (rs1_value != registers[rs1]) {
                fail(error, 11);
                return;
            }
            RvrAlu3Compact record{};
            record.from_pc = pc_base + slot * 4;
            record.from_timestamp = timestamp;
            record.reads_prev_timestamp[0] = timestamps[rs1];
            timestamps[rs1] = timestamp++;
            record.write_prev_timestamp = timestamps[rd];
            uint64_t previous = registers[rd];
            record.write_prev_data[0] = uint32_t(previous);
            record.write_prev_data[1] = uint32_t(previous >> 32);
            timestamps[rd] = timestamp++;
            touched[rs1] = true;
            touched[rd] = true;
            record.b[0] = uint32_t(rs1_value);
            record.b[1] = uint32_t(rs1_value >> 32);
            int64_t immediate = int64_t(int32_t(entry.c << 20) >> 20);
            record.c[0] = uint32_t(immediate);
            record.c[1] = uint32_t(uint64_t(immediate) >> 32);
            if (rd != 0) registers[rd] = rs1_value + uint64_t(immediate);
            addi_output[addi_cursor++] = record;
        }
    }
    if (instruction_cursor != header.instruction_count || addi_cursor != addi.count) {
        fail(error, 12);
        return;
    }
    size_t output_cursor = 0;
    for (uint32_t reg = 0; reg < 32; ++reg) {
        if (!touched[reg]) continue;
        if (output_cursor >= touched_capacity) {
            fail(error, 13);
            return;
        }
        uint64_t value = registers[reg];
        TouchedMemoryRecord record{};
        record.addr_space = 1;
        record.block_ptr = reg * 4;
        record.timestamp = timestamps[reg];
        record.values[0] = Fp(uint16_t(value)).asRaw();
        record.values[1] = Fp(uint16_t(value >> 16)).asRaw();
        record.values[2] = Fp(uint16_t(value >> 32)).asRaw();
        record.values[3] = Fp(uint16_t(value >> 48)).asRaw();
        touched_output[output_cursor++] = record;
    }
    *touched_count = uint32_t(output_cursor);
}

} // namespace

extern "C" int _rvr_g2_addi_predecode(
    DeviceBufferConstView<uint8_t> d_wire,
    uint8_t const *d_expected_fingerprint,
    G2BlockEntryV1 const *d_blocks,
    size_t block_count,
    RvrOperandEntry const *d_operands,
    size_t operand_count,
    uint32_t pc_base,
    DeviceBufferConstView<uint8_t> d_initial_memory_bytes,
    uint32_t initial_timestamp,
    size_t expected_addi_count,
    uint32_t *d_program_frequencies,
    size_t frequency_count,
    DeviceRawBufferConstView d_addi_output,
    DeviceRawBufferConstView d_touched_output,
    uint32_t *d_touched_count,
    uint32_t *d_error,
    cudaStream_t stream
) {
    if (d_addi_output.size % sizeof(RvrAlu3Compact) != 0 ||
        d_touched_output.size % sizeof(TouchedMemoryRecord) != 0 ||
        d_initial_memory_bytes.size % sizeof(DeviceInitialMemory) != 0) {
        return 1;
    }
    g2_addi_predecode<<<1, 1, 0, stream>>>(
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
        expected_addi_count,
        d_program_frequencies,
        frequency_count,
        reinterpret_cast<RvrAlu3Compact *>(d_addi_output.ptr),
        d_addi_output.size / sizeof(RvrAlu3Compact),
        reinterpret_cast<TouchedMemoryRecord *>(d_touched_output.ptr),
        d_touched_output.size / sizeof(TouchedMemoryRecord),
        d_touched_count,
        d_error
    );
    return CHECK_KERNEL();
}
