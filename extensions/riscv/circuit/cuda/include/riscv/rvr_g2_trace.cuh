#pragma once

#include "riscv/cores/load.cuh"
#include "riscv/cores/load_sign_extend.cuh"
#include "riscv/cores/store.cuh"
#include "riscv/rvr_compact.cuh"

#include <stddef.h>
#include <stdint.h>

namespace riscv {

// Persistent, device-resident portion of the G2 predecode. Unlike the old
// compact consumer buffers, these entries retain only program/lane indices;
// each owning trace kernel reconstructs its record in registers.
struct G2PreparedInstruction {
    uint32_t slot;
    int32_t kind;
    uint32_t row_index;
    uint32_t v0_index;
    uint32_t v1_index;
};
static_assert(sizeof(G2PreparedInstruction) == 20, "G2 prepared-instruction size drift");

struct G2TimelineEvent {
    uint64_t address_key;
    // The emitted value until replay; the authoritative predecessor value
    // afterward.
    uint64_t value;
    uint32_t instruction;
    uint32_t residual_index;
    uint32_t previous_timestamp;
    uint8_t action;
    uint8_t width;
    uint8_t byte_offset;
    uint8_t expected_mode;
};
static_assert(sizeof(G2TimelineEvent) == 32, "G2 timeline-event size drift");

// Host ABI passed by value to a per-kind trace launcher. All address fields
// are CUDA device pointers represented as u64 so the Rust/C layout is stable.
struct G2TraceSource {
    uint64_t prepared;
    uint64_t row_instructions;
    uint64_t timestamp_offsets;
    uint64_t timeline;
    uint64_t v0;
    uint64_t v1;
    uint64_t residual_values;
    uint64_t error;
    uint32_t instruction_count;
    uint32_t row_start;
    uint32_t row_count;
    uint32_t timeline_capacity;
    uint32_t operand_count;
    uint32_t v0_count;
    uint32_t v1_count;
    uint32_t residual_count;
    uint32_t pc_base;
    uint32_t initial_timestamp;
    uint32_t kind;
    uint8_t v0_width;
    uint8_t v1_width;
    uint16_t reserved;
};
static_assert(sizeof(G2TraceSource) == 112, "G2 trace-source ABI drift");

struct G2TraceRow {
    G2PreparedInstruction prepared;
    RvrOperandEntry entry;
    uint32_t instruction;
    uint32_t timeline_index;
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t local_index;
};

static __device__ __forceinline__ void g2_trace_fail(G2TraceSource const &source, uint32_t code) {
    uint32_t *error = reinterpret_cast<uint32_t *>(source.error);
    if (error != nullptr) atomicCAS(error, 0u, code);
}

static __device__ __forceinline__ bool g2_trace_row(
    G2TraceSource const &source,
    RvrOperandEntry const *operand_table,
    uint32_t row_index,
    G2TraceRow &row
) {
    if (source.reserved != 0 || source.prepared == 0 || source.row_instructions == 0 ||
        source.timestamp_offsets == 0 || source.timeline == 0 || source.error == 0 ||
        row_index >= source.row_count) {
        g2_trace_fail(source, 90);
        return false;
    }
    uint32_t const *row_instructions =
        reinterpret_cast<uint32_t const *>(source.row_instructions);
    uint32_t instruction = row_instructions[size_t(source.row_start) + row_index];
    if (instruction >= source.instruction_count) {
        g2_trace_fail(source, 91);
        return false;
    }
    G2PreparedInstruction const *prepared =
        reinterpret_cast<G2PreparedInstruction const *>(source.prepared);
    G2PreparedInstruction decoded = prepared[instruction];
    if (decoded.slot >= source.operand_count || operand_table == nullptr ||
        decoded.kind != int32_t(source.kind)) {
        g2_trace_fail(source, 92);
        return false;
    }
    uint32_t local_index = 0;
    if (source.kind == 30) {
        if (decoded.row_index > row_index || decoded.v0_index == 0 ||
            row_index - decoded.row_index >= decoded.v0_index) {
            g2_trace_fail(source, 92);
            return false;
        }
        local_index = row_index - decoded.row_index;
    } else if (decoded.row_index != row_index) {
        g2_trace_fail(source, 92);
        return false;
    }
    uint32_t const *timestamp_offsets =
        reinterpret_cast<uint32_t const *>(source.timestamp_offsets);
    uint32_t timeline_index = timestamp_offsets[instruction];
    if (timeline_index >= source.timeline_capacity ||
        source.initial_timestamp > UINT32_MAX - timeline_index ||
        decoded.slot > (UINT32_MAX - source.pc_base) / 4u) {
        g2_trace_fail(source, 93);
        return false;
    }
    row = {
        decoded,
        operand_table[decoded.slot],
        instruction,
        timeline_index,
        source.pc_base + decoded.slot * 4u,
        source.initial_timestamp + timeline_index,
        local_index,
    };
    return true;
}

static __device__ __forceinline__ bool g2_trace_event(
    G2TraceSource const &source,
    G2TraceRow const &row,
    uint32_t relative_timestamp,
    G2TimelineEvent &event
) {
    if (relative_timestamp >= source.timeline_capacity - row.timeline_index) {
        g2_trace_fail(source, 94);
        return false;
    }
    G2TimelineEvent const *timeline =
        reinterpret_cast<G2TimelineEvent const *>(source.timeline);
    event = timeline[row.timeline_index + relative_timestamp];
    if (event.address_key == UINT64_MAX || event.instruction != row.instruction) {
        g2_trace_fail(source, 94);
        return false;
    }
    return true;
}

static __device__ __forceinline__ bool g2_trace_residual_value(
    G2TraceSource const &source, uint32_t index, uint64_t &value
) {
    if (source.residual_values == 0 || index == UINT32_MAX || index >= source.residual_count) {
        g2_trace_fail(source, 104);
        return false;
    }
    uint32_t const *words = reinterpret_cast<uint32_t const *>(
        source.residual_values + size_t(index) * sizeof(uint64_t)
    );
    value = uint64_t(words[0]) | (uint64_t(words[1]) << 32);
    return true;
}

static __device__ __forceinline__ bool g2_trace_lane_value(
    G2TraceSource const &source, bool second, uint32_t index, uint64_t &value
) {
    uint64_t base = second ? source.v1 : source.v0;
    uint32_t count = second ? source.v1_count : source.v0_count;
    uint8_t width = second ? source.v1_width : source.v0_width;
    if (index == UINT32_MAX || base == 0 || index >= count || (width != 4 && width != 8)) {
        g2_trace_fail(source, second ? 96 : 95);
        return false;
    }
    uint8_t const *lane = reinterpret_cast<uint8_t const *>(base);
    if (width == 4) {
        value = reinterpret_cast<uint32_t const *>(lane)[index];
    } else {
        uint32_t const *words = reinterpret_cast<uint32_t const *>(lane + size_t(index) * 8u);
        value = uint64_t(words[0]) | (uint64_t(words[1]) << 32);
    }
    return true;
}

static __device__ __forceinline__ void g2_trace_store_u64(uint32_t (&words)[2], uint64_t value) {
    words[0] = uint32_t(value);
    words[1] = uint32_t(value >> 32);
}

static __device__ __forceinline__ uint64_t g2_trace_standard_immediate(
    RvrOperandEntry const &entry
) {
    if (entry.flags & RVR_OPERAND_FLAG_RS2_IMM_SIGN)
        return uint64_t(int64_t(int16_t(entry.c)));
    return uint64_t(entry.c);
}

static __device__ __forceinline__ uint32_t g2_trace_effective_address(
    uint32_t pointer, RvrOperandEntry const &entry
) {
    int32_t offset = int32_t(entry.c & 0xffffu);
    if (entry.flags & RVR_OPERAND_FLAG_LS_IMM_SIGN) offset = int32_t(int16_t(offset));
    return pointer + uint32_t(offset);
}

static __device__ __forceinline__ uint8_t g2_trace_memory_width(
    G2TraceSource const &source, RvrOperandEntry const &entry
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
    default: g2_trace_fail(source, 97); return 0;
    }
}

static __device__ __forceinline__ void g2_trace_u64_to_u16(
    uint64_t value, uint16_t (&out)[BLOCK_FE_WIDTH]
) {
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; ++i) out[i] = uint16_t(value >> (16 * i));
}

static __device__ __forceinline__ bool g2_trace_alu3(
    G2TraceSource const &source,
    RvrOperandEntry const *operand_table,
    uint32_t row_index,
    RvrAlu3Compact &record
) {
    G2TraceRow row;
    if (!g2_trace_row(source, operand_table, row_index, row)) return false;
    RvrOperandEntry const &entry = row.entry;
    record = {};
    record.from_pc = row.from_pc;
    record.from_timestamp = row.from_timestamp;

    G2TimelineEvent first, second, write;
    if (!g2_trace_event(source, row, 0, first)) return false;
    record.reads_prev_timestamp[0] = first.previous_timestamp;

    uint64_t b = 0, c = 0;
    if (entry.access_pattern == 8) {
        if (source.kind != 29 || !g2_trace_event(source, row, 1, write))
            return false;
        b = first.value;
        c = uint64_t(int64_t(int32_t(entry.c << 20) >> 20));
    } else if (entry.access_pattern == 0 || entry.access_pattern == 1) {
        if (!g2_trace_event(source, row, 2, write))
            return false;
        b = first.value;
        if (entry.flags & RVR_OPERAND_FLAG_RS2_IMM) {
            c = g2_trace_standard_immediate(entry);
        } else {
            if (!g2_trace_event(source, row, 1, second))
                return false;
            c = second.value;
            record.reads_prev_timestamp[1] = second.previous_timestamp;
        }
    } else if (entry.access_pattern == 2 || entry.access_pattern == 3) {
        if (!g2_trace_event(source, row, 1, second))
            return false;
        b = first.value;
        c = second.value;
        record.reads_prev_timestamp[1] = second.previous_timestamp;
        if (entry.access_pattern == 3 ||
            (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED)) {
            if (!g2_trace_event(source, row, 2, write)) return false;
            record.write_prev_timestamp = write.previous_timestamp;
            g2_trace_store_u64(record.write_prev_data, write.value);
        }
    } else {
        g2_trace_fail(source, 98);
        return false;
    }
    if (entry.access_pattern != 2 && entry.access_pattern != 3) {
        record.write_prev_timestamp = write.previous_timestamp;
        g2_trace_store_u64(record.write_prev_data, write.value);
    }
    g2_trace_store_u64(record.b, b);
    g2_trace_store_u64(record.c, c);
    return true;
}

static __device__ __forceinline__ bool g2_trace_branch2(
    G2TraceSource const &source,
    RvrOperandEntry const *operand_table,
    uint32_t row_index,
    RvrBranch2Compact &record
) {
    G2TraceRow row;
    G2TimelineEvent first, second;
    uint64_t b, c;
    if (!g2_trace_row(source, operand_table, row_index, row) || row.entry.access_pattern != 4 ||
        !g2_trace_event(source, row, 0, first) || !g2_trace_event(source, row, 1, second)) {
        g2_trace_fail(source, 99);
        return false;
    }
    b = first.value;
    c = second.value;
    record = {};
    record.from_pc = row.from_pc;
    record.from_timestamp = row.from_timestamp;
    record.reads_prev_timestamp[0] = first.previous_timestamp;
    record.reads_prev_timestamp[1] = second.previous_timestamp;
    g2_trace_store_u64(record.b, b);
    g2_trace_store_u64(record.c, c);
    return true;
}

static __device__ __forceinline__ bool g2_trace_wr1(
    G2TraceSource const &source,
    RvrOperandEntry const *operand_table,
    uint32_t row_index,
    RvrWr1Compact &record
) {
    G2TraceRow row;
    if (!g2_trace_row(source, operand_table, row_index, row) ||
        (row.entry.access_pattern != 5 && row.entry.access_pattern != 6)) {
        g2_trace_fail(source, 100);
        return false;
    }
    record = {};
    record.from_pc = row.from_pc;
    record.from_timestamp = row.from_timestamp;
    if (row.entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED) {
        G2TimelineEvent write;
        if (!g2_trace_event(source, row, 0, write)) return false;
        record.write_prev_timestamp = write.previous_timestamp;
        g2_trace_store_u64(record.write_prev_data, write.value);
    }
    return true;
}

static __device__ __forceinline__ bool g2_trace_rw1(
    G2TraceSource const &source,
    RvrOperandEntry const *operand_table,
    uint32_t row_index,
    RvrRw1Compact &record
) {
    G2TraceRow row;
    G2TimelineEvent read;
    uint64_t b;
    if (!g2_trace_row(source, operand_table, row_index, row) || row.entry.access_pattern != 7 ||
        !g2_trace_event(source, row, 0, read)) {
        g2_trace_fail(source, 101);
        return false;
    }
    b = read.value;
    record = {};
    record.from_pc = row.from_pc;
    record.from_timestamp = row.from_timestamp;
    record.read_prev_timestamp = read.previous_timestamp;
    g2_trace_store_u64(record.b, b);
    if (row.entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED) {
        G2TimelineEvent write;
        if (!g2_trace_event(source, row, 1, write)) return false;
        record.write_prev_timestamp = write.previous_timestamp;
        g2_trace_store_u64(record.write_prev_data, write.value);
    }
    return true;
}

static __device__ __forceinline__ bool g2_trace_load(
    G2TraceSource const &source,
    RvrOperandEntry const *operand_table,
    uint32_t row_index,
    Rv64LoadRecord &record
) {
    G2TraceRow row;
    G2TimelineEvent base, block0;
    if (!g2_trace_row(source, operand_table, row_index, row) || row.entry.access_pattern != 2 ||
        !g2_trace_event(source, row, 0, base) || !g2_trace_event(source, row, 1, block0)) {
        g2_trace_fail(source, 102);
        return false;
    }
    RvrOperandEntry const &entry = row.entry;
    uint64_t pointer_value = base.value;
    if (row.prepared.v0_index != UINT32_MAX &&
        !g2_trace_lane_value(source, false, row.prepared.v0_index, pointer_value))
        return false;
    if (pointer_value > UINT32_MAX) {
        g2_trace_fail(source, 102);
        return false;
    }
    uint8_t width = g2_trace_memory_width(source, entry);
    if (width <= 1) return false;
    uint32_t pointer = uint32_t(pointer_value);
    bool crossing = (g2_trace_effective_address(pointer, entry) & 7u) + width > 8u;
    G2TimelineEvent block1{};
    if (crossing && !g2_trace_event(source, row, 2, block1)) return false;

    record = {};
    record.adapter.from_pc = row.from_pc;
    record.adapter.from_timestamp = row.from_timestamp;
    record.adapter.rs1_val = pointer;
    record.adapter.rs1_aux_record.prev_timestamp = base.previous_timestamp;
    record.adapter.read_data_aux[0].prev_timestamp = block0.previous_timestamp;
    record.adapter.read_data_aux[1].prev_timestamp =
        crossing ? block1.previous_timestamp : UINT32_MAX;
    record.adapter.imm = uint16_t(entry.c);
    record.adapter.imm_sign = (entry.flags & RVR_OPERAND_FLAG_LS_IMM_SIGN) != 0;
    record.adapter.rs1_ptr = uint8_t(entry.b);
    record.adapter.rd_ptr = (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED)
                                ? uint8_t(entry.a)
                                : UINT8_MAX;
    g2_trace_u64_to_u16(block0.value, record.core.read_data[0]);
    g2_trace_u64_to_u16(crossing ? block1.value : 0, record.core.read_data[1]);
    if (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED) {
        G2TimelineEvent write;
        if (!g2_trace_event(source, row, 3, write)) return false;
        record.adapter.write_prev_timestamp = write.previous_timestamp;
        g2_trace_u64_to_u16(write.value, record.adapter.write_prev_data);
    }
    return true;
}

static __device__ __forceinline__ bool g2_trace_store(
    G2TraceSource const &source,
    RvrOperandEntry const *operand_table,
    uint32_t row_index,
    Rv64StoreRecord &record
) {
    G2TraceRow row;
    G2TimelineEvent base, value, block0;
    if (!g2_trace_row(source, operand_table, row_index, row) || row.entry.access_pattern != 3 ||
        !g2_trace_event(source, row, 0, base) || !g2_trace_event(source, row, 1, value) ||
        !g2_trace_event(source, row, 2, block0)) {
        g2_trace_fail(source, 103);
        return false;
    }
    RvrOperandEntry const &entry = row.entry;
    uint64_t pointer_value = base.value;
    if (row.prepared.v0_index != UINT32_MAX &&
        !g2_trace_lane_value(source, false, row.prepared.v0_index, pointer_value))
        return false;
    if (pointer_value > UINT32_MAX) {
        g2_trace_fail(source, 103);
        return false;
    }
    uint8_t width = g2_trace_memory_width(source, entry);
    if (width <= 1) return false;
    uint32_t pointer = uint32_t(pointer_value);
    bool crossing = (g2_trace_effective_address(pointer, entry) & 7u) + width > 8u;
    G2TimelineEvent block1{};
    if (crossing && !g2_trace_event(source, row, 3, block1)) return false;

    record = {};
    record.adapter.from_pc = row.from_pc;
    record.adapter.from_timestamp = row.from_timestamp;
    record.adapter.rs1_val = pointer;
    record.adapter.rs1_aux_record.prev_timestamp = base.previous_timestamp;
    record.adapter.read_data_aux.prev_timestamp = value.previous_timestamp;
    record.adapter.write_prev_timestamps[0] = block0.previous_timestamp;
    record.adapter.write_prev_timestamps[1] =
        crossing ? block1.previous_timestamp : UINT32_MAX;
    record.adapter.imm = uint16_t(entry.c);
    record.adapter.rs1_ptr = uint8_t(entry.b);
    record.adapter.rs2_ptr = uint8_t(entry.a);
    record.adapter.imm_sign = (entry.flags & RVR_OPERAND_FLAG_LS_IMM_SIGN) != 0;
    record.adapter.mem_as = (entry.flags & RVR_OPERAND_FLAG_LS_PUBLIC_VALUES) ? 3 : 2;
    g2_trace_u64_to_u16(value.value, record.core.read_data);
    g2_trace_u64_to_u16(block0.value, record.core.prev_data[0]);
    g2_trace_u64_to_u16(crossing ? block1.value : 0, record.core.prev_data[1]);
    return true;
}

struct G2HintReplayRow {
    uint32_t from_pc;
    uint32_t timestamp;
    uint32_t local_idx;
    uint32_t num_words;
    uint32_t mem_ptr_ptr;
    uint32_t mem_ptr;
    uint32_t mem_ptr_prev_timestamp;
    uint32_t num_words_ptr;
    uint32_t num_words_prev_timestamp;
    uint32_t write_prev_timestamp;
    uint64_t write_prev_data;
    uint64_t data;
    uint64_t reserved;
};
static_assert(sizeof(G2HintReplayRow) == 64, "G2 HintStore replay-row size drift");

static __device__ __forceinline__ bool g2_trace_hint(
    G2TraceSource const &source,
    RvrOperandEntry const *operand_table,
    uint32_t row_index,
    G2HintReplayRow &record
) {
    G2TraceRow row;
    if (!g2_trace_row(source, operand_table, row_index, row) || source.kind != 30 ||
        row.entry.access_pattern != 9 || row.entry.local_opcode > 1) {
        g2_trace_fail(source, 105);
        return false;
    }
    G2TimelineEvent mem_ptr, write;
    if (!g2_trace_event(source, row, 0, mem_ptr) ||
        !g2_trace_event(source, row, row.local_index * 3u + 2u, write))
        return false;
    uint64_t data;
    if (!g2_trace_residual_value(source, write.residual_index, data)) return false;
    record = {};
    record.from_pc = row.from_pc;
    record.timestamp = row.from_timestamp;
    record.local_idx = row.local_index;
    record.num_words = row.prepared.v0_index;
    record.mem_ptr_ptr = row.entry.b;
    record.mem_ptr = uint32_t(mem_ptr.value);
    record.mem_ptr_prev_timestamp =
        row.local_index == 0 ? mem_ptr.previous_timestamp : 0;
    record.num_words_ptr = row.entry.local_opcode == 0 ? UINT32_MAX : row.entry.a;
    if (row.local_index == 0 && row.entry.local_opcode != 0) {
        G2TimelineEvent num_words;
        if (!g2_trace_event(source, row, 1, num_words) ||
            num_words.value != row.prepared.v0_index) {
            g2_trace_fail(source, 105);
            return false;
        }
        record.num_words_prev_timestamp = num_words.previous_timestamp;
    }
    record.write_prev_timestamp = write.previous_timestamp;
    record.write_prev_data = write.value;
    record.data = data;
    return true;
}

template <typename Record> struct G2TraceView;

template <> struct G2TraceView<RvrAlu3Compact> {
    G2TraceSource source;
    RvrOperandEntry const *operand_table;
    __device__ __forceinline__ size_t len() const { return source.row_count; }
    __device__ __forceinline__ RvrAlu3Compact operator[](size_t index) const {
        RvrAlu3Compact record{};
        g2_trace_alu3(source, operand_table, uint32_t(index), record);
        return record;
    }
};

template <> struct G2TraceView<RvrBranch2Compact> {
    G2TraceSource source;
    RvrOperandEntry const *operand_table;
    __device__ __forceinline__ size_t len() const { return source.row_count; }
    __device__ __forceinline__ RvrBranch2Compact operator[](size_t index) const {
        RvrBranch2Compact record{};
        g2_trace_branch2(source, operand_table, uint32_t(index), record);
        return record;
    }
};

template <> struct G2TraceView<RvrWr1Compact> {
    G2TraceSource source;
    RvrOperandEntry const *operand_table;
    __device__ __forceinline__ size_t len() const { return source.row_count; }
    __device__ __forceinline__ RvrWr1Compact operator[](size_t index) const {
        RvrWr1Compact record{};
        g2_trace_wr1(source, operand_table, uint32_t(index), record);
        return record;
    }
};

template <> struct G2TraceView<RvrRw1Compact> {
    G2TraceSource source;
    RvrOperandEntry const *operand_table;
    __device__ __forceinline__ size_t len() const { return source.row_count; }
    __device__ __forceinline__ RvrRw1Compact operator[](size_t index) const {
        RvrRw1Compact record{};
        g2_trace_rw1(source, operand_table, uint32_t(index), record);
        return record;
    }
};

template <> struct G2TraceView<Rv64LoadRecord> {
    G2TraceSource source;
    RvrOperandEntry const *operand_table;
    __device__ __forceinline__ size_t len() const { return source.row_count; }
    __device__ __forceinline__ Rv64LoadRecord operator[](size_t index) const {
        Rv64LoadRecord record{};
        g2_trace_load(source, operand_table, uint32_t(index), record);
        return record;
    }
};

template <> struct G2TraceView<Rv64LoadSignExtendRecord> {
    G2TraceSource source;
    RvrOperandEntry const *operand_table;
    __device__ __forceinline__ size_t len() const { return source.row_count; }
    __device__ __forceinline__ Rv64LoadSignExtendRecord operator[](size_t index) const {
        Rv64LoadRecord decoded{};
        g2_trace_load(source, operand_table, uint32_t(index), decoded);
        Rv64LoadSignExtendRecord record{};
        record.adapter = decoded.adapter;
        record.core = decoded.core;
        return record;
    }
};

template <> struct G2TraceView<Rv64StoreRecord> {
    G2TraceSource source;
    RvrOperandEntry const *operand_table;
    __device__ __forceinline__ size_t len() const { return source.row_count; }
    __device__ __forceinline__ Rv64StoreRecord operator[](size_t index) const {
        Rv64StoreRecord record{};
        g2_trace_store(source, operand_table, uint32_t(index), record);
        return record;
    }
};

} // namespace riscv

// Every standard AIR exposes the same host ABI to the central G2 dispatcher.
// Individual translation units keep their established lookup arguments and
// ignore the superset fields they do not consume.
#define RVR_G2_TRACEGEN_PARAMETERS                                                        \
    Fp *trace,                                                                           \
        size_t height,                                                                   \
        size_t width,                                                                    \
        riscv::G2TraceSource source,                                                      \
        riscv::RvrOperandEntry const *operand_table,                                      \
        uint32_t pc_base,                                                                 \
        size_t pointer_max_bits,                                                          \
        uint32_t *range_checker,                                                          \
        uint32_t range_checker_num_bins,                                                  \
        uint32_t *bitwise_lookup,                                                         \
        uint32_t *range_tuple_checker,                                                    \
        uint2 range_tuple_checker_sizes,                                                  \
        uint32_t timestamp_max_bits,                                                      \
        cudaStream_t stream

#define RVR_G2_REFERENCE_PARAMETERS                                                       \
    Fp *trace,                                                                           \
        size_t height,                                                                   \
        size_t width,                                                                    \
        DeviceRawBufferConstView records,                                                 \
        riscv::RvrOperandEntry const *operand_table,                                      \
        uint32_t pc_base,                                                                 \
        size_t pointer_max_bits,                                                          \
        uint32_t *range_checker,                                                          \
        uint32_t range_checker_num_bins,                                                  \
        uint32_t *bitwise_lookup,                                                         \
        uint32_t *range_tuple_checker,                                                    \
        uint2 range_tuple_checker_sizes,                                                  \
        uint32_t timestamp_max_bits,                                                      \
        cudaStream_t stream

#ifdef OPENVM_RVR_CUDA_G2_ONLY
#define OPENVM_RVR_G2_REFERENCE(...) return int(cudaErrorNotSupported);
#else
#define OPENVM_RVR_G2_REFERENCE(...) __VA_ARGS__
#endif

#define OPENVM_RVR_G2_PRELOAD_LOCALS()                                                   \
    [[maybe_unused]] riscv::G2TraceSource source{};                                      \
    [[maybe_unused]] riscv::RvrOperandEntry const *operand_table = nullptr;              \
    [[maybe_unused]] uint32_t pc_base = 0;                                               \
    [[maybe_unused]] size_t pointer_max_bits = 0;                                        \
    [[maybe_unused]] uint32_t *range_checker = nullptr;                                  \
    [[maybe_unused]] uint32_t range_checker_num_bins = 0;                                \
    [[maybe_unused]] uint32_t *bitwise_lookup = nullptr;                                 \
    [[maybe_unused]] uint32_t *range_tuple_checker = nullptr;                            \
    [[maybe_unused]] uint2 range_tuple_checker_sizes{};                                  \
    [[maybe_unused]] uint32_t timestamp_max_bits = 0

#define DEFINE_RVR_G2_TRACEGEN_LAUNCHER(                                                  \
    name, cols, kernel, record, threads, ...                                              \
)                                                                                        \
    extern "C" int name(RVR_G2_TRACEGEN_PARAMETERS) {                                    \
        assert(width == sizeof(cols<uint8_t>));                                           \
        auto [grid, block] = kernel_launch_params(height, threads);                       \
        kernel<<<grid, block, 0, stream>>>(                                               \
            trace, height, riscv::G2TraceView<record>{source, operand_table}, __VA_ARGS__ \
        );                                                                                \
        return CHECK_KERNEL();                                                            \
    }                                                                                     \
    extern "C" int name##_preload(Fp *trace, cudaStream_t stream) {                    \
        OPENVM_RVR_G2_PRELOAD_LOCALS();                                                  \
        cudaFuncAttributes attributes{};                                                  \
        cudaError_t status = cudaFuncGetAttributes(                                      \
            &attributes, kernel<riscv::G2TraceView<record>>                              \
        );                                                                               \
        if (status != cudaSuccess) return int(status);                                   \
        kernel<<<1, 1, 0, stream>>>(                                                     \
            trace, 0, riscv::G2TraceView<record>{source, operand_table}, __VA_ARGS__     \
        );                                                                               \
        return CHECK_KERNEL();                                                           \
    }                                                                                     \
    extern "C" int name##_reference(RVR_G2_REFERENCE_PARAMETERS) {                       \
        OPENVM_RVR_G2_REFERENCE(                                                          \
        assert(width == sizeof(cols<uint8_t>));                                           \
        auto [grid, block] = kernel_launch_params(height, threads);                       \
        kernel<<<grid, block, 0, stream>>>(                                               \
            trace, height, records.as_typed<record>(), __VA_ARGS__                        \
        );                                                                                \
        return CHECK_KERNEL();                                                            \
        )                                                                                 \
    }

#define DEFINE_RVR_G2_TRACEGEN_LAUNCHER_WITH_WIDTH(                                       \
    name, cols, kernel, record, threads, ...                                              \
)                                                                                        \
    extern "C" int name(RVR_G2_TRACEGEN_PARAMETERS) {                                    \
        assert(width == sizeof(cols<uint8_t>));                                           \
        auto [grid, block] = kernel_launch_params(height, threads);                       \
        kernel<<<grid, block, 0, stream>>>(                                               \
            trace, height, width, riscv::G2TraceView<record>{source, operand_table},       \
            __VA_ARGS__                                                                  \
        );                                                                                \
        return CHECK_KERNEL();                                                            \
    }                                                                                     \
    extern "C" int name##_preload(Fp *trace, cudaStream_t stream) {                    \
        OPENVM_RVR_G2_PRELOAD_LOCALS();                                                  \
        cudaFuncAttributes attributes{};                                                  \
        cudaError_t status = cudaFuncGetAttributes(                                      \
            &attributes, kernel<riscv::G2TraceView<record>>                              \
        );                                                                               \
        if (status != cudaSuccess) return int(status);                                   \
        kernel<<<1, 1, 0, stream>>>(                                                     \
            trace, 0, sizeof(cols<uint8_t>),                                             \
            riscv::G2TraceView<record>{source, operand_table}, __VA_ARGS__               \
        );                                                                               \
        return CHECK_KERNEL();                                                           \
    }                                                                                     \
    extern "C" int name##_reference(RVR_G2_REFERENCE_PARAMETERS) {                       \
        OPENVM_RVR_G2_REFERENCE(                                                          \
        assert(width == sizeof(cols<uint8_t>));                                           \
        auto [grid, block] = kernel_launch_params(height, threads);                       \
        kernel<<<grid, block, 0, stream>>>(                                               \
            trace, height, width, records.as_typed<record>(), __VA_ARGS__                 \
        );                                                                                \
        return CHECK_KERNEL();                                                            \
        )                                                                                 \
    }

extern "C" int _add_sub_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_add_sub_w_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _addi_tracegen_g2_common(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _auipc_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _beq_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _bitwise_logic_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _blt_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_div_rem_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_div_rem_w_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _jal_lui_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _jalr_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_less_than_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_load_byte_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_load_halfword_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_load_word_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_load_doubleword_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_load_sign_extend_byte_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_load_sign_extend_halfword_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_load_sign_extend_word_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _mul_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _mulh_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_mul_w_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_shift_logical_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_shift_right_arithmetic_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_shift_w_logical_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_shift_w_right_arithmetic_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_store_byte_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_store_halfword_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_store_word_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _rv64_store_doubleword_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);
extern "C" int _hintstore_tracegen_g2(RVR_G2_TRACEGEN_PARAMETERS);

#define DECLARE_RVR_G2_PRELOAD(name)                                                      \
    extern "C" int name##_preload(Fp *trace, cudaStream_t stream);
DECLARE_RVR_G2_PRELOAD(_add_sub_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_add_sub_w_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_addi_tracegen_g2_common)
DECLARE_RVR_G2_PRELOAD(_auipc_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_beq_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_bitwise_logic_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_blt_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_div_rem_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_div_rem_w_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_jal_lui_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_jalr_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_less_than_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_load_byte_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_load_halfword_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_load_word_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_load_doubleword_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_load_sign_extend_byte_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_load_sign_extend_halfword_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_load_sign_extend_word_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_mul_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_mulh_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_mul_w_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_shift_logical_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_shift_right_arithmetic_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_shift_w_logical_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_shift_w_right_arithmetic_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_store_byte_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_store_halfword_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_store_word_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_rv64_store_doubleword_tracegen_g2)
DECLARE_RVR_G2_PRELOAD(_hintstore_tracegen_g2)
#undef DECLARE_RVR_G2_PRELOAD

extern "C" int _add_sub_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_add_sub_w_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _addi_tracegen_g2_common_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _auipc_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _beq_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _bitwise_logic_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _blt_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_div_rem_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_div_rem_w_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _jal_lui_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _jalr_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_less_than_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_load_byte_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_load_halfword_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_load_word_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_load_doubleword_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_load_sign_extend_byte_tracegen_g2_reference(
    RVR_G2_REFERENCE_PARAMETERS
);
extern "C" int _rv64_load_sign_extend_halfword_tracegen_g2_reference(
    RVR_G2_REFERENCE_PARAMETERS
);
extern "C" int _rv64_load_sign_extend_word_tracegen_g2_reference(
    RVR_G2_REFERENCE_PARAMETERS
);
extern "C" int _mul_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _mulh_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_mul_w_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_shift_logical_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_shift_right_arithmetic_tracegen_g2_reference(
    RVR_G2_REFERENCE_PARAMETERS
);
extern "C" int _rv64_shift_w_logical_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_shift_w_right_arithmetic_tracegen_g2_reference(
    RVR_G2_REFERENCE_PARAMETERS
);
extern "C" int _rv64_store_byte_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_store_halfword_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_store_word_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _rv64_store_doubleword_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
extern "C" int _hintstore_tracegen_g2_reference(RVR_G2_REFERENCE_PARAMETERS);
