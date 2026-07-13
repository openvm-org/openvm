#include "primitives/buffer_view.cuh"
#include "riscv/rvr_compact.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <stdint.h>

using namespace riscv;

namespace {

static constexpr uint32_t DELTA_STRIDE = 32;
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
};

struct DeltaRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint64_t v0;
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

struct ProgramLogEntry {
    uint16_t opcode;
    uint16_t _pad0;
    uint32_t timestamp;
    uint64_t pc;
};
static_assert(sizeof(ProgramLogEntry) == 16, "program log size drift");

struct EventPayload {
    uint64_t address_key;
    uint32_t timestamp;
    uint32_t output_index_plus_one;
};
static_assert(sizeof(EventPayload) == 16, "event payload size drift");

struct PredPayload {
    uint32_t timestamp;
    uint32_t output_index_plus_one;
};
static_assert(sizeof(PredPayload) == 8, "predecessor payload size drift");

struct DeltaAirOutputDesc {
    uint64_t base;
    uint32_t count;
    uint32_t stride;
    uint32_t sorted_start;
    uint32_t _reserved;
};
static_assert(sizeof(DeltaAirOutputDesc) == 24, "delta output descriptor size drift");

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
    if (entry.air_idx == INVALID_AIR || entry.access_pattern > DELTA_RW1) {
        fail(error, 4);
        return false;
    }
    return true;
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
        int32_t imm = int32_t(entry.c & 0xffffu);
        if (entry.flags & RVR_OPERAND_FLAG_LS_IMM_SIGN) {
            imm = int32_t(int16_t(imm));
        }
        uint32_t effective = uint32_t(record.v1) + uint32_t(imm);
        addr_space =
            (entry.flags & RVR_OPERAND_FLAG_LS_PUBLIC_VALUES) ? uint8_t(3) : uint8_t(2);
        address = uint64_t(effective & ~uint32_t(7));
        return true;
    };

    switch (entry.access_pattern) {
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
        if (slot == 1) return memory();
        if (slot == 2 && (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED)) return reg(entry.a);
        return false;
    case DELTA_STORE:
        if (slot == 0) return reg(entry.b);
        if (slot == 1) return reg(entry.a);
        if (slot == 2) return memory();
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
    default:
        return false;
    }
}

__global__ void build_events(
    DeltaRecord const *delta,
    size_t delta_count,
    MemoryLogEntry const *memory,
    size_t memory_count,
    ProgramLogEntry const *program,
    size_t program_count,
    RvrOperandEntry const *table,
    size_t operand_count,
    uint32_t pc_base,
    uint8_t const *arena_native_flags,
    size_t num_airs,
    uint32_t *timestamp_keys,
    EventPayload *payloads,
    uint32_t *error
) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t delta_events = delta_count * 3;
    size_t memory_begin = delta_events;
    size_t program_begin = memory_begin + memory_count;
    size_t event_count = program_begin + program_count * 3;
    if (idx >= event_count) return;

    EventPayload payload{INVALID_ADDRESS, UINT32_MAX, 0};
    if (idx < delta_events) {
        size_t record_idx = idx / 3;
        uint32_t access_slot = idx % 3;
        DeltaRecord record = delta[record_idx];
        RvrOperandEntry entry;
        if (operand_for_pc(record.from_pc, pc_base, table, operand_count, entry, error)) {
            uint8_t as;
            uint64_t address;
            if (delta_access(record, entry, access_slot, as, address)) {
                payload.address_key = address_key(as, address, error);
                payload.timestamp = record.from_timestamp + access_slot;
                payload.output_index_plus_one = uint32_t(idx + 1);
            }
        }
    } else if (idx < program_begin) {
        MemoryLogEntry entry = memory[idx - memory_begin];
        payload.address_key = address_key(entry.addr_space, entry.address & ~uint64_t(7), error);
        payload.timestamp = entry.timestamp;
    } else {
        size_t local = idx - program_begin;
        ProgramLogEntry program_entry = program[local / 3];
        uint32_t access_slot = local % 3;
        if (program_entry.pc <= UINT32_MAX && program_entry.pc >= pc_base &&
            ((uint32_t(program_entry.pc) - pc_base) % PC_STEP) == 0) {
            size_t table_slot = (uint32_t(program_entry.pc) - pc_base) / PC_STEP;
            if (table_slot < operand_count) {
                RvrOperandEntry entry = table[table_slot];
                if (entry.air_idx < num_airs && entry.access_pattern <= DELTA_RW1 &&
                    arena_native_flags[entry.air_idx]) {
                DeltaRecord synthetic{
                    uint32_t(program_entry.pc), program_entry.timestamp, 0, 0, 0
                };
                uint8_t as;
                uint64_t address;
                if (delta_access(synthetic, entry, access_slot, as, address)) {
                    payload.address_key = address_key(as, address, error);
                    payload.timestamp = program_entry.timestamp + access_slot;
                }
                }
            }
        }
    }
    timestamp_keys[idx] = payload.timestamp;
    payloads[idx] = payload;
}

__global__ void extract_address_keys(
    EventPayload const *events, uint64_t *keys, PredPayload *payloads, size_t count
) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    EventPayload event = events[idx];
    keys[idx] = event.address_key;
    payloads[idx] = PredPayload{event.timestamp, event.output_index_plus_one};
}

__global__ void scatter_predecessors(
    uint64_t const *keys,
    PredPayload const *events,
    size_t count,
    uint32_t *prev_timestamps
) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    PredPayload event = events[idx];
    if (event.output_index_plus_one == 0) return;
    uint32_t prev = 0;
    if (idx != 0 && keys[idx] != INVALID_ADDRESS && keys[idx - 1] == keys[idx]) {
        prev = events[idx - 1].timestamp;
    }
    prev_timestamps[event.output_index_plus_one - 1] = prev;
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

__global__ void partition_records(
    DeltaRecord const *delta,
    uint32_t const *prev,
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
    uint32_t const *record_prev = prev + size_t(record_idx) * 3;
    store_u32(dst, record.from_pc);
    store_u32(dst + 4, record.from_timestamp);
    switch (entry.access_pattern) {
    case DELTA_ALU3:
    case DELTA_ALU3_REG:
    case DELTA_LOAD:
    case DELTA_STORE:
        if (desc.stride != sizeof(RvrAlu3Compact)) {
            fail(error, 8);
            return;
        }
        store_u32(dst + 8, record_prev[0]);
        store_u32(dst + 12, record_prev[1]);
        store_u32(dst + 16, record_prev[2]);
        store_u64_words(dst + 20, record.v0);
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
        store_u64_words(dst + 12, record.v0);
        break;
    case DELTA_RW1:
        if (desc.stride != sizeof(RvrRw1Compact)) {
            fail(error, 11);
            return;
        }
        store_u32(dst + 8, record_prev[0]);
        store_u32(dst + 12, record_prev[1]);
        store_u64_words(dst + 16, record.v1);
        store_u64_words(dst + 24, record.v0);
        break;
    default:
        fail(error, 12);
    }
}

} // namespace

extern "C" int _rvr_delta_predecode(
    DeviceBufferConstView<uint8_t> d_delta_bytes,
    size_t delta_count,
    DeviceBufferConstView<uint8_t> d_memory_bytes,
    size_t memory_count,
    DeviceBufferConstView<uint8_t> d_program_bytes,
    size_t program_count,
    RvrOperandEntry const *d_operand_table,
    size_t operand_count,
    uint32_t pc_base,
    uint8_t const *d_arena_native_flags,
    size_t num_airs,
    DeltaAirOutputDesc const *d_outputs,
    uint32_t *d_error,
    cudaStream_t stream
) {
    if (delta_count > UINT32_MAX / 3 ||
        d_delta_bytes.size != delta_count * sizeof(DeltaRecord) ||
        d_memory_bytes.size != memory_count * sizeof(MemoryLogEntry) ||
        d_program_bytes.size != program_count * sizeof(ProgramLogEntry)) {
        return int(cudaErrorInvalidValue);
    }
    // There is no compact output and no predecessor consumed by a later
    // delta record. In particular, avoid zero-byte allocations when a segment
    // contains only residual / arena-native instructions.
    if (delta_count == 0) return 0;
    size_t event_count = delta_count * 3 + memory_count + program_count * 3;

    uint32_t *ts_a = nullptr, *ts_b = nullptr, *indices_a = nullptr, *indices_b = nullptr;
    uint64_t *addr_a = nullptr, *addr_b = nullptr;
    uint8_t *air_a = nullptr, *air_b = nullptr;
    EventPayload *events_a = nullptr, *events_b = nullptr;
    PredPayload *pred_a = nullptr, *pred_b = nullptr;
    uint32_t *prev = nullptr;
    void *temp = nullptr;
    size_t ts_temp = 0, addr_temp = 0, air_temp = 0;
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
    CUDA_ALLOC(pred_a, event_count * sizeof(PredPayload));
    CUDA_ALLOC(pred_b, event_count * sizeof(PredPayload));
    CUDA_ALLOC(prev, delta_count * 3 * sizeof(uint32_t));
    CUDA_ALLOC(air_a, delta_count * sizeof(uint8_t));
    CUDA_ALLOC(air_b, delta_count * sizeof(uint8_t));
    CUDA_ALLOC(indices_a, delta_count * sizeof(uint32_t));
    CUDA_ALLOC(indices_b, delta_count * sizeof(uint32_t));
    CUDA_TRY(cudaMemsetAsync(prev, 0, delta_count * 3 * sizeof(uint32_t), stream));

    {
        dim3 block(256);
        dim3 grid((event_count + block.x - 1) / block.x);
        build_events<<<grid, block, 0, stream>>>(
            reinterpret_cast<DeltaRecord const *>(d_delta_bytes.ptr),
            delta_count,
            reinterpret_cast<MemoryLogEntry const *>(d_memory_bytes.ptr),
            memory_count,
            reinterpret_cast<ProgramLogEntry const *>(d_program_bytes.ptr),
            program_count,
            d_operand_table,
            operand_count,
            pc_base,
            d_arena_native_flags,
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
        nullptr, addr_temp, addr_a, addr_b, pred_a, pred_b, event_count, 0, 64, stream
    ));
    CUDA_TRY(cub::DeviceRadixSort::SortPairs(
        nullptr, air_temp, air_a, air_b, indices_a, indices_b, delta_count, 0, 8, stream
    ));
    {
        size_t temp_bytes = ts_temp;
        if (addr_temp > temp_bytes) temp_bytes = addr_temp;
        if (air_temp > temp_bytes) temp_bytes = air_temp;
        CUDA_ALLOC(temp, temp_bytes);
    }
    CUDA_TRY(cub::DeviceRadixSort::SortPairs(
        temp, ts_temp, ts_a, ts_b, events_a, events_b, event_count, 0, 32, stream
    ));
    {
        dim3 block(256);
        dim3 grid((event_count + block.x - 1) / block.x);
        extract_address_keys<<<grid, block, 0, stream>>>(
            events_b, addr_a, pred_a, event_count
        );
        CUDA_TRY(cudaGetLastError());
    }
    CUDA_TRY(cub::DeviceRadixSort::SortPairs(
        temp, addr_temp, addr_a, addr_b, pred_a, pred_b, event_count, 0, 64, stream
    ));
    {
        dim3 block(256);
        dim3 grid((event_count + block.x - 1) / block.x);
        scatter_predecessors<<<grid, block, 0, stream>>>(addr_b, pred_b, event_count, prev);
        CUDA_TRY(cudaGetLastError());
    }
    {
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
    CUDA_TRY(cub::DeviceRadixSort::SortPairs(
        temp, air_temp, air_a, air_b, indices_a, indices_b, delta_count, 0, 8, stream
    ));
    {
        dim3 block(256);
        dim3 grid((delta_count + block.x - 1) / block.x);
        partition_records<<<grid, block, 0, stream>>>(
            reinterpret_cast<DeltaRecord const *>(d_delta_bytes.ptr),
            prev,
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
    if (air_b) cudaFreeAsync(air_b, stream);
    if (air_a) cudaFreeAsync(air_a, stream);
    if (prev) cudaFreeAsync(prev, stream);
    if (pred_b) cudaFreeAsync(pred_b, stream);
    if (pred_a) cudaFreeAsync(pred_a, stream);
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
