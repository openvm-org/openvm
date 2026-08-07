#pragma once

#include "primitives/buffer_view.cuh"

static constexpr uint32_t PREFLIGHT_WRITE_BIT = 1u << 31;
static constexpr uint32_t PREFLIGHT_ADDRESS_SPACE_MASK = ~PREFLIGHT_WRITE_BIT;
static constexpr uint32_t MEMORY_PREDECESSOR_SEED_BIT = 1u << 31;
static constexpr uint32_t MEMORY_PREDECESSOR_INDEX_MASK = ~MEMORY_PREDECESSOR_SEED_BIT;

struct PreflightProgramEvent {
    uint32_t pc;
    uint32_t timestamp;
};

struct PreflightMemoryEvent {
    uint32_t timestamp;
    uint32_t address_space_and_kind;
    uint32_t pointer;
    // Eight-byte inline payload: U8 cells 0/1 occupy the low/high bytes of
    // value[0], cells 2/3 occupy value[1], and value[2..4] must be zero. U16
    // uses all four lanes; Field32 stores a sidecar index in value[0] and
    // value[1], and requires value[2] and value[3] to be zero.
    uint16_t value[4];
};

struct PreflightInitialWrite {
    uint32_t address_space;
    uint32_t pointer;
    uint16_t initial_value[4];
};

// One field-cell memory-bus block represented by canonical unsigned integers.
// For FIELD32 events/seeds, the low 32 bits of the compact value payload index
// the corresponding dense event/first-write sidecar; the high 32 bits are zero.
struct GpuFieldBlock {
    uint32_t values[4];
};

struct GpuReplayInstruction {
    uint32_t words[8];
};

struct GpuReplayStep {
    uint32_t program_index;
    uint32_t memory_start;
};

using RvrFieldBlock = GpuFieldBlock;
using RvrReplayInstruction = GpuReplayInstruction;
using RvrReplayStep = GpuReplayStep;

static_assert(sizeof(PreflightProgramEvent) == 8);
static_assert(sizeof(PreflightMemoryEvent) == 20);
static_assert(sizeof(PreflightInitialWrite) == 16);
static_assert(sizeof(GpuFieldBlock) == 16);
static_assert(sizeof(GpuReplayInstruction) == 32);
static_assert(sizeof(GpuReplayStep) == 8);

__device__ __forceinline__ uint32_t preflight_address_space(PreflightMemoryEvent const &event) {
    return event.address_space_and_kind & PREFLIGHT_ADDRESS_SPACE_MASK;
}

__device__ __forceinline__ bool preflight_is_write(PreflightMemoryEvent const &event) {
    return (event.address_space_and_kind & PREFLIGHT_WRITE_BIT) != 0;
}

__device__ __forceinline__ void preflight_encode_u8_block(
    uint32_t bytes,
    uint16_t (&out)[4]
) {
    out[0] = uint16_t(bytes);
    out[1] = uint16_t(bytes >> 16);
    out[2] = 0;
    out[3] = 0;
}

__device__ __forceinline__ void preflight_decode_u8_block(
    uint16_t const (&value)[4],
    uint8_t (&out)[4]
) {
    uint32_t bytes = uint32_t(value[0]) | (uint32_t(value[1]) << 16);
#pragma unroll
    for (uint32_t lane = 0; lane < 4; lane++) {
        out[lane] = uint8_t(bytes >> (8 * lane));
    }
}

__device__ __forceinline__ void preflight_set_error(uint32_t *error, uint32_t code) {
    atomicCAS(error, 0u, code);
}
