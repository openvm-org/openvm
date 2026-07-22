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
    uint32_t value[4];
};

struct PreflightInitialWrite {
    uint32_t address_space;
    uint32_t pointer;
    uint32_t initial_value[4];
};

struct RvrReplayInstruction {
    uint32_t words[8];
};

struct RvrReplayStep {
    uint32_t program_index;
    uint32_t memory_start;
};

static_assert(sizeof(PreflightProgramEvent) == 8);
static_assert(sizeof(PreflightMemoryEvent) == 28);
static_assert(sizeof(PreflightInitialWrite) == 24);
static_assert(sizeof(RvrReplayInstruction) == 32);
static_assert(sizeof(RvrReplayStep) == 8);

__device__ __forceinline__ uint32_t preflight_address_space(PreflightMemoryEvent const &event) {
    return event.address_space_and_kind & PREFLIGHT_ADDRESS_SPACE_MASK;
}

__device__ __forceinline__ bool preflight_is_write(PreflightMemoryEvent const &event) {
    return (event.address_space_and_kind & PREFLIGHT_WRITE_BIT) != 0;
}

__device__ __forceinline__ void preflight_set_error(uint32_t *error, uint32_t code) {
    atomicCAS(error, 0u, code);
}
