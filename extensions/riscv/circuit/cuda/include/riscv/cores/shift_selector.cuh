#pragma once

#include "primitives/encoder.cuh"
#include "system/memory/params.cuh"

// Byte shifts of an effective pointer inside an 8-byte memory block; every load/store core
// encodes shift `i` as selector case `i`. Mirrors `NUM_BYTE_SHIFTS` in
// `extensions/riscv/circuit/src/adapters/mod.rs`.
constexpr uint32_t NUM_BYTE_SHIFTS = 2 * BLOCK_FE_WIDTH;
// Maximal degree of the load/store shift-selector flag expressions.
constexpr uint32_t SHIFT_SELECTOR_MAX_DEGREE = 2;

// Selector encoder shared by all load/store cores: one case per byte shift, with the zero
// point reserved for invalid rows. Mirrors `shift_encoder` in
// `extensions/riscv/circuit/src/adapters/mod.rs`.
__device__ inline Encoder shift_encoder(size_t selector_width) {
    return Encoder(
        NUM_BYTE_SHIFTS, SHIFT_SELECTOR_MAX_DEGREE, true, (uint32_t)selector_width
    );
}
