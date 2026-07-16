#pragma once

#include "primitives/encoder.cuh"
#include "system/memory/params.cuh"

// Number of byte offsets within one memory block.
constexpr uint32_t NUM_BYTE_SHIFTS = 2 * BLOCK_FE_WIDTH;
// Number of columns in the byte-shift selector encoding.
constexpr size_t BYTE_SHIFT_SELECTOR_WIDTH = 3;
// Maximal degree of the load/store shift-selector flag expressions.
constexpr uint32_t SHIFT_SELECTOR_MAX_DEGREE = 2;

// Encodes one selector case per byte offset; the zero point represents an invalid row.
__device__ inline Encoder shift_encoder() {
    return Encoder(
        NUM_BYTE_SHIFTS, SHIFT_SELECTOR_MAX_DEGREE, true, BYTE_SHIFT_SELECTOR_WIDTH
    );
}
