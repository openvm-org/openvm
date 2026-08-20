#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "system/memory/params.cuh"

// CUDA mirror of the host-side block-index value helper in
// `openvm_riscv_circuit::adapters` (see `extensions/riscv/circuit/src/adapters/mod.rs`).
// The heap AIRs convert a guest *byte* pointer into its memory-bus block index with a
// range-checked quotient; the trace side only registers the matching range-check counts.

// Bit width of the low block-index limb (`byte_lo / MEMORY_BLOCK_BYTES < 2^13`). Mirrors
// `BLOCK_INDEX_Q_BITS` in `openvm_riscv_circuit::adapters`.
inline constexpr size_t BLOCK_INDEX_Q_BITS = openvm::U16_BITS - 3;
static_assert(MEMORY_BLOCK_BYTES == 1 << 3);

// Mirrors `add_block_index_range_checks`: registers the quotient and high-limb range-check
// counts for one aligned heap base byte pointer.
__device__ __forceinline__ void add_block_index_range_checks(
    VariableRangeChecker &range_checker,
    uint32_t byte_ptr,
    size_t byte_ptr_max_bits
) {
    range_checker.add_count((byte_ptr & 0xffffu) / MEMORY_BLOCK_BYTES, BLOCK_INDEX_Q_BITS);
    range_checker.add_count(
        byte_ptr >> openvm::U16_BITS, uint32_t(byte_ptr_max_bits) - openvm::U16_BITS
    );
}
