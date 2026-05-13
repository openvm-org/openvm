#pragma once

// Single source of truth for memory-layout constants on the CUDA side.
// Mirrors the CPU-side constants in `openvm_circuit::arch::config` and
// `openvm_circuit::system::memory::controller`. Keep names in sync with the
// Rust side — any mismatch is a bug.
//
//   `DIGEST_WIDTH`        cells per merkle leaf and per Poseidon2 half
//   `POSEIDON2_WIDTH`     cells per Poseidon2 permutation input
//   `MEMORY_BLOCK_BYTES`  bytes per guest-visible memory access (= RV64 word
//                         pair). Stays constant across the u16 cell flip.
//   `BUS_PTR_SCALE`       normalized memory-bus pointer scale: bus_ptr =
//                         BUS_PTR_SCALE * cell_idx.
//   `BLOCK_FE_WIDTH`      field elements (= cells) per memory bus message =
//                         MEMORY_BLOCK_BYTES / BUS_PTR_SCALE.
//   `BLOCKS_PER_LEAF`     memory bus messages per merkle leaf =
//                         DIGEST_WIDTH / BLOCK_FE_WIDTH.
//   `BUS_BLOCK_STRIDE`    bus-pointer delta between consecutive bus messages =
//                         BUS_PTR_SCALE * BLOCK_FE_WIDTH.
//   `BUS_LEAF_STRIDE`     bus-pointer delta between consecutive merkle leaves
//                         = BUS_PTR_SCALE * DIGEST_WIDTH.
//   `U16_CELL_SIZE`       host byte width of a u16-celled storage cell (= 2).
//   `DEFERRAL_AS`         address space index for the F-celled deferral AS.
//                         Matches `openvm_instructions::DEFERRAL_AS`.

#include "poseidon2.cuh" // brings in CELLS / CELLS_OUT from stark-backend

inline constexpr size_t DIGEST_WIDTH = CELLS_OUT; // 8
inline constexpr size_t POSEIDON2_WIDTH = CELLS;  // 16

inline constexpr size_t MEMORY_BLOCK_BYTES = 8;
inline constexpr size_t BUS_PTR_SCALE = 2;

inline constexpr size_t BLOCK_FE_WIDTH = MEMORY_BLOCK_BYTES / BUS_PTR_SCALE;
inline constexpr size_t BLOCKS_PER_LEAF = DIGEST_WIDTH / BLOCK_FE_WIDTH;
inline constexpr size_t BUS_BLOCK_STRIDE = BUS_PTR_SCALE * BLOCK_FE_WIDTH;
inline constexpr size_t BUS_LEAF_STRIDE = BUS_PTR_SCALE * DIGEST_WIDTH;

inline constexpr size_t U16_CELL_SIZE = 2;

inline constexpr uint32_t DEFERRAL_AS = 4;

static_assert(BLOCK_FE_WIDTH * BUS_PTR_SCALE == MEMORY_BLOCK_BYTES, "memory layout invariant");
static_assert(BLOCKS_PER_LEAF * BLOCK_FE_WIDTH == DIGEST_WIDTH, "merkle layout invariant");
