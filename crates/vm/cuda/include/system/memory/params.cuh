#pragma once

// Memory-layout constants on the CUDA side. Mirrors the CPU-side constants in
// `openvm_circuit::arch::config` and `openvm_circuit::system::memory::controller`.
//
// Terminology:
//   Cell    one storage word in an address space (u16 for RV64 ASes, Fp for
//           DEFERRAL_AS).
//   Block   the unit of one memory-bus message: BLOCK_FE_WIDTH cells =
//           MEMORY_BLOCK_BYTES bytes.
//   Digest  the output of one Poseidon2 compression (DIGEST_WIDTH cells); also
//           one merkle leaf.
//   Leaf    one merkle-tree leaf = one Poseidon2 half = DIGEST_WIDTH cells =
//           BLOCKS_PER_LEAF blocks.
//
// u16-celled AS layout (RV64 register/memory/public-values).
// One merkle leaf = 16 bytes = 8 u16 cells = 2 bus blocks:
//
//   byte_ptr:     0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
//                ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
//   u8 storage:  │b0 │b1 │b2 │b3 │b4 │b5 │b6 │b7 │b8 │b9 │b10│b11│b12│b13│b14│b15│
//                └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
//                ╰──────╯╰──────╯╰──────╯╰──────╯╰──────╯╰──────╯╰──────╯╰──────╯   u16 cells (LE)
//   ptr:             0       1       2       3       4       5       6       7
//   bus_ptr:         0       2       4       6       8      10      12      14
//                ╰─────────── block 0 ──────────╯╰─────────── block 1 ──────────╯
//                ╰────────────────── one merkle leaf / digest ──────────────────╯
//
//   byte_ptr = U16_CELL_SIZE * ptr     (coincides with bus_ptr for u16 ASes)
//   bus_ptr  = BUS_PTR_SCALE  * ptr
//   block stride on bus: BUS_BLOCK_STRIDE = 8  = BUS_PTR_SCALE * BLOCK_FE_WIDTH
//   leaf  stride on bus: BUS_LEAF_STRIDE  = 16 = BUS_PTR_SCALE * DIGEST_WIDTH
//
// F-celled AS layout (DEFERRAL_AS). Each cell holds one Fp element
// (size_of::<F>() bytes on host; 4 for BabyBear). The bus addresses cells
// directly, so cell / block / leaf counts use the same pointer units as the u16 AS.
//
//   byte_ptr:       0       4       8       12      16      20      24      28
//                ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
//   F storage:   │  F0   │  F1   │  F2   │  F3   │  F4   │  F5   │  F6   │  F7   │
//                └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
//   ptr:        0       1       2       3       4       5       6       7
//   bus_ptr:         0       2       4       6       8      10      12      14
//                ╰─────────── block 0 ──────────╯╰─────────── block 1 ──────────╯
//                ╰────────────────── one merkle leaf / digest ──────────────────╯
//
//   byte_ptr = size_of::<F>() * ptr     (host-memory stride; ≠ bus_ptr)
//   bus_ptr  = BUS_PTR_SCALE   * ptr    (same formula as u16 ASes)
//   block stride on bus: BUS_BLOCK_STRIDE = 8  = BUS_PTR_SCALE * BLOCK_FE_WIDTH
//   leaf  stride on bus: BUS_LEAF_STRIDE  = 16 = BUS_PTR_SCALE * DIGEST_WIDTH
//   The bus addresses cells, not bytes.

#include "poseidon2.cuh" // brings in CELLS / CELLS_OUT from stark-backend

// Cells per Poseidon2 half (and per merkle leaf).
inline constexpr size_t DIGEST_WIDTH = CELLS_OUT;
// Cells per Poseidon2 permutation input.
inline constexpr size_t POSEIDON2_WIDTH = CELLS;

// Bytes per memory-bus block (one RV64 8-byte word pair).
inline constexpr size_t MEMORY_BLOCK_BYTES = 8;
// Normalized memory-bus pointer scale: bus_ptr = BUS_PTR_SCALE * ptr.
inline constexpr size_t BUS_PTR_SCALE_BITS = 1;
inline constexpr size_t BUS_PTR_SCALE = 1 << BUS_PTR_SCALE_BITS;

// Cells per memory-bus block.
inline constexpr size_t BLOCK_FE_WIDTH = MEMORY_BLOCK_BYTES / BUS_PTR_SCALE;
// Blocks per merkle leaf.
inline constexpr size_t BLOCKS_PER_LEAF = DIGEST_WIDTH / BLOCK_FE_WIDTH;
// Bus-pointer delta between consecutive blocks.
inline constexpr size_t BUS_BLOCK_STRIDE = BUS_PTR_SCALE * BLOCK_FE_WIDTH;
// Bus-pointer delta between consecutive merkle leaves.
inline constexpr size_t BUS_LEAF_STRIDE = BUS_PTR_SCALE * DIGEST_WIDTH;

// Host byte width of one u16-celled storage cell.
inline constexpr size_t U16_CELL_SIZE = 2;

// Address space ID for the F-celled deferral AS. Matches
// `openvm_instructions::DEFERRAL_AS`.
inline constexpr uint32_t DEFERRAL_AS = 4;
