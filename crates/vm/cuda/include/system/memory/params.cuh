#pragma once

// Memory-layout constants on the CUDA side. Mirrors the CPU-side constants in
// `openvm_circuit::arch::config` and `openvm_circuit::system::memory::controller`.
//
// Terminology:
//   Cell    one storage word in an address space.
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
//                ╰─────────── block 0 ──────────╯╰─────────── block 1 ──────────╯
//                ╰────────────────── one merkle leaf / digest ──────────────────╯
//
//   byte_ptr = U16_CELL_SIZE * ptr
//
// F-celled AS layout (DEFERRAL_AS). Each cell holds one Fp element
// (size_of::<F>() bytes on host; 4 for BabyBear).
//
//   byte_ptr:       0       4       8       12      16      20      24      28
//                ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
//   F storage:   │  F0   │  F1   │  F2   │  F3   │  F4   │  F5   │  F6   │  F7   │
//                └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
//   ptr:        0       1       2       3       4       5       6       7
//                ╰─────────── block 0 ──────────╯╰─────────── block 1 ──────────╯
//                ╰────────────────── one merkle leaf / digest ──────────────────╯
//
//   byte_ptr = size_of::<F>() * ptr
//
// In every AS, ptr is AS-native. Block k starts at ptr k * BLOCK_FE_WIDTH, and
// merkle leaf l starts at ptr l * DIGEST_WIDTH.

#include "poseidon2.cuh" // brings in CELLS / CELLS_OUT from stark-backend
#include <cstddef>
#include <cstdint>

// Cells per Poseidon2 half (and per merkle leaf).
inline constexpr size_t DIGEST_WIDTH = CELLS_OUT;
// Cells per Poseidon2 permutation input.
inline constexpr size_t POSEIDON2_WIDTH = CELLS;

// Bytes per memory-bus block.
inline constexpr size_t MEMORY_BLOCK_BYTES = 8;

// Host byte width of one u16-celled storage cell.
inline constexpr size_t U16_CELL_SIZE = 2;

// Cells per memory-bus block.
inline constexpr size_t BLOCK_FE_WIDTH = MEMORY_BLOCK_BYTES / U16_CELL_SIZE;
// Blocks per merkle leaf.
inline constexpr size_t BLOCKS_PER_LEAF = DIGEST_WIDTH / BLOCK_FE_WIDTH;

// Number of little-endian 16-bit limbs used to represent an AS-native memory pointer on the
// memory bus.
inline constexpr size_t POINTER_LIMBS = 2;
inline constexpr size_t POINTER_LIMB_BITS = 16;

// log2 of DIGEST_WIDTH (= CELLS_OUT = 8).
inline constexpr size_t DIGEST_WIDTH_BITS = 3;

// Number of low bits of a leaf label kept in the low limb so that
// `low * DIGEST_WIDTH` fits in one 16-bit memory-bus pointer limb.
inline constexpr size_t LOW_LEAF_BITS = POINTER_LIMB_BITS - DIGEST_WIDTH_BITS;

// Number of bottom merkle levels omitted from large GPU subtree buffers.
inline constexpr size_t OMITTED_BOTTOM_LEVELS = 3;

// Address space ID for the F-celled deferral AS. Matches
// `openvm_instructions::DEFERRAL_AS`.
inline constexpr uint32_t DEFERRAL_AS = 4;
