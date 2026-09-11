#pragma once

#include <cstdint>

// System metadata stores these tags as u8; postflight structs store them as u32.
enum MemoryCellType : uint32_t {
    CELL_UNSUPPORTED = 0,
    CELL_U8 = 1,
    CELL_U16 = 2,
    CELL_FIELD32 = 3,
};

// Memory-layout constants on the CUDA side. Mirrors the CPU-side constants in
// `openvm_circuit::arch::config` and `openvm_circuit::system::memory::controller`.
//
// Terminology:
//   Cell    one storage word in an address space.
//   Block   the unit of one memory-bus message: BLOCK_FE_WIDTH cells. Its host byte width
//           depends on the cell layout (4 for U8, 8 for U16, 16 for Field32).
//   Digest  the output of one Poseidon2 compression (DIGEST_WIDTH cells); also
//           one merkle leaf.
//   Leaf    one merkle-tree leaf = one Poseidon2 half = DIGEST_WIDTH cells =
//           BLOCKS_PER_LEAF blocks.
//
// U8-celled AS layout (public values): one block is 4 bytes and one leaf is 8 bytes.
// The pointer counts U8 cells and is therefore also a byte pointer.
//
// U16-celled AS layout (RV64 register/memory).
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
// Field32-celled AS layout. Each cell holds one Fp element
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
// In every AS, ptr counts cells. Memory-bus block index k starts at ptr k * BLOCK_FE_WIDTH, and
// merkle leaf l starts at ptr l * DIGEST_WIDTH.

#include "poseidon2.cuh" // brings in CELLS / CELLS_OUT from stark-backend
#include "primitives/constants.h"
#include <cstddef>

// log2 of the number of cells per Poseidon2 half.
inline constexpr size_t DIGEST_WIDTH_BITS = 3;
// Cells per Poseidon2 half (and per merkle leaf).
inline constexpr size_t DIGEST_WIDTH = 1 << DIGEST_WIDTH_BITS;
static_assert(DIGEST_WIDTH == CELLS_OUT);
// Cells per Poseidon2 permutation input.
inline constexpr size_t POSEIDON2_WIDTH = CELLS;

// Host byte width of one u16-celled storage cell.
inline constexpr size_t U16_CELL_SIZE = 1 << openvm::U16_CELL_SIZE_BITS;

// Cells per memory-bus block.
inline constexpr size_t BLOCK_FE_WIDTH = 4;
// Byte width of one RV64 u16 register/memory block.
inline constexpr size_t MEMORY_BLOCK_BYTES = BLOCK_FE_WIDTH * U16_CELL_SIZE;
// Blocks per merkle leaf.
inline constexpr size_t BLOCKS_PER_LEAF = DIGEST_WIDTH / BLOCK_FE_WIDTH;

// Upper bound on a subtree's `base_height`, i.e. on the number of bottom merkle levels that may be
// omitted from its buffer. Sizes the thread-local scratch of `recompute_omitted_node`, which
// rebuilds an omitted node from `2^base_height` raw-memory leaves. The host picks the actual
// number of omitted levels per subtree (`OMITTED_BOTTOM_LEVELS` in `merkle_tree/mod.rs`).
inline constexpr size_t MAX_OMITTED_LEVELS = 3;
