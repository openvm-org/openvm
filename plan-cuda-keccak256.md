# Plan: RV64 CUDA tracegen for keccak256

## Base branch

`rv64-cuda-easy-instructions`. No dependency on riscv load*/loadstore CUDA
branches.

## Context

The keccak256 extension has 3 CUDA chips: `keccakf_op`, `xorin`, `keccakf_perm`.
Only `keccakf_op` and `xorin` have RV32 references. `keccakf_perm` deals with
pure keccak permutation math (u64 lanes, u16 limbs) and has no register/memory
dependencies.

The CPU side is already fully migrated to RV64. The CUDA side still uses
`RV32_REGISTER_NUM_LIMBS` and `RV32_CELL_BITS`.

**Critical difference from deferral**: This is NOT a pure rename.
`DEFAULT_BLOCK_SIZE` changed from 4 to 8, which changes the number of memory
operations per keccak round. The CUDA column/record structs have structural
mismatches with the Rust side.

## Key structural change: memory operation count

`DEFAULT_BLOCK_SIZE` went from 4 (RV32) to 8 (RV64). Memory is accessed in
`DEFAULT_BLOCK_SIZE`-byte chunks, so:

| Constant | RV32 (CUDA current) | RV64 (Rust/target) |
|----------|--------------------|--------------------|
| keccakf_op word count | `KECCAK_WIDTH_BYTES / 4 = 50` | `KECCAK_WIDTH_BYTES / 8 = 25` |
| xorin word count | `KECCAK_RATE_BYTES / 4 = 34` | `KECCAK_RATE_BYTES / 8 = 17` |
| `MemoryWriteAuxCols` prev_data size | 4 | 8 |
| xorin `len_limbs` | `[T; 4]` array | single `T len_limb` |

This affects column struct sizes, record struct sizes, auxiliary column arrays,
and the memory read/write loop counts in the trace generation kernels.

Reference: SHA2 PR (#2749) handled the same `DEFAULT_BLOCK_SIZE` change by
updating `SHA2_READ_SIZE` and `SHA2_WRITE_SIZE` from 4 to 8, halving the
number of block/state reads.

## Constants strategy

Use the **same constant names** as the CPU side for readability. Two concepts
must be kept distinct:

1. **Pointer limbs** (`RV64_WORD_NUM_LIMBS = 4`): How many byte-limbs a u32
   pointer decomposes into. Used for `buffer_ptr_limbs`, `input_ptr_limbs`,
   range checks. Already in `constants.h`.

2. **Memory block size** (`DEFAULT_BLOCK_SIZE = 8`): How many bytes per memory
   read/write operation. Determines memory op counts. NOT yet in `constants.h`.

Currently each extension defines its own block size constant (deferral:
`MEMORY_OP_SIZE`, SHA2: `SHA2_READ_SIZE`, keccak: `KECCAK_WORD_SIZE`). We add a
shared `DEFAULT_BLOCK_SIZE` to `constants.h` and use CPU-matching derived names.

### Constant mapping (GPU → CPU reference)

| GPU constant (proposed) | Value | CPU reference |
|---|---|---|
| `DEFAULT_BLOCK_SIZE` (new in constants.h) | 8 | `openvm_circuit::arch::DEFAULT_BLOCK_SIZE` |
| `KECCAK_WIDTH_MEM_OPS` (replaces `KECCAK_WIDTH_WORDS`) | 25 | `constants.rs::KECCAK_WIDTH_MEM_OPS = KECCAK_WIDTH_BYTES / DEFAULT_BLOCK_SIZE` |
| `KECCAK_RATE_MEM_OPS` (replaces `XORIN_NUM_WORDS`) | 17 | `constants.rs::KECCAK_RATE_MEM_OPS = KECCAK_RATE_BYTES / DEFAULT_BLOCK_SIZE` |
| `RV64_WORD_NUM_LIMBS` (already in constants.h) | 4 | `riscv::RV64_WORD_NUM_LIMBS` |
| `RV64_CELL_BITS` (already in constants.h) | 8 | `riscv::RV64_CELL_BITS` |

Dead constants to remove from `constants.h`: `KECCAK_WORD_SIZE`,
`KECCAK_ABSORB_READS`, `KECCAK_DIGEST_WRITES` (defined but never used in any
kernel).

## Files to change

### 1. `crates/circuits/primitives/cuda/include/primitives/constants.h`

**a) Add `DEFAULT_BLOCK_SIZE` to the `program` namespace** (next to
`DEFAULT_PC_STEP`):
```cpp
namespace program {
inline constexpr size_t PC_BITS = 30;
inline constexpr size_t DEFAULT_PC_STEP = 4;
inline constexpr size_t DEFAULT_BLOCK_SIZE = 8;
} // namespace program
```

**b) Replace dead keccak adapter constants** with CPU-matching names:
```cpp
// Old (unused):
inline constexpr size_t KECCAK_WORD_SIZE = 4;
inline constexpr size_t KECCAK_ABSORB_READS = KECCAK_RATE_BYTES / KECCAK_WORD_SIZE;
inline constexpr size_t KECCAK_DIGEST_WRITES = KECCAK_DIGEST_BYTES / KECCAK_WORD_SIZE;

// New (matching CPU constants.rs):
inline constexpr size_t KECCAK_WIDTH_MEM_OPS = KECCAK_WIDTH_BYTES / program::DEFAULT_BLOCK_SIZE;
inline constexpr size_t KECCAK_RATE_MEM_OPS = KECCAK_RATE_BYTES / program::DEFAULT_BLOCK_SIZE;
```

### 2. `extensions/keccak256/circuit/cuda/include/keccakf_op.cuh`

**a) Replace `KECCAK_WIDTH_WORDS` with `KECCAK_WIDTH_MEM_OPS`**
```cpp
// Old: inline constexpr size_t KECCAK_WIDTH_WORDS = KECCAK_WIDTH_BYTES / RV32_REGISTER_NUM_LIMBS;
// New: use KECCAK_WIDTH_MEM_OPS from constants.h (= 25)
```
Remove the local definition; use the shared constant directly.

**b) `buffer_ptr_limbs`: rename constant**
```cpp
// Old: T buffer_ptr_limbs[RV32_REGISTER_NUM_LIMBS]
// New: T buffer_ptr_limbs[RV64_WORD_NUM_LIMBS]
```
Same array size (4), just rename. This is pointer decomposition, not memory
block size.

**c) `buffer_word_aux` arrays shrink from 50 to 25 entries**
All `[KECCAK_WIDTH_WORDS]` → `[KECCAK_WIDTH_MEM_OPS]` (now 25).

**d) `MemoryReadAuxRecord` / `MemoryBaseAuxCols`**
These are size-independent (just timestamps). No structural change needed,
only the array count changes.

### 3. `extensions/keccak256/circuit/cuda/include/xorin.cuh`

**a) Remove `XORIN_WORD_SIZE` and `XORIN_NUM_WORDS`**
Replace with shared constants:
- `XORIN_WORD_SIZE` → `DEFAULT_BLOCK_SIZE` (for memory block size = 8)
- `XORIN_WORD_SIZE` → `RV64_WORD_NUM_LIMBS` (for pointer limbs = 4)
- `XORIN_NUM_WORDS` → `KECCAK_RATE_MEM_OPS` (= 17)

Critical: these were conflated in RV32 (both = 4). In RV64 they diverge:
pointer limbs stay 4, memory block size becomes 8.

**b) `len_limbs[XORIN_WORD_SIZE]` → single `len_limb`**
CPU has `pub len_limb: T`. CUDA must match.

**c) `buffer_ptr_limbs`, `input_ptr_limbs`: use `RV64_WORD_NUM_LIMBS`**
These are pointer limbs (4 elements), not memory blocks.

**d) `MemoryWriteAuxCols<T, XORIN_WORD_SIZE>` → `MemoryWriteAuxCols<T, DEFAULT_BLOCK_SIZE>`**
prev_data changes from 4 to 8 bytes per write.

**e) All `[XORIN_NUM_WORDS]` arrays → `[KECCAK_RATE_MEM_OPS]`** (34 → 17)

### 4. `extensions/keccak256/circuit/cuda/src/keccakf_op.cu`

**a) Buffer pointer decomposition**
```cpp
// Old: uint8_t buffer_ptr_limbs[RV32_REGISTER_NUM_LIMBS] (4 bytes)
// New: uint8_t buffer_ptr_limbs[RV64_WORD_NUM_LIMBS] (still 4, just rename)
```

**b) Range check constants**
```cpp
// Old: RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS = 32 bits
// New: RV64_CELL_BITS * RV64_WORD_NUM_LIMBS = 32 bits (same value, rename)
```

**c) Memory read loops**
The loop that reads `KECCAK_WIDTH_WORDS` memory words changes from 50 to 25
iterations (`KECCAK_WIDTH_MEM_OPS`), each reading 8 bytes instead of 4.

### 5. `extensions/keccak256/circuit/cuda/src/xorin.cu`

**a) Range check constants**: same rename as keccakf_op

**b) Memory read/write loops**: use `KECCAK_RATE_MEM_OPS` (34 → 17 iterations)

**c) `len_limbs` handling**: adapt to single `len_limb` field

### 6. `extensions/keccak256/circuit/src/cuda/mod.rs`

- `RV32_CELL_BITS` → `RV64_CELL_BITS` (same value 8, pure rename)

### 7. `extensions/keccak256/circuit/src/extension/cuda.rs`

Already uses `Rv64*` types — verify no changes needed.

## Files NOT changing

- `keccakf_perm.cu`, `keccakf_perm.cuh`, `p3_keccakf.cuh` — pure keccak math
- `cuda_abi.rs` — pure FFI passthrough
- `extension/cuda.rs` — already RV64

## Testing

Existing CUDA tests in both `keccakf_op/tests.rs` and `xorin/tests.rs`:

**keccakf_op tests:**
- `test_keccakf_cuda_tracegen` — 3 random ops
- `test_keccakf_cuda_tracegen_single` — single op
- `test_keccakf_cuda_tracegen_zero_state` — zero state

**xorin tests:**
- `test_xorin_cuda_tracegen` — 5 random + 7 specific lengths
- `test_xorin_cuda_tracegen_single` — single 16-byte op

**Test updates needed**: The CUDA test helpers write registers as 4-byte `u32`
values (`(ptr as u32).to_le_bytes()`). These need to be updated to write 8-byte
`u64` values to match RV64 register width, similar to the `GpuChipTestBuilder`
fix in system-airs.

Also check that memory writes in tests use 8-byte blocks instead of 4-byte
blocks to match `DEFAULT_BLOCK_SIZE = 8`.

## Verification checklist

- [ ] `DEFAULT_BLOCK_SIZE = 8` added to `constants.h` `program` namespace
- [ ] `KECCAK_WIDTH_MEM_OPS` = 25 in constants.h (replaces dead `KECCAK_WORD_SIZE`)
- [ ] `KECCAK_RATE_MEM_OPS` = 17 in constants.h (replaces dead `KECCAK_ABSORB_READS`)
- [ ] Dead constants removed: `KECCAK_WORD_SIZE`, `KECCAK_ABSORB_READS`,
  `KECCAK_DIGEST_WRITES`
- [ ] keccakf_op.cuh uses `KECCAK_WIDTH_MEM_OPS` (not local `KECCAK_WIDTH_WORDS`)
- [ ] xorin.cuh uses `KECCAK_RATE_MEM_OPS` and `DEFAULT_BLOCK_SIZE` (not local
  `XORIN_WORD_SIZE` / `XORIN_NUM_WORDS`)
- [ ] `len_limbs[4]` → single `len_limb` in xorin cols (matching Rust)
- [ ] Pointer limbs use `RV64_WORD_NUM_LIMBS` (4), clearly distinct from
  `DEFAULT_BLOCK_SIZE` (8)
- [ ] `MemoryWriteAuxCols` prev_data size = `DEFAULT_BLOCK_SIZE` (8) in xorin
- [ ] All `RV32_*` constants renamed to `RV64_*` equivalents
- [ ] No remaining `RV32` references in `extensions/keccak256/circuit/`
- [ ] CUDA column struct sizes match Rust `width()` (verified by assert in kernel)
- [ ] CUDA test helpers write 8-byte registers, not 4-byte
- [ ] Memory read/write loops iterate correct count with correct block size
- [ ] All existing CUDA tests pass
- [ ] `cargo check --features cuda -p openvm-keccak256-circuit` compiles
