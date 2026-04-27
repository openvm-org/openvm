# Plan: RV64 CUDA tracegen for loadstore

## Base branch

`rv64-cuda-load-sign-extend` — includes the Rv64LoadStoreAdapter CUDA types in
`loadstore.cuh` that this kernel also uses, plus all the easy-instructions Rust
glue fixes.

## Context

The CPU side of `loadstore` was updated to RV64 in PR #2714. The CUDA kernel
still generates the old RV32 trace layout. The key changes:

- **Opcode set changed**: RV32 had 8 opcodes (`LOADW, LOADBU, LOADHU, STOREW,
  STOREH, STOREB, LOADB, LOADH`) all handled by this one chip. RV64 splits them:
  this chip handles 8 unsigned/store opcodes (`LOADD, LOADBU, LOADHU, LOADWU,
  STORED, STOREW, STOREH, STOREB`), while `LOADB, LOADH, LOADW` (sign-extending
  loads) moved to load_sign_extend. The enum ordering also changed.
- **NUM_CELLS**: 4 → 8 (registers are 8 bytes).
- **Selector encoding**: RV32 used 4 boolean `flags[4]`. RV64 uses an `Encoder`
  with 7-wide `selector[7]` to encode 30 `(opcode, shift)` cases. The CUDA kernel
  must replicate the encoder's point-generation logic.
- **`write_data` computation**: expanded for 8-byte loads/stores with shifts 0..7.
- **`is_load`**: now `matches!(opcode, LOADD | LOADWU | LOADHU | LOADBU)`.

The adapter (`Rv64LoadStoreAdapter*`) is already implemented in `loadstore.cuh`
from the load_sign_extend PR.

## Files to change (4 files)

### 1. `extensions/riscv/circuit/cuda/src/loadstore.cu` — main work

**a) `LoadStoreCoreCols` — flags[4] → selector[7]**

Old (RV32):
```cpp
T flags[4];
T is_valid;
T is_load;
T read_data[NUM_CELLS];
T prev_data[NUM_CELLS];
T write_data[NUM_CELLS];
```

New (RV64):
```cpp
T selector[LOADSTORE_SELECTOR_WIDTH];  // LOADSTORE_SELECTOR_WIDTH = 7
T is_valid;
T is_load;
T read_data[NUM_CELLS];
T prev_data[NUM_CELLS];
T write_data[NUM_CELLS];
```

**b) `LoadStoreCoreRecord` — same shape, just NUM_CELLS=8**

The Rust record is unchanged in structure:
```cpp
template <size_t NUM_CELLS> struct LoadStoreCoreRecord {
    uint8_t local_opcode;
    uint8_t shift_amount;
    uint8_t read_data[NUM_CELLS];
    uint32_t prev_data[NUM_CELLS];
};
```

IMPORTANT: field order must match the Rust `#[repr(C)]` struct exactly.

**c) Opcode enum — update to Rv64**

Old:
```cpp
enum Rv32LoadStoreOpcode { LOADW, LOADBU, LOADHU, STOREW, STOREH, STOREB, LOADB, LOADH };
```

New (must match Rust enum ordering exactly):
```cpp
enum Rv64LoadStoreOpcode { LOADD, LOADBU, LOADHU, LOADWU, STORED, STOREW, STOREH, STOREB };
```

Note: LOADB, LOADH, LOADW are excluded — they are sign-extending loads handled
by the load_sign_extend chip. This chip only handles the first 8 opcodes
(through STOREB).

**d) Encoder logic — use existing CUDA `Encoder` from `primitives/encoder.cuh`**

There is already a CUDA `Encoder` implementation in
`crates/circuits/primitives/cuda/include/primitives/encoder.cuh` that mirrors
the Rust `Encoder`. It is used in production by `sha2_hasher.cu`.

Usage pattern (from sha2):
```cpp
#include "primitives/encoder.cuh"

Encoder encoder(30, 2, true);  // matches Rust Encoder::new(30, 2, true)
uint32_t case_idx = from_opcode_shift(opcode, shift);
encoder.write_flag_pt(row.slice_from(COL_INDEX(Cols, selector)), case_idx);
```

The kernel needs a helper function that maps `(opcode, shift)` → case index
(0..29), mirroring `InstructionCase::from_opcode_shift` in Rust. This can be
a device function with a switch statement.

**e) `fill_trace_row` — rewrite**

Port the CPU logic from `core.rs`. Key changes:

1. Compute `write_data` via `run_write_data()` — expanded for 8-byte ops:
   - `LOADD`: `write_data[i] = read_data[i]` (full 8-byte copy)
   - `LOADWU`: `write_data[i] = read_data[i+shift]` for `i < 4`, else 0
   - `LOADHU`: `write_data[i] = read_data[i+shift]` for `i < 2`, else 0
   - `LOADBU`: `write_data[0] = read_data[shift]`, rest 0
   - `STORED`: `write_data[i] = read_data[i]` (full 8-byte copy)
   - `STOREW`: splice 4 bytes from `read_data` into `prev_data` at `shift`
   - `STOREH`: splice 2 bytes from `read_data` into `prev_data` at `shift`
   - `STOREB`: `write_data = prev_data` with `write_data[shift] = read_data[0]`

2. Set `is_valid = 1`, `is_load` based on opcode
3. Set `selector` from the encoder point table
4. Write `read_data`, `prev_data`, `write_data`

**f) Wrapper types and kernel — rename Rv32 to Rv64**

```cpp
Rv32LoadStoreCols → Rv64LoadStoreCols  (uses RV64_REGISTER_NUM_LIMBS)
Rv32LoadStoreRecord → Rv64LoadStoreRecord
rv32_load_store_tracegen → rv64_load_store_tracegen
_rv32_load_store_tracegen → _rv64_load_store_tracegen
```

Use `Rv64LoadStoreAdapter*` types from `loadstore.cuh` (already added by
load_sign_extend PR).

### 2. `extensions/riscv/circuit/src/loadstore/cuda.rs`

Update constants (same pattern as load_sign_extend):
- `RV32_REGISTER_NUM_LIMBS` → `RV64_REGISTER_NUM_LIMBS`
- Add `use std::mem::size_of;`

The type renames (`Rv32*` → `Rv64*`) are already done by easy-instructions.

### 3. `extensions/riscv/circuit/src/cuda_abi.rs` (loadstore_cuda module only)

Rename the extern function:
- `_rv32_load_store_tracegen` → `_rv64_load_store_tracegen`

### 4. `extensions/riscv/circuit/src/loadstore/tests.rs`

Update existing CUDA tests (already present but using old Rv32 types):

- Imports: `Rv32LoadStoreAdapterRecord` → `Rv64LoadStoreAdapterRecord`,
  `Rv32LoadStoreChipGpu` → `Rv64LoadStoreChipGpu`, etc.
- `RV32_REGISTER_NUM_LIMBS` → `RV64_REGISTER_NUM_LIMBS`
- `Rv32LoadStoreExecutor` → `Rv64LoadStoreExecutor` in GpuHarness type
- Add new opcodes to test_case list: `LOADD, STORED, LOADWU`
- Re-enable tests by changing `#[cfg(all(test, any()))]` → `#[cfg(test)]` in
  `mod.rs`
- Remove stale `RV32_REGISTER_AS` import in the CUDA mem_config setup (should
  use default config which already has RV64 settings)

## Encoder details

The CUDA `Encoder` in `primitives/encoder.cuh` already handles the math. The
only thing the loadstore kernel needs is the `(opcode, shift)` → case index
mapping. The 30 cases in `InstructionCase::ALL` order are:

```
 0: LoadD0      10: LoadBu3    20: StoreH4
 1: LoadWu0     11: LoadBu4    21: StoreH6
 2: LoadWu4     12: LoadBu5    22: StoreB0
 3: LoadHu0     13: LoadBu6    23: StoreB1
 4: LoadHu2     14: LoadBu7    24: StoreB2
 5: LoadHu4     15: StoreD0    25: StoreB3
 6: LoadHu6     16: StoreW0    26: StoreB4
 7: LoadBu0     17: StoreW4    27: StoreB5
 8: LoadBu1     18: StoreH0    28: StoreB6
 9: LoadBu2     19: StoreH2    29: StoreB7
```

A `__device__` helper function maps `(Rv64LoadStoreOpcode opcode, uint8_t shift)`
to one of these indices, matching `InstructionCase::from_opcode_shift` in Rust.

## Testing

The loadstore tests already have a CUDA section that needs updating:
- Fix `Rv32*` → `Rv64*` type references
- Add `LOADD`, `STORED`, `LOADWU` to the `test_case` list (currently only has
  the 6 RV32 opcodes)
- The `set_and_execute` function already supports all RV64 opcodes

## Verification checklist

- [ ] CUDA `LoadStoreCoreCols` field order matches Rust `#[repr(C)]` struct
  (selector[7] not flags[4])
- [ ] CUDA `LoadStoreCoreRecord` field order matches Rust `#[repr(C)]` struct
- [ ] CUDA `Rv64LoadStoreOpcode` enum values match Rust enum ordering exactly
- [ ] Encoder selector points match CPU `Encoder::get_flag_pt` output for all
  30 cases
- [ ] `run_write_data` logic handles all 8 opcodes × valid shifts correctly
- [ ] `is_load` matches: `LOADD | LOADWU | LOADHU | LOADBU`
- [ ] Kernel function name in `.cu` matches `extern "C"` name in `cuda_abi.rs`
- [ ] All 8 opcodes handled by this chip tested in CUDA test_case list
