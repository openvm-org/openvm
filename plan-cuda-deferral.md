# Plan: RV64 CUDA tracegen for deferral

## Base branch

`rv64-cuda-easy-instructions` (or whichever riscv CUDA branch is merged first).
Deferral doesn't depend on loadstore/load_sign_extend CUDA changes.

## Context

The deferral extension has 4 CUDA chips: `call`, `output`, `count`, `poseidon2`.
Only `call` and `output` have RV32 references — they use `RV32_REGISTER_NUM_LIMBS`
and `RV32_CELL_BITS` for pointer/register handling. The CPU side was already
updated to RV64 (uses `RV64_WORD_NUM_LIMBS` and `RV64_CELL_BITS`).

**Key insight**: `RV32_REGISTER_NUM_LIMBS = RV64_WORD_NUM_LIMBS = 4` and
`RV32_CELL_BITS = RV64_CELL_BITS = 8`. The values are identical — this is purely
a constant rename with no logic changes. The struct layouts and computations
remain the same.

**Compilation is currently broken**: `extension/cuda.rs` imports
`Rv32DeferralConfig` which was renamed to `Rv64DeferralConfig` on the CPU side.
The crate does not compile with `--features cuda`.

## Files to change (5 files)

### 1. `extensions/deferral/circuit/cuda/src/call.cu`

Rename constants only — no logic changes:
- `RV32_REGISTER_NUM_LIMBS` → `RV64_WORD_NUM_LIMBS` (6 occurrences)
- `RV32_CELL_BITS` → `RV64_CELL_BITS` (2 occurrences)

These appear in:
- `DeferralCallAdapterRecord`: `rd_val[RV32_REGISTER_NUM_LIMBS]`,
  `rs_val[RV32_REGISTER_NUM_LIMBS]`
- `DeferralCallAdapterCols`: same fields
- `limb_shift_bits` calculation (2 places)
- `output_len[RV32_REGISTER_NUM_LIMBS - 1]` range check

### 2. `extensions/deferral/circuit/cuda/src/output.cu`

Same rename pattern:
- `RV32_REGISTER_NUM_LIMBS` → `RV64_WORD_NUM_LIMBS` (in header record/cols)
- `RV32_CELL_BITS` → `RV64_CELL_BITS` (in limb_shift calculation)

### 3. `extensions/deferral/circuit/src/call/cuda.rs`

- `RV32_CELL_BITS` → `RV64_CELL_BITS` (import and usage for
  `BitwiseOperationLookupChipGPU`)

### 4. `extensions/deferral/circuit/src/output/cuda.rs`

- Same as call: `RV32_CELL_BITS` → `RV64_CELL_BITS`

### 5. `extensions/deferral/circuit/src/extension/cuda.rs`

- `Rv32DeferralConfig` → `Rv64DeferralConfig` (import and VmConfig type)
- `Rv32DeferralGpuBuilder` → `Rv64DeferralGpuBuilder` (struct name)

## Files NOT changing

- `count.cu`, `poseidon2.cu` — no RV32/RV64 references
- `count/cuda.rs`, `poseidon2/cuda.rs` — no RV32/RV64 references
- `cuda_abi.rs` — no RV32/RV64 references
- `def_types.h`, `def_poseidon2_buffer.cuh`, `canonicity.cuh` — no changes

## Testing

Deferral already has CUDA tracegen tests on the develop branch:
- `call/tests.rs`: `test_cuda_rand_deferral_call_tracegen`
- `output/tests.rs`: `test_cuda_rand_deferral_output_tracegen`

These tests already use `Rv64*` types (the test code was updated during the CPU
RV64 migration). They just need `--features cuda` to compile, which requires
fixing the `Rv32DeferralConfig` import in `extension/cuda.rs`.

After the constant renames:
1. `cargo check --features cuda -p openvm-deferral-circuit` should compile
2. Existing CUDA tests should pass without modification

## Verification checklist

- [ ] `RV32_REGISTER_NUM_LIMBS` → `RV64_WORD_NUM_LIMBS` in all `.cu` files
  (same value 4, purely rename)
- [ ] `RV32_CELL_BITS` → `RV64_CELL_BITS` in all `.cu` and `cuda.rs` files
  (same value 8, purely rename)
- [ ] `Rv32DeferralConfig` → `Rv64DeferralConfig` in `extension/cuda.rs`
- [ ] `Rv32DeferralGpuBuilder` → `Rv64DeferralGpuBuilder` in `extension/cuda.rs`
- [ ] No remaining `Rv32`/`RV32` references in `extensions/deferral/circuit/`
- [ ] `cargo check --features cuda -p openvm-deferral-circuit` compiles
- [ ] Existing CUDA tests pass: `cargo test --features cuda -p
  openvm-deferral-circuit`
- [ ] Struct layouts unchanged (same binary representation, just constant names)
