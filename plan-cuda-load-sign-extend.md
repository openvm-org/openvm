# Plan: RV64 CUDA tracegen for load_sign_extend

## Base branch

`rv64-cuda-system-airs` — provides the system-level GPU fixes (memory inventory,
boundary `BLOCKS_PER_CHUNK=1`, test infra 8-byte register writes) that all chip
CUDA work depends on. No dependency on `rv64-cuda-easy-instructions`.

## Context

The CPU side of `load_sign_extend` was updated to RV64 in PR #2715 (already on
`develop-v2.1.0-rv64` and `rv64-cuda-system-airs`). The CUDA tracegen kernel
still generates the old RV32 trace layout. The kernel must be rewritten to match
the new CPU `fill_trace_row` logic.

The loadstore adapter columns and record have the **same binary layout** between
RV32 and RV64 — both use 4 limbs for `rs1_data` and `u32` for `rs1_val` (upper
4 bytes of the 8-byte register are constrained to zero in the AIR, not stored).
The only adapter difference is a range-check constant (`>> 3` vs `>> 2`).

## Files to change (6 files)

### 1. `crates/circuits/primitives/cuda/include/primitives/constants.h`

Add RV64 constants after the RV32 block inside `namespace riscv`:

```cpp
inline constexpr size_t RV64_REGISTER_NUM_LIMBS = 8;
inline constexpr size_t RV64_WORD_NUM_LIMBS = 4;
inline constexpr size_t RV64_CELL_BITS = 8;
inline constexpr size_t RV64_IMM_AS = 0;
```

Note: `rv64-cuda-easy-instructions` adds the same lines. Merge conflict will be
trivial (identical content at the same location).

### 2. `extensions/riscv/circuit/cuda/include/rv32im/adapters/loadstore.cuh`

Add `Rv64LoadStoreAdapter`, `Rv64LoadStoreAdapterCols`, and
`Rv64LoadStoreAdapterRecord` **below** the existing `Rv32*` types. The struct
layouts are identical to `Rv32*` (both use `rs1_data[RV32_REGISTER_NUM_LIMBS]` /
`rs1_val: u32` — the RV64 adapter Cols uses `RV64_WORD_NUM_LIMBS = 4` for
`rs1_data`, which equals `RV32_REGISTER_NUM_LIMBS`).

Differences from the Rv32 version in `fill_trace_row`:
- Range check: `ptr_limbs[0] >> 3` with `RV64_CELL_BITS * 2 - 3` bits
  (was `>> 2` with `RV32_CELL_BITS * 2 - 2`)

Reference: CPU filler at `adapters/loadstore.rs:537-545`.

### 3. `extensions/riscv/circuit/cuda/src/load_sign_extend.cu` — main work

This is the core kernel rewrite. Every section changes:

**a) `LoadSignExtendCoreCols` — 3 flags to 7 flags**

Old (RV32):
```cpp
T opcode_loadb_flag0;
T opcode_loadb_flag1;
T opcode_loadh_flag;
```

New (RV64):
```cpp
T opcode_loadb_flag0;
T opcode_loadb_flag1;
T opcode_loadb_flag2;
T opcode_loadb_flag3;
T opcode_loadh_flag0;
T opcode_loadh_flag2;
T opcode_loadw_flag;
```

**b) `LoadSignExtendCoreRecord` — add `is_word`**

Old: `{ bool is_byte; uint8_t shift_amount; uint8_t read_data[N]; uint8_t prev_data[N]; }`
New: `{ bool is_byte; bool is_word; uint8_t shift_amount; uint8_t read_data[N]; uint8_t prev_data[N]; }`

IMPORTANT: field order must match the Rust `#[repr(C)]` struct exactly.

**c) `LoadSignExtendCore::fill_trace_row` — complete rewrite**

Port the CPU logic from `core.rs:312-348`. Key changes:

1. Compute `shift_most_sig_bit = (shift >> 2) & 1` and `inner_shift = shift & 3`
   (was: `shift & 2` to decide rotate-by-2 vs not)

2. Rotation: rotate `read_data` left by `shift_most_sig_bit * 4`
   (was: rotate left by 2 when `shift & 2`)

3. `most_sig_limb` selection:
   - byte: `shifted[inner_shift]`
   - word: `shifted[3]`
   - half: `shifted[inner_shift + 1]`
   (was: byte=`read_data[shift]`, half=`read_data[NUM_CELLS/2 - 1 + shift]`)

4. Set all 7 flags based on `(is_byte, is_half, is_word)` and `inner_shift`:
   ```
   is_half = !is_byte && !is_word
   loadb_flag0 = is_byte && inner_shift == 0
   loadb_flag1 = is_byte && inner_shift == 1
   loadb_flag2 = is_byte && inner_shift == 2
   loadb_flag3 = is_byte && inner_shift == 3
   loadh_flag0 = is_half && inner_shift == 0
   loadh_flag2 = is_half && inner_shift == 2
   loadw_flag  = is_word
   ```

**d) Wrapper types and kernel — rename Rv32 to Rv64**

```cpp
// Old
template <typename T> struct Rv32LoadSignExtendCols { ... RV32_REGISTER_NUM_LIMBS ... };
struct Rv32LoadSignExtendRecord { ... };
__global__ void rv32_load_sign_extend_tracegen(...);
extern "C" int _rv32_load_sign_extend_tracegen(...);

// New
template <typename T> struct Rv64LoadSignExtendCols { ... RV64_REGISTER_NUM_LIMBS ... };
struct Rv64LoadSignExtendRecord { ... };
__global__ void rv64_load_sign_extend_tracegen(...);
extern "C" int _rv64_load_sign_extend_tracegen(...);
```

Use `Rv64LoadStoreAdapter*` types from the updated `loadstore.cuh`.

### 4. `extensions/riscv/circuit/src/load_sign_extend/cuda.rs`

Rename types and constants:
- `Rv32LoadStoreAdapterCols` → `Rv64LoadStoreAdapterCols`
- `Rv32LoadStoreAdapterRecord` → `Rv64LoadStoreAdapterRecord`
- `RV32_REGISTER_NUM_LIMBS` → `RV64_REGISTER_NUM_LIMBS`
- `Rv32LoadSignExtendChipGpu` → `Rv64LoadSignExtendChipGpu`

### 5. `extensions/riscv/circuit/src/cuda_abi.rs` (load_sign_extend_cuda module only)

Rename the extern function:
- `_rv32_load_sign_extend_tracegen` → `_rv64_load_sign_extend_tracegen`

### 6. `extensions/riscv/circuit/src/extension/cuda.rs`

Update the two load_sign_extend references:
- `Rv32LoadSignExtendAir` → `Rv64LoadSignExtendAir`
- `Rv32LoadSignExtendChipGpu` → `Rv64LoadSignExtendChipGpu`

(All other chips remain `Rv32*` — they haven't had their CUDA updated yet.)

## Testing

### Current state

`load_sign_extend/tests.rs` (542 lines) has CPU-only tests:
- `rand_load_sign_extend_test` — randomized test for LOADB/LOADH/LOADW (100 ops each)
- Specific shift tests: `positive_loadb_shift7_test`, `positive_loadh_shift6_test`,
  `positive_loadw_shift4_test`
- Sanity tests for sign/zero extension per opcode
- Negative tests for alignment violations and invalid flag pranks

There are **zero CUDA tests**. No `#[cfg(feature = "cuda")]` blocks exist.

### What to add

Follow the pattern from `base_alu/tests.rs` on `rv64-cuda-easy-instructions`.
Add a `#[cfg(feature = "cuda")]` section at the end of `tests.rs`:

**1. CUDA imports block**

```rust
#[cfg(feature = "cuda")]
use {
    crate::{
        adapters::Rv64LoadStoreAdapterRecord,
        LoadSignExtendCoreRecord, Rv64LoadSignExtendChipGpu,
    },
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};
```

**2. `GpuHarness` type alias**

```rust
#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv64LoadSignExtendExecutor,
    Rv64LoadSignExtendAir,
    Rv64LoadSignExtendChipGpu,
    Rv64LoadSignExtendChip<F>,
>;
```

**3. `create_cuda_harness` function**

Reuses the existing `create_harness_fields` to build the AIR, executor, and CPU
chip. Constructs the GPU chip separately. Key detail: `create_harness_fields`
takes `Arc<VariableRangeCheckerChip>` (CPU type), obtainable from
`tester.range_checker().cpu_chip.clone().unwrap()`.

```rust
#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let range_checker_cpu = tester.range_checker().cpu_chip.clone().unwrap();
    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        range_checker_cpu,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = Rv64LoadSignExtendChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}
```

**4. CUDA test function**

One randomized test covering all three opcodes (matching the CPU
`rand_load_sign_extend_test`). The existing `set_and_execute` works with
`GpuChipTestBuilder` because it takes `&mut impl TestBuilder<F>`.

```rust
#[cfg(feature = "cuda")]
#[test_case(LOADB, 100)]
#[test_case(LOADH, 100)]
#[test_case(LOADW, 100)]
fn test_cuda_rand_load_sign_extend_tracegen(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();

    let mut harness = create_cuda_harness(&tester);
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester, &mut harness.executor, &mut harness.dense_arena,
            &mut rng, opcode, None, None, None, None,
        );
    }

    type Record<'a> = (
        &'a mut Rv64LoadStoreAdapterRecord,
        &'a mut LoadSignExtendCoreRecord<RV64_REGISTER_NUM_LIMBS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64LoadStoreAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
```

### What the test does

`load_gpu_harness` calls `load_and_compare` which:
1. Runs `gpu_chip.generate_proving_ctx(dense_arena)` — invokes our CUDA kernel
2. Runs `cpu_chip.generate_proving_ctx(matrix_arena)` — runs Rust `fill_trace_row`
3. Asserts every element of the GPU trace matches the CPU trace

Then `finalize().simple_test()` runs the full STARK prove+verify with
periphery chips (range checker, memory, etc.).

### No new test types needed

The randomized test with 100 ops per opcode covers all shift values
probabilistically (shift 0..7 for LOADB, {0,2,4,6} for LOADH, {0,4} for LOADW).
The existing CPU-only tests for specific shifts, sanity, and negative cases
don't need GPU variants — they test execution/constraint logic, not tracegen.

## Verification checklist

- [ ] CUDA `LoadSignExtendCoreCols` field order matches Rust `#[repr(C)]` struct
- [ ] CUDA `LoadSignExtendCoreRecord` field order matches Rust `#[repr(C)]` struct
- [ ] CUDA `Rv64LoadStoreAdapterCols` / `Record` layout matches Rust
- [ ] `fill_trace_row` logic produces identical output to CPU `TraceFiller::fill_trace_row`
- [ ] Range check in adapter uses `>> 3` with `RV64_CELL_BITS * 2 - 3` bits
- [ ] Kernel function name in `.cu` matches `extern "C"` name in `cuda_abi.rs`
- [ ] `extension/cuda.rs` compiles (only load_sign_extend lines changed)
