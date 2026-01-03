# EC_MUL Multirow Chip Documentation

## Overview

The EC_MUL multirow chip implements elliptic curve scalar multiplication using a double-and-add algorithm. Unlike single-row chips, this chip generates **257 trace rows per instruction**: 256 compute rows (one per scalar bit) plus 1 digest row for memory I/O.

> **Note**: Each source file in this module (`mod.rs`, `columns.rs`, `air.rs`, `execution.rs`, `trace.rs`, `field_expr.rs`) contains detailed Rust doc comments that complement this README. Use `cargo doc` or browse the source files directly for implementation-level documentation.

## Quick Reference

| Aspect | Value |
|--------|-------|
| Rows per instruction | 257 (256 compute + 1 digest) |
| Scalar size | 256 bits |
| Supported curves | secp256k1, P-256, BN254, BLS12_381 |
| Opcodes | `EC_MUL`, `SETUP_EC_MUL` |
| Constraint degree | 7 (requires `--app-log-blowup 3`) |

## Benchmarking

For performance benchmarks using real-world workloads (e.g., Ethereum block execution), see the branch `manh/ec-mul-bench` on the [openvm-reth-benchmark](https://github.com/openvm-org/openvm-reth-benchmark) repository.

## Instruction Entry Point

### Transpiler

The EC_MUL instruction is defined in `extensions/ecc/transpiler/src/lib.rs`:

```rust
pub enum Rv32WeierstrassOpcode {
    EC_ADD_NE,
    SETUP_EC_ADD_NE,
    EC_DOUBLE,
    SETUP_EC_DOUBLE,
    EC_MUL,         // Scalar multiplication
    SETUP_EC_MUL,   // Setup for scalar multiplication
}
```

The transpiler (`EccTranspilerExtension`) translates RISC-V custom instructions to these opcodes based on `funct7`:

- `SwBaseFunct7::SwEcMul` → `EC_MUL`
- `SwBaseFunct7::SwSetupMul` → `SETUP_EC_MUL`

### Extension Registration

In `extensions/ecc/circuit/src/extension/weierstrass.rs`:

1. **Executor Registration**: `WeierstrassExtensionExecutor::EcMulRv32_32` or `EcMulRv32_48`
2. **AIR Registration**: `EcMulAir<NUM_LIMBS, NUM_BLOCKS, BLOCK_SIZE>`
3. **Chip Registration**: `EcMulChip` via `get_ec_mul_chip()`

Each curve in the configuration gets its own EC_MUL chip instance with `offset = CLASS_OFFSET + curve_index * COUNT`.

### Instruction Format

```text
EC_MUL rd, rs1, rs2
  - rd:  Register containing destination pointer (where result is written)
  - rs1: Register containing base point pointer (Px, Py)
  - rs2: Register containing scalar pointer (256-bit scalar)
```

## Architecture

### Row Types

1. **Compute Rows (rows 0-255)**: Each row processes one bit of the 256-bit scalar using FieldExpr:
   - Performs point doubling: `R_doubled = 2 * R`
   - Performs point addition: `R_added = R_doubled + P`
   - Selects result based on scalar bit: `R_next = bit ? R_added : R_doubled`

2. **Digest Row (row 256)**: Handles all memory I/O:
   - Reads 3 register pointers (rd, rs1, rs2)
   - Reads scalar (32 bytes) and base point (2 coordinates)
   - Writes result point (2 coordinates)
   - Sends execution bridge interaction

### Algorithm

Uses MSB-first double-and-add:

```text
R = ∞ (point at infinity)
for i in 255..=0:
    R = 2R
    if scalar_bit[i] == 1:
        R = R + P
return R
```

## File Structure

Each file contains detailed Rust doc comments explaining its purpose and implementation details:

| File | Purpose | Key Doc Sections |
|------|---------|------------------|
| `mod.rs` | Module exports, constants, chip constructors | Module overview, usage examples, current status |
| `columns.rs` | Column struct definitions for trace layout | Layout diagrams, known issues, width calculations |
| `air.rs` | AIR constraints | Constraint overview, known issues, transition logic |
| `field_expr.rs` | FieldExpr for EC point operations | Input/output mapping, flag semantics |
| `execution.rs` | Instruction execution logic | Execution flow, setup vs normal mode, curve dispatch |
| `trace.rs` | Trace generation (Executor and Filler) | Record layout, filling phases, timing notes |

## Column Layout

### Compute Row Layout

```text
[EcMulFlagsCols | EcMulControlCols | FieldExpr columns...]
     ^                  ^                    ^
     |                  |                    |
  flags like        scalar bits,         is_valid, inputs,
  is_compute_row,   base point limbs,    vars, q_limbs,
  is_digest_row,    accumulator limbs    carry_limbs, flags
  row_idx, etc.
```

### Digest Row Layout

```text
[EcMulFlagsCols | EcMulControlCols | from_state | pointers | aux cols | data...]
     ^                  ^               ^
     |                  |               |
  Same as compute    Same as compute   ExecutionState (pc, timestamp)
```

### Total Width

```rust
total_width = max(compute_width, digest_width)
compute_width = base_width + expr_width
digest_width = size_of(EcMulDigestCols)
base_width = size_of(EcMulFlagsCols) + size_of(EcMulControlCols)
```

## Key Components

### EcMulFlagsCols

- `is_compute_row`: 1 for rows 0-255
- `is_digest_row`: 1 for row 256
- `is_first_compute_row`: 1 only for row 0
- `is_setup`: 1 for SETUP_EC_MUL opcode
- `is_inf`: 1 when accumulator R is at infinity
- `row_idx[22]`: Encoded row index using Encoder with max_degree=2

### EcMulControlCols

- `scalar_bits[256]`: The 256-bit scalar
- `base_px_limbs`, `base_py_limbs`: Base point P coordinates
- `rx_limbs`, `ry_limbs`: Current accumulator R coordinates

### FieldExpr (ec_mul_step_expr)

- **Inputs**: (Rx, Ry, Px, Py)
- **Flags**:
  - `flags[0]`: scalar bit (0 or 1)
  - `flags[1]`: use_safe_denom (1 when is_setup OR is_inf)
- **Outputs**: (R_next_x, R_next_y)

## Execution Layer (execution.rs)

The execution layer implements `InterpreterExecutor` and `InterpreterMeteredExecutor` traits for `EcMulExecutor`. It handles the VM-side computation before trace generation.

### Execution Flow

```text
1. Pre-compute Phase (pre_compute_impl)
   ├── Parse instruction fields (rd, rs1, rs2, d, e)
   ├── Validate address spaces (d=REGISTER_AS, e=MEMORY_AS)
   ├── Determine if this is SETUP_EC_MUL or EC_MUL
   └── Store pre-computed data in EcMulPreCompute struct

2. Dispatch Phase (dispatch! macro)
   ├── Identify curve type from (modulus, a_coeff)
   │   - K256 (secp256k1)
   │   - P256
   │   - BN254
   │   - BLS12_381
   └── Select appropriate handler with const generics

3. Execution Phase (execute_e12_impl)
   ├── Read register pointers (rs_vals)
   ├── Read point_data (NUM_BLOCKS * BLOCK_SIZE bytes)
   ├── Read scalar_data (32 bytes)
   ├── If IS_SETUP:
   │   ├── Validate prime matches expected modulus
   │   ├── Validate a_coeff matches expected value
   │   ├── Validate scalar matches expected curve order
   │   └── Run FieldExpr 256 times (run_ec_mul_expr_for_setup)
   ├── Else (normal EC_MUL):
   │   └── Call native ec_mul() from curves.rs
   ├── Write output_data to memory
   └── Increment PC
```

### Key Structures

#### EcMulPreCompute

```rust
struct EcMulPreCompute<'a> {
    expr: &'a FieldExpr,         // Reference to the FieldExpr for this chip
    scalar_biguint: &'a BigUint, // Expected curve order (for setup validation)
    rs_addrs: [u8; 2],           // Register addresses [rs1, rs2]
    a: u8,                       // Destination register address (rd)
    flag_idx: u8,                // FieldExpr flag index (for multi-op chips)
}
```

#### Curve Type Dispatch

The `dispatch!` macro creates monomorphized handler functions for each curve type:

```rust
match (is_setup, curve_type) {
    (true, CurveType::K256)  => execute_impl<..., K256, true>,
    (false, CurveType::K256) => execute_impl<..., K256, false>,
    // ... other curves
}
```

This enables the compiler to generate optimized code paths for each curve.

### Setup Mode vs Normal Mode

| Aspect | EC_MUL (Normal) | SETUP_EC_MUL |
|--------|-----------------|--------------|
| **Input** | Base point (Px, Py) | Curve params (prime, a_coeff) |
| **Scalar** | Actual scalar value | Curve order |
| **Validation** | None | Prime, a_coeff, scalar must match |
| **Computation** | Native `ec_mul()` | FieldExpr 256 iterations |
| **Purpose** | Compute k*P | Validate curve config |

### run_ec_mul_expr_for_setup

For setup mode, we must run the FieldExpr to produce output consistent with what the trace filler generates:

```rust
fn run_ec_mul_expr_for_setup(...) -> [[u8; BLOCK_SIZE]; NUM_BLOCKS] {
    // Start at infinity
    let mut rx = BigUint::ZERO;
    let mut ry = BigUint::ZERO;
    
    // Run 256 iterations (MSB first)
    for bit in scalar_bits {
        let inputs = vec![rx, ry, px, py];
        let flags = vec![bit, true]; // is_setup_flag = true
        let vars = expr.execute(inputs, flags);
        rx = vars[output_indices[0]].clone();
        ry = vars[output_indices[1]].clone();
    }
    
    // Convert to output format
    ...
}
```

### Metered Execution (E2)

The metered execution path (`execute_e2_impl`) additionally:

1. Reports height change: `ctx.on_height_change(chip_idx, EC_MUL_TOTAL_ROWS)`
   - This tells the VM that this instruction generates 257 rows
   - Required for proper trace matrix sizing

### Native Curve Libraries

The `ec_mul()` function in `curves.rs` dispatches to native implementations:

- **K256**: `k256::ProjectivePoint` (secp256k1)
- **P256**: `p256::ProjectivePoint` (NIST P-256)
- **BN254**: `halo2curves_axiom::bn256::G1`
- **BLS12_381**: `halo2curves_axiom::bls12_381::G1`

## Trace Generation Flow

1. **Execution** (`EcMulExecutor`):
   - Reads register pointers
   - Reads scalar and base point from memory
   - Performs native EC multiplication using k256 library
   - Writes result to memory
   - Records all data for trace filling

2. **Trace Filling** (`EcMulFiller`):
   - Parses the record from trace buffer
   - **Important**: Copies record data BEFORE zeroing trace (timestamp fix)
   - Fills 256 compute rows with FieldExpr witnesses
   - Fills digest row with memory I/O data

## Known Issues

### Issue 1: FieldExpr Evaluation & Interaction Mismatch (CRITICAL)

**Problem**: The AIR evaluates FieldExpr's `SubAir::eval()` on ALL rows (257 per instruction), but trace generation only calls FieldExpr's `generate_subrow()` for compute rows (256 per instruction). This causes two critical issues:

1. **Constraint failures**: FieldExpr reads `is_valid` from offset `base_width`, which contains garbage (`from_state.pc`) for digest rows
2. **Range check interaction mismatch**: FieldExpr sends range check interactions gated by `is_valid`, but trace only generates requests for compute rows

**Interaction Mismatch Details**:

| Component | Behavior |
|-----------|----------|
| **AIR (all 257 rows)** | FieldExpr sends range check interactions with multiplicity = `is_valid` (see `builder.rs` lines 436-443) |
| **Digest row in AIR** | `is_valid = from_state.pc` (garbage, e.g., 0x1000), sends garbage multiplicity |
| **Trace (256 rows only)** | Only calls `generate_subrow()` for compute rows; `range_checker.add_count()` called 256 times |
| **Result** | Grand product mismatch: AIR sends extra interactions for digest row, trace doesn't match |

**Constraint Failure Details**:

- `assert_bool(is_valid)` fails (pc is not 0 or 1)
- `assert_bool(is_setup)` fails where `is_setup = is_valid - flags[0] - flags[1]` (garbage)
- STARK verification error: "out-of-domain evaluation mismatch"

**Root Cause**: Digest row's `from_state.pc` overlaps with FieldExpr's `is_valid` column position at offset `base_width`.

**Fix Options**:

**Option A: Generate dummy FieldExpr on digest row (Recommended)**

Since `is_valid` is designed to gate padding/dummy rows (not instruction rows), and the digest row is an active part of the EC_MUL instruction, we should:

1. Call `generate_subrow()` on digest row with dummy inputs and `is_valid = 0`
2. This generates matching range check requests for the FieldExpr columns
3. Overlay digest-specific columns starting AFTER the FieldExpr columns

```rust
// In trace.rs, for digest row:
// 1. Generate dummy FieldExpr columns (is_valid = 0)
let dummy_inputs = vec![BigUint::ZERO; 4];  // Rx, Ry, Px, Py
let dummy_flags = vec![false, false];        // bit, is_setup
self.ec_mul_step_expr.generate_subrow(
    (self.range_checker.as_ref(), dummy_inputs, dummy_flags),
    &mut row[compute_base_width..compute_base_width + expr_width],
);
// Set is_valid = 0 manually
row[compute_base_width] = F::ZERO;

// 2. Digest-specific columns start after FieldExpr columns
let digest_start = compute_base_width + expr_width;
// ... fill digest columns at digest_start offset
```

**Column Layout with Option A**:

```text
Compute Row:
[EcMulFlagsCols | EcMulControlCols | FieldExpr (is_valid=1) | padding...]

Digest Row:
[EcMulFlagsCols | EcMulControlCols | FieldExpr (is_valid=0) | DigestSpecific...]
```

Total width = base_width + expr_width + digest_specific_width (larger but correct)

**Option B: Add padding field (Simpler but semantically awkward)**

Add a padding field so `is_valid = 0` for digest rows without generating FieldExpr:

```rust
pub struct EcMulDigestCols<...> {
    pub flags: EcMulFlagsCols<T>,
    pub control: EcMulControlCols<T, NUM_LIMBS>,
    pub field_expr_gate: T,  // Always 0, acts as FieldExpr's is_valid
    pub from_state: ExecutionState<T>,
    // ... rest of fields
}
```

This works but treats digest row as if it were a padding row, which is semantically incorrect. The trace doesn't call `generate_subrow()` for digest row, so no range check requests are generated - but this is fine because `is_valid = 0` means multiplicity = 0 in the AIR.

**Trade-offs**:

| Aspect | Option A (Dummy FieldExpr) | Option B (Padding Field) |
|--------|---------------------------|--------------------------|
| Semantic correctness | ✅ Digest row is an active row | ⚠️ Treats digest as padding |
| Trace width | Larger (base + expr + digest) | Smaller (max of compute/digest) |
| Implementation | More complex | Simpler |
| Range check requests | Generated with is_valid=0 | Not generated |

### Issue 2: Point Addition to Infinity (CRITICAL)

**Problem**: When adding P to infinity (R = ∞), the FieldExpr produces garbage instead of P, causing memory bridge verification failure.

**What's Already Implemented**:

- The `is_inf` flag exists in `EcMulFlagsCols` and is tracked correctly through compute rows
- In the AIR: `use_safe_denom = is_setup OR is_inf` (line 544)
- In trace filler: `flags[1] = is_setup || is_inf` (lines 516-520)
- Safe denominators (1) are used when `is_inf = 1` to avoid division by zero

**What's Still Broken**:

- Safe denominators prevent division by zero but **don't fix the output values**
- When R = (0,0) with safe denom: `R_doubled = (a², -a³)` (garbage, not infinity)
- Then `R_added = garbage + P` = more garbage, not P
- The output selection `bit ? R_added : R_doubled` returns garbage either way

**Root Cause Flow**:

1. Execution: Native `ec_mul()` writes correct result to memory (e.g., P for scalar with MSB=1)
2. Memory controller: Records the correct result
3. Trace filler: Computes `result_data` using FieldExpr (garbage when is_inf=1 and bit=1)
4. Memory bridge: Sends write interaction with garbage data
5. Grand product check fails because interactions don't match

**Fix Required**: Modify output selection in `field_expr.rs` to handle infinity explicitly:

```rust
// After computing ax, ay, dx, dy, add is_inf_flag as a 3rd flag:
let is_inf_flag = (*builder).borrow_mut().new_flag();

// When is_inf = 1 and bit = 1, output should be P (base point), not R_added
// When is_inf = 1 and bit = 0, output should stay at "infinity" (0,0 representation)
let use_base_point = is_inf_flag && bit_flag;  // Need to output P directly
let out_x = select(use_base_point, px, select(bit_flag, ax, dx));
let out_y = select(use_base_point, py, select(bit_flag, ay, dy));
```

The AIR and trace filler would need to pass `is_inf` as a separate flag (flag 2) to FieldExpr.

### Issue 3: Adding Two Equal Points - P = D (POTENTIAL)

**Problem**: When the doubled point D has the same x-coordinate as the base point P, the addition formula has `Px - Dx = 0` in the denominator.

**Details**:

- After doubling R to get D, we compute `D + P` using the addition formula
- Addition formula has denominator `Px - Dx`
- If `Px = Dx` (the doubled point has same x as base point), division by zero occurs
- This can happen for specific (curve, scalar, base point) combinations
- Currently, safe denominators are only used when `is_setup OR is_inf`
- If P = D occurs during normal computation, the constraint will fail

**Potential Fixes**:

1. **Add `points_equal` flag**: Detect when `Px = Dx` and use doubling formula instead
2. **Use complete addition formulas**: E.g., from Renes-Costello-Batina (more expensive)
3. **Assume non-occurrence**: Document that this is undefined behavior for edge cases

**Note**: For random scalars on standard curves, this is statistically unlikely but not impossible.

### Issue 4: High Constraint Degree (CONFIGURATION)

**Problem**: The Encoder with `max_degree=2` generates constraints up to degree 7, requiring `app_log_blowup >= 3`.

**Current Implementation**:

- Encoder with `max_degree=2`: Uses 22 columns (since C(24,2) = 276 >= 257)
- `contains_flag()` returns a degree-2 Lagrange polynomial
- Encoder's internal `eval()` has degree-3 constraints (falling factorial)
- Row transition loop (lines 250-263) creates 256 individual degree-7 constraints

**Constraint Degree Analysis**:

| Constraint Location | Description | Degree |
|---------------------|-------------|--------|
| Encoder `falling_factorial` | `x*(x-1)*(x-2)` for each of 22 vars | 3 |
| Encoder `falling_factorial(sum)` | `(sum)*(sum-1)*(sum-2)` | 3 |
| Encoder `sum_of_unused` | Lagrange polynomials | 2 |
| `contains_flag` calls | Lagrange polynomial for single point | 2 |
| `both_in_instruction` | `is_compute * (next_is_compute + next_is_digest)` | 2 |
| Row index transition | `when_transition * when(both_in_instruction) * when(is_current) * is_next` | **7** |
| FieldExpr constraints | Depends on expression complexity | ~3-5 |

**Breakdown of Degree 7 Constraint** (lines 258-262):

```rust
builder.when_transition()           // +1 (transition indicator)
       .when(both_in_instruction)   // +2 (degree 2 expression)
       .when(is_current)            // +2 (Lagrange polynomial from contains_flag)
       .assert_one(is_next);        // +2 (Lagrange polynomial)
                                    // = 7 total
```

**Workaround**:

```bash
cargo run --bin ecrecover -- --app-log-blowup 3
```

This sets `max_constraint_degree = (1 << 3) + 1 = 9`, sufficient for degree 7.

**Potential Fixes**:

1. **Replace Encoder with simple increment** (Recommended):
   - Replace `row_idx: [T; 22]` (Encoder) with single `row_idx: T` column
   - Constrain: `when(in_instruction).assert_eq(next.row_idx, local.row_idx + 1)` → degree 3
   - Range-check: `row_idx ∈ [0, 256]` via VariableRangeChecker
   - Constrain: `is_first_compute → row_idx = 0`, `is_digest → row_idx = 256`
   - **Result**: Max degree drops from 7 to ~4-5, columns drop from 22 to 1

2. **Reduce Encoder max_degree to 3** (Not recommended):
   - Would need 10 columns (C(13,3) = 286 >= 257)
   - Lagrange polynomials become degree 3
   - Transition constraints become degree 1+2+3+3 = **9** (worse!)

3. **Binary encoding**:
   - Use 9 binary columns (log2(512) covers 0-256)
   - Each column boolean: degree 2 to check
   - Increment requires carry logic: more constraints but degree 2-3
   - **Result**: 9 columns (vs 22), max degree ~3-4

4. **Accept higher degree** (Current approach):
   - Document `--app-log-blowup 3` requirement
   - Accept the 8x proof size increase for this chip
   - Simpler to implement, but performance cost

**Recommendation**: Fix #1 is the best approach - replacing Encoder with a simple row counter eliminates 256 degree-7 constraints and reduces column count from 22 to 1.

## AIR Constraints Summary

### Row Type Selectors

- `is_compute_row` and `is_digest_row` are boolean
- `is_valid = is_compute_row + is_digest_row` (0 for padding rows)
- `is_first_compute_row = (row_idx == 0) AND is_compute_row`
- Encoder constraints for `row_idx`

### Compute Row Constraints

- FieldExpr SubAir evaluation (point doubling + addition + selection)
- Input linking: FieldExpr inputs = control column values
- Flag linking: `flags[0] = scalar_bit`, `flags[1] = is_setup OR is_inf`
- `is_inf` propagation: `is_inf_next = is_inf * (1 - bit)`

### State Transitions

- Accumulator update: `next.rx = outputs_x`, `next.ry = outputs_y`
- Constant propagation: scalar_bits, base_px, base_py, is_setup
- First compute row: `rx = 0`, `ry = 0` (infinity)
- Last compute to digest: output equals result_data

### Digest Row Constraints

- Memory I/O: register reads, scalar read, point reads, result writes
- Execution bridge interaction
- Scalar bit decomposition
- Base point limb decomposition

## Testing

Run with increased max constraint degree:

```bash
cargo run --bin ecrecover -- --app-log-blowup 3
```

The `--app-log-blowup 3` sets `max_constraint_degree = 9`, which is needed because the chip has degree 7 constraints (from Encoder).

## Dependencies

- `openvm-mod-circuit-builder`: FieldExpr for modular arithmetic
- `openvm-circuit-primitives`: Encoder, range checking
- `openvm-ecc-transpiler`: Opcode definitions
- `k256`: Native secp256k1 operations for execution
- `halo2curves_axiom`: Alternative curve implementations

## Further Reading

### Source Code Comments

Each source file contains comprehensive Rust doc comments. To explore:

```bash
# Generate and open HTML documentation
cargo doc --package openvm-ecc-circuit --open

# Or read source files directly - each has a module-level doc comment
# explaining the file's purpose, structure, and known issues
```

### Related Components

- **Transpiler**: `extensions/ecc/transpiler/src/lib.rs` - Opcode definitions
- **Extension**: `extensions/ecc/circuit/src/extension/weierstrass.rs` - Chip registration
- **Mod-builder**: `crates/circuits/mod-builder/src/builder.rs` - FieldExpr infrastructure
- **Curves**: `extensions/ecc/circuit/src/weierstrass_chip/curves.rs` - Native curve operations
