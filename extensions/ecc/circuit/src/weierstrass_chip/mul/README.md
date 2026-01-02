# EC_MUL Multirow Chip Documentation

## Overview

The EC_MUL multirow chip implements elliptic curve scalar multiplication using a double-and-add algorithm. Unlike single-row chips, this chip generates **257 trace rows per instruction**: 256 compute rows (one per scalar bit) plus 1 digest row for memory I/O.

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

- `mod.rs` - Module exports, constants, and chip constructors
- `columns.rs` - Column struct definitions for trace layout
- `air.rs` - AIR constraints
- `field_expr.rs` - FieldExpr for EC point operations
- `execution.rs` - Instruction execution logic
- `trace.rs` - Trace generation (Executor and Filler)

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

### Issue 1: FieldExpr Column Layout Mismatch (CRITICAL)

**Problem**: The AIR evaluates FieldExpr's `SubAir::eval()` on ALL rows, including digest rows. FieldExpr reads `is_valid` from offset `base_width`, but for digest rows, this offset contains `from_state.pc` (non-zero).

**Symptoms**:

- `assert_bool(is_valid)` fails (pc is not 0 or 1)
- `assert_bool(is_setup)` fails (garbage value)
- Range check interactions sent with wrong count
- STARK verification error: "out-of-domain evaluation mismatch"

**Root Cause**: Digest row's `from_state.pc` overlaps with FieldExpr's `is_valid` position.

**Suggested Fixes**:

1. **Add Padding Field (Recommended)**:

   ```rust
   pub struct EcMulDigestCols<...> {
       pub flags: EcMulFlagsCols<T>,
       pub control: EcMulControlCols<T, NUM_LIMBS>,
       pub field_expr_gate: T,  // Always 0, acts as FieldExpr's is_valid
       pub from_state: ExecutionState<T>,
       // ... rest of fields
   }
   ```

   This ensures `local[base_width] = 0` for digest rows.

2. **Gate mod-builder's Setup Constraint**:
   In `crates/circuits/mod-builder/src/builder.rs`, line 367:

   ```rust
   // Change from:
   builder.assert_bool(is_setup.clone());
   // To:
   builder.when(is_valid.clone()).assert_bool(is_setup.clone());
   ```

   This gates the constraint by `is_valid` so it passes when `is_valid = 0`.

3. **Generate FieldExpr Trace on Digest Row**:
   Generate FieldExpr columns with `is_valid = 0` on the digest row, then overlay digest-specific columns starting at `base_width + 1`. Requires adding the padding field.

### Issue 2: Point Addition to Infinity (POTENTIAL)

**Problem**: The current FieldExpr handles the case when R = ∞ (using safe denominators), but the formula `R_added = R_doubled + P` may not correctly compute `P` when adding `P` to infinity.

**Details**:

- When `is_inf = 1` and `bit = 1`, we want `R_next = P` (adding base point to infinity)
- Current formula: `R_doubled = 2*R` (with safe denom), `R_added = R_doubled + P`
- Since R = (0,0), the arithmetic produces garbage values (even with safe denominators)
- The output is `R_added` which is not equal to `P`

**Potential Fix**:

- Add explicit selection: `R_next = is_inf && bit ? P : (bit ? R_added : R_doubled)`
- Or ensure the FieldExpr arithmetic correctly handles the infinity case

**Note**: This may already be handled correctly by the native execution in `ec_mul` which uses proper curve libraries. The issue would only manifest in the AIR constraints if they don't match.

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

**Breakdown of Degree 7 Constraint**:

```
builder.when_transition()           // +1 (transition indicator)
       .when(both_in_instruction)   // +2 (degree 2 filter)
       .when(is_current)            // +2 (Lagrange polynomial)
       .assert_one(is_next);        // +2 (constraint with Lagrange)
                                    // = 7 total
```

**Workaround**:

```bash
cargo run --bin ecrecover -- --app-log-blowup 3
```

This sets `max_constraint_degree = (1 << 3) + 1 = 9`, sufficient for degree 7.

**Potential Fixes**:

1. **Reduce Encoder max_degree**: Using `max_degree=3` would require only 9 columns (vs 22) but Lagrange polynomials would be degree 3, making some constraints degree 8+
2. **Simplify row index checking**: Instead of checking all 257 row transitions individually, use a simpler increment-by-1 constraint
3. **Binary encoding**: Use 9 binary columns (log2(257)) with degree-1 constraints, but more columns and constraints overall
4. **Accept higher degree**: Document `--app-log-blowup 3` requirement and accept the performance tradeoff

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
