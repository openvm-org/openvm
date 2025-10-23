# ProofIdx SubAir

## Overview

This SubAir ensures that each proof is properly marked with a start flag.

## Columns

- **is_enabled**: Flag indicating whether the row is enabled for the current proof_idx
- **proof_idx**: Current proof index
- **is_proof_start**: Boolean flag marking the first row of a proof

**Note:**
- The `is_enabled` flag is NOT constrained to be boolean by this SubAir. It is the caller's responsibility if they want to ensure `is_enabled` is boolean.
- The `is_proof_start` flag is NOT constrained on disabled rows (when `is_enabled = 0`).
- The `proof_idx` does not necessarily start at 0.
- The `proof_idx` column either stays the same or increases by 1.
- Even for disabled proofs, dummy rows are included with appropriate `proof_idx` values.
- Padding rows are not zero-filled; their `proof_idx` values form a non-decreasing sequence from the last enabled row (e.g., if the last enabled row has `proof_idx = k`, padding rows can have `proof_idx = k+1, k+1, k+2, k+3, ...`).

## Constraints

Let $\Delta\text{proof\\_idx} = \text{proof\\_idx}\_{\text{next}} - \text{proof\\_idx}\_{\text{local}}$
```rust
let proof_idx_diff = next.proof_idx - local.proof_idx;
```

### 1. Base Constraints

#### `proof_idx` increments by 0 or 1

$$
\Delta\text{proof\\_idx} \in \\\{0, 1\\\}\quad \text{(transition)} \qquad (1)
$$
```rust
builder
    .when_transition()
    .assert_bool(proof_idx_diff.clone());
```

### 2. Boundary Constraints

#### First row enabled implies proof start

$$
\text{is\\_enabled} \Rightarrow \text{is\\_proof\\_start} = 1\quad \text{(first row)} \qquad (2)
$$
```rust
builder
    .when_first_row()
    .when(local.is_enabled)
    .assert_one(local.is_proof_start);
```

### 3. Proof Constraints

#### 3.1. Within Proof ($\Delta\text{proof\\_idx} \neq 1$)

At transition rows, $\Delta\text{proof\\_idx} \neq 1 \iff \Delta\text{proof\\_idx} = 0$ by (1), indicating we are within the same proof.

##### Enabled state consistency

$$
\text{is\\_enabled}\_{\text{local}} = \text{is\\_enabled}\_{\text{next}}\quad (\Delta\text{proof\\_idx} \neq 1,\ \text{transition}) \qquad (3)
$$
```rust
builder
    .when_transition()
    .when_ne(proof_idx_diff.clone(), AB::Expr::ONE)
    .assert_eq(local.is_enabled, next.is_enabled);
```

##### No proof start within proof

$$
\text{is\\_enabled}\_{\text{local}} \Rightarrow \text{is\\_proof\\_start}\_{\text{next}} = 0\quad (\Delta\text{proof\\_idx} \neq 1,\ \text{transition}) \qquad (4)
$$
```rust
builder
    .when_transition()
    .when_ne(proof_idx_diff.clone(), AB::Expr::ONE)
    .when(local.is_enabled)
    .assert_zero(next.is_proof_start);
```

#### 3.2. At Proof Boundaries ($\Delta\text{proof\\_idx} \neq 0$)

When $\Delta\text{proof\\_idx} \neq 0$, we are at a proof boundary.

##### Enabled next row implies proof start

$$
\text{is\\_enabled}\_{\text{next}} \Rightarrow \text{is\\_proof\\_start}\_{\text{next}} = 1\quad (\Delta\text{proof\\_idx} \neq 0) \qquad (5)
$$
```rust
builder
    .when(proof_idx_diff)
    .when(next.is_enabled)
    .assert_one(next.is_proof_start);
```

## Case Analysis

### 1. Single Row ($\text{is\\_enabled} = 1$)

- By (2): $\text{is\\_proof\\_start} = 1$

Single enabled row is properly marked as start.

### 2. Single Proof ($\Delta\text{proof\\_idx} \neq 1$ for all transition rows, $\text{is\\_enabled} = 1$)

- By (2): $\text{is\\_proof\\_start} = 1$ on first row
- By (4): $\text{is\\_proof\\_start} = 0$ on all interior rows

Proof is properly marked with start on first row only.

### 3. Multiple Proofs

#### 3.1. Within Proof Transitions ($\Delta\text{proof\\_idx} \neq 1$, $\text{is\\_enabled}\_{\text{local}} = 1$)

- By (3): $\text{is\\_enabled}\_{\text{next}} = 1$
- By (4): $\text{is\\_proof\\_start}\_{\text{next}} = 0$

Interior enabled rows maintain enabled state with no start marker.

#### 3.2. At Proof Boundaries ($\Delta\text{proof\\_idx} \neq 0$, $\text{is\\_enabled}\_{\text{next}} = 1$)

- By (5): $\text{is\\_proof\\_start}\_{\text{next}} = 1$

Any transition to an enabled row properly marks it with start flag.
