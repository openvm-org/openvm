# ProofIdx SubAir

## Overview

This SubAir ensures that each proof is properly marked with start and end flags.

## Columns

- **proof_idx**: Current proof index
- **is_enabled**: Boolean flag indicating whether the row is enabled for the current proof_idx
- **is_proof_start**: Boolean flag marking the first row of a proof
- **is_proof_end**: Boolean flag marking the last row of a proof

**Note:**
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

#### Boolean `is_enabled` flag

$$
\text{is\\_enabled} \in \\\{0, 1\\\} \qquad (1)
$$
```rust
builder.assert_bool(local.is_enabled);
```

#### `proof_idx` increments by 0 or 1

$$
\Delta\text{proof\\_idx} \in \\\{0, 1\\\}\quad \text{(transition)} \qquad (2)
$$
```rust
builder
    .when_transition()
    .assert_bool(proof_idx_diff.clone());
```

### 2. Boundary Constraints

#### First row enabled implies proof start

$$
\text{is\\_enabled} \Rightarrow \text{is\\_proof\\_start} = 1\quad \text{(first row)} \qquad (3)
$$
```rust
builder
    .when_first_row()
    .when(local.is_enabled)
    .assert_one(local.is_proof_start);
```

#### Last row enabled implies proof end

$$
\text{is\\_enabled} \Rightarrow \text{is\\_proof\\_end} = 1\quad \text{(last row)} \qquad (4)
$$
```rust
builder
    .when_last_row()
    .when(local.is_enabled)
    .assert_one(local.is_proof_end);
```

### 3. Disabled Row Constraints

#### No proof start on disabled rows

$$
\neg\text{is\\_enabled} \Rightarrow \text{is\\_proof\\_start} = 0 \qquad (5)
$$
```rust
builder
    .when_ne(local.is_enabled, AB::Expr::ONE)
    .assert_zero(local.is_proof_start);
```

#### No proof end on disabled rows

$$
\neg\text{is\\_enabled} \Rightarrow \text{is\\_proof\\_end} = 0 \qquad (6)
$$
```rust
builder
    .when_ne(local.is_enabled, AB::Expr::ONE)
    .assert_zero(local.is_proof_end);
```

### 4. Proof Constraints

#### 4.1. Within Proof ($\Delta\text{proof\\_idx} \neq 1$)

At transition rows, $\Delta\text{proof\\_idx} \neq 1 \iff \Delta\text{proof\\_idx} = 0$ by (2), indicating we are within the same proof.

##### Enabled state consistency

$$
\text{is\\_enabled}\_{\text{local}} = \text{is\\_enabled}\_{\text{next}}\quad (\Delta\text{proof\\_idx} \neq 1,\ \text{transition}) \qquad (7)
$$
```rust
builder
    .when_transition()
    .when_ne(proof_idx_diff.clone(), AB::Expr::ONE)
    .assert_eq(local.is_enabled, next.is_enabled);
```

##### No proof end within proof

$$
\text{is\\_enabled}\_{\text{local}} \Rightarrow \text{is\\_proof\\_end}\_{\text{local}} = 0\quad (\Delta\text{proof\\_idx} \neq 1,\ \text{transition}) \qquad (8)
$$
```rust
builder
    .when_transition()
    .when_ne(proof_idx_diff.clone(), AB::Expr::ONE)
    .when(local.is_enabled)
    .assert_zero(local.is_proof_end);
```

##### No proof start within proof

$$
\text{is\\_enabled}\_{\text{local}} \Rightarrow \text{is\\_proof\\_start}\_{\text{next}} = 0\quad (\Delta\text{proof\\_idx} \neq 1,\ \text{transition}) \qquad (9)
$$
```rust
builder
    .when_transition()
    .when_ne(proof_idx_diff.clone(), AB::Expr::ONE)
    .when(local.is_enabled)
    .assert_zero(next.is_proof_start);
```

#### 4.2. At Proof Boundaries ($\Delta\text{proof\\_idx} \neq 0$)

When $\Delta\text{proof\\_idx} \neq 0$, we are at a proof boundary.

##### Enabled local row implies proof end

$$
\text{is\\_enabled}\_{\text{local}} \Rightarrow \text{is\\_proof\\_end}\_{\text{local}} = 1\quad (\Delta\text{proof\\_idx} \neq 0) \qquad (10)
$$
```rust
builder
    .when(proof_idx_diff.clone())
    .when(local.is_enabled)
    .assert_one(local.is_proof_end);
```

##### Enabled next row implies proof start

$$
\text{is\\_enabled}\_{\text{next}} \Rightarrow \text{is\\_proof\\_start}\_{\text{next}} = 1\quad (\Delta\text{proof\\_idx} \neq 0) \qquad (11)
$$
```rust
builder
    .when(proof_idx_diff)
    .when(next.is_enabled)
    .assert_one(next.is_proof_start);
```

## Case Analysis

### 1. Single Row

#### Case 1.1: Enabled single row ($\text{is\\_enabled} = 1$)

- By (3): $\text{is\\_proof\\_start} = 1$
- By (4): $\text{is\\_proof\\_end} = 1$

Single enabled row is properly marked as both start and end.

#### Case 1.2: Disabled single row ($\text{is\\_enabled} = 0$)

- By (5): $\text{is\\_proof\\_start} = 0$
- By (6): $\text{is\\_proof\\_end} = 0$

Single disabled row has no markers.

### 2. Single Proof ($\Delta\text{proof\\_idx} = 0$ for all transition rows)

#### Case 2.1: Enabled proof ($\text{is\\_enabled} = 1$ throughout)

- By (3): $\text{is\\_proof\\_start} = 1$ on first row
- By (4): $\text{is\\_proof\\_end} = 1$ on last row
- By (8): $\text{is\\_proof\\_end} = 0$ on all interior rows
- By (9): $\text{is\\_proof\\_start} = 0$ on all interior rows

Proof is properly marked with start on first row and end on last row only.

#### Case 2.2: Disabled proof ($\text{is\\_enabled} = 0$ throughout)

- By (5): $\text{is\\_proof\\_start} = 0$ on all rows
- By (6): $\text{is\\_proof\\_end} = 0$ on all rows

Disabled rows maintain no markers throughout.

### 3. Multiple Proofs

#### 3.1. Within Proof Transitions ($\Delta\text{proof\\_idx} \neq 1$)

##### Case 3.1.1: Enabled interior rows ($\text{is\\_enabled}\_{\text{local}} = 1$)

- By (7): $\text{is\\_enabled}\_{\text{next}} = 1$
- By (8): $\text{is\\_proof\\_end}\_{\text{local}} = 0$
- By (9): $\text{is\\_proof\\_start}\_{\text{next}} = 0$

Interior enabled rows maintain enabled state with no markers.

##### Case 3.1.2: Disabled interior rows ($\text{is\\_enabled}\_{\text{local}} = 0$)

- By (7): $\text{is\\_enabled}\_{\text{next}} = 0$
- By (5): $\text{is\\_proof\\_start}\_{\text{local}} = 0$ and $\text{is\\_proof\\_start}\_{\text{next}} = 0$
- By (6): $\text{is\\_proof\\_end}\_{\text{local}} = 0$ and $\text{is\\_proof\\_end}\_{\text{next}} = 0$

Disabled padding maintains disabled state with no markers.

#### 3.2. At Proof Boundaries ($\Delta\text{proof\\_idx} \neq 0$)

##### Case 3.2.1: Enabled to enabled ($\text{is\\_enabled}\_{\text{local}} = 1, \text{is\\_enabled}\_{\text{next}} = 1$)

- By (10): $\text{is\\_proof\\_end}\_{\text{local}} = 1$
- By (11): $\text{is\\_proof\\_start}\_{\text{next}} = 1$

Boundary between two enabled proofs is properly marked with end and start flags.

##### Case 3.2.2: Enabled to disabled ($\text{is\\_enabled}\_{\text{local}} = 1, \text{is\\_enabled}\_{\text{next}} = 0$)

- By (10): $\text{is\\_proof\\_end}\_{\text{local}} = 1$
- By (5): $\text{is\\_proof\\_start}\_{\text{next}} = 0$

End of enabled proof properly marked; next disabled row has no start marker.

##### Case 3.2.3: Disabled to enabled ($\text{is\\_enabled}\_{\text{local}} = 0, \text{is\\_enabled}\_{\text{next}} = 1$)

- By (6): $\text{is\\_proof\\_end}\_{\text{local}} = 0$
- By (11): $\text{is\\_proof\\_start}\_{\text{next}} = 1$

Start of enabled proof properly marked; previous disabled row has no end marker.

##### Case 3.2.4: Disabled to disabled ($\text{is\\_enabled}\_{\text{local}} = 0, \text{is\\_enabled}\_{\text{next}} = 0$)

- By (5): $\text{is\\_proof\\_start}\_{\text{next}} = 0$
- By (6): $\text{is\\_proof\\_end}\_{\text{local}} = 0$

Transition between disabled rows with proof index increment has no markers.
