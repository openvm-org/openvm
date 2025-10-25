# NestedForLoopSubAir

## Overview

This SubAir ensures that the first row of each loop iteration at each nesting level is properly marked with the `is_first` flag. It is parameterized by const generics `DEPTH_MINUS_ONE` and `DEPTH_MINUS_TWO` (e.g., `NestedForLoopSubAir<1, 0>` for a single nested loop, `NestedForLoopSubAir<2, 1>` for two nested loops).

## Columns

For a nested loop with `DEPTH` levels (numbered 0 to `DEPTH-1`), where level 0 is the outermost and level `DEPTH-1` is the innermost:

**I/O columns (`NestedForLoopIoCols`):**
- **is_enabled**: Flag indicating whether the row is enabled
- **counter**: Array of loop counters for all parent loops (excludes innermost loop, levels 0 to `DEPTH-2`)
- **is_first**: Array of flags indicating the first row of each loop iteration (excludes outermost loop, levels 1 to `DEPTH-1`)

**Auxiliary columns (`NestedForLoopAuxCols`):**
- **is_transition**: Array of parent loop transition flags (excludes outermost loop, levels 1 to `DEPTH-2`)

**Note:** The current (innermost) level (level `DEPTH-1`) does not have `counter` or auxiliary columns. The caller should add these columns if needed.

### Behavior

- Each `counter[level]` either stays the same or increases by 1 during a transition.
- `counter[level]` values do not necessarily start at 0.
- Disabled rows can appear between enabled rows and must have appropriate `counter[level]` values.
- Padding rows are not zero-filled; their `counter[level]` values form a non-decreasing sequence from the last enabled row (e.g., if the last enabled row has `counter[level] = k`, padding rows can have `counter[level] = k+1, k+1, k+2, k+3, ...`).

## Usage

**What this SubAir constrains:**
- `counter[level]` increments by 0 or 1 for all parent loops (excludes innermost loop, levels 0 to `DEPTH-2`)
- `is_first[level]` flags for all loops (excludes outermost loop, levels 1 to `DEPTH-1`) are set correctly at loop iteration boundaries (on enabled rows only)
- `is_transition[level]` auxiliary flags mark transitions within parent loop iterations (excludes outermost loop, levels 1 to `DEPTH-2`) (on enabled rows only)

**What the caller must constrain:**
- `is_enabled` being boolean (this SubAir does NOT constrain it)
- Initial or final values of `counter[level]` (for all parent loops, excludes innermost loop)
- `is_first[level]` flags on disabled rows (if required; this SubAir does NOT constrain them when `is_enabled = 0`)
- Innermost loop counter and its behavior (see example below)

**Example:**

For `DEPTH=3` loops like:
```rust
for i in 0..N {
    for j in 0..M {
        for k in START..END.step_by(STEP) {
            // ...
        }
    }
}
```

Add columns:
- **counter**: `T` - counter for current (innermost) loop
- **loop_io**: `NestedForLoopIoCols<T, 2>` - I/O loop columns
- **loop_aux**: `NestedForLoopAuxCols<T, 1>` - auxiliary loop columns

Example constraint code:
```rust
// Constrain is_enabled to be boolean
builder.assert_bool(local_io.loop_io.is_enabled);

// Constrain loop_io.counter columns using builder.when().assert_eq() or interactions

// loop starts at START
builder
    .when(local_io.loop_io.is_first)
    .assert_eq(local_io.counter, AB::Expr::from_canonical_u32(START));

// loop increments by STEP
let is_inner_transition = NestedForLoopSubAir::local_is_transition(
    next_io.loop_io.is_enabled,
    next_io.loop_io.is_first,
);
builder
    .when(is_inner_transition.clone())
    .assert_eq(next_io.counter, local_io.counter + AB::Expr::from_canonical_u32(STEP));

// loop ends at END
let is_inner_last = AB::Expr::ONE - is_inner_transition;
builder
    .when(is_inner_last)
    .assert_eq(local_io.counter, AB::Expr::from_canonical_u32(END));
```

## Constraints

The same constraint pattern applies at each loop level.

- For level 0 (outermost parent loop), constraints use `builder.when_first_row()` and `builder.when_transition()`.
- For level > 0 (nested within parent loops), constraints use `builder.when(parent_is_first)` and `builder.when(is_transition)` where parent refers to `level - 1`.

At a given level, let $\Delta\text{counter} = \text{counter}\_{\text{next}} - \text{counter}\_{\text{local}}$
```rust
let counter_diff = next_io.counter[level] - local_io.counter[level];
```

### 1. Base Constraints

#### `counter[level]` increments by 0 or 1

$$
\Delta\text{counter} \in \\\{0, 1\\\}\quad \text{(transition within level)} \qquad (1)
$$
```rust
// Level 0
builder.when_transition().assert_bool(counter_diff.clone());

// Level > 0
builder.when(is_transition).assert_bool(counter_diff.clone());
```

### 2. Boundary Constraints

#### First row enabled sets `is_first`

$$
\text{is\\_enabled} \Rightarrow \text{is\\_first} = 1\quad \text{(first row of level)} \qquad (2)
$$
```rust
// Level 0 (outermost parent loop)
builder.when_first_row().when(local_io.is_enabled).assert_one(local_io.is_first[0]);

// Level > 0 (nested within parent loops)
builder.when(parent_is_first).when(local_io.is_enabled).assert_one(local_io.is_first[level]);
```

### 3. Loop Constraints

#### 3.1. Within Loop Iteration ($\Delta\text{counter} \neq 1$)

At transition rows within a level, $\Delta\text{counter} \neq 1 \iff \Delta\text{counter} = 0$ by (1), indicating we are within the same loop iteration.

##### Enabled state consistency

$$
\text{is\\_enabled}\_{\text{local}} = \text{is\\_enabled}\_{\text{next}}\quad (\Delta\text{counter} \neq 1,\ \text{transition within level}) \qquad (3)
$$
```rust
builder
    .when(level_transition)
    .when_ne(counter_diff.clone(), AB::Expr::ONE)
    .assert_eq(local_io.is_enabled, next_io.is_enabled);
```

##### `is_first` not set within iteration

$$
\text{is\\_enabled}\_{\text{local}} \Rightarrow \text{is\\_first}\_{\text{next}} = 0\quad (\Delta\text{counter} \neq 1,\ \text{transition within level}) \qquad (4)
$$
```rust
builder
    .when(level_transition)
    .when_ne(counter_diff.clone(), AB::Expr::ONE)
    .when(local_io.is_enabled)
    .assert_zero(next_io.is_first[level]);
```

#### 3.2. At Loop Iteration Boundaries ($\Delta\text{counter} \neq 0$)

When $\Delta\text{counter} \neq 0$, we are at a loop iteration boundary.

##### Enabled next row sets `is_first`

$$
\text{is\\_enabled}\_{\text{next}} \Rightarrow \text{is\\_first}\_{\text{next}} = 1\quad (\Delta\text{counter} \neq 0) \qquad (5)
$$
```rust
builder
    .when(counter_diff)
    .when(next_io.is_enabled)
    .assert_one(next_io.is_first[level]);
```

## Case Analysis

These cases apply at each loop level independently.

### 1. Single Row ($\text{is\\_enabled} = 1$)

- By (2): $\text{is\\_first} = 1$

Single enabled row has `is_first` set.

### 2. Single Loop Iteration ($\Delta\text{counter} \neq 1$ for all transition rows, $\text{is\\_enabled} = 1$)

- By (2): $\text{is\\_first} = 1$ on first row
- By (4): $\text{is\\_first} = 0$ on all interior rows

Loop iteration has `is_first` set on first row only.

### 3. Multiple Loop Iterations

#### 3.1. Within Loop Iteration ($\Delta\text{counter} \neq 1$, $\text{is\\_enabled}\_{\text{local}} = 1$)

- By (3): $\text{is\\_enabled}\_{\text{next}} = 1$
- By (4): $\text{is\\_first}\_{\text{next}} = 0$

Interior enabled rows maintain enabled state with `is_first` unset.

#### 3.2. At Loop Iteration Boundaries ($\Delta\text{counter} \neq 0$, $\text{is\\_enabled}\_{\text{next}} = 1$)

- By (5): $\text{is\\_first}\_{\text{next}} = 1$

Any transition to a new iteration has `is_first` set.

## Parent Loop Context

For loops nested within parent loops (level > 0), constraints use the parent loop's (`level - 1`) `is_first` and `is_transition` values.

The `is_transition[parent_level]` column stores the parent transition value to reduce the constraint degree. This value ensures child loop constraints are only enforced when transitioning within the parent loop iteration.

```rust
// calculated as: next_io.is_enabled * (1 - next_io.is_first[parent_level])
let parent_is_transition = Self::local_is_transition(next_io.is_enabled, next_io.is_first[parent_level]);

builder.assert_eq(local_aux.is_transition[parent_level], parent_is_transition.clone());
```
