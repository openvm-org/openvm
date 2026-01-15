# Range Tuple

This chip efficiently range checks tuples of values using a single interaction when the product of their ranges is relatively small (less than ~2^20). For example, when checking pairs `(x, y)` against their respective bit limits, this approach is more efficient than performing separate range checks.

**Columns:**
- `tuple`: Array of N columns containing all possible tuple combinations within the specified ranges
- `tuple_inverse`: Array of (N-1) columns where `tuple_inverse[i] = inv(tuple[i] - (bus.sizes[i] - 1))` for 0 <= i <= N-2. Used to detect when a tuple component reaches its maximum value.
- `prefix_product`: Array of (N-1) columns when N > 2, or 0 columns when N == 2. For 0 <= i <= N-2, `prefix_product[i] = is_last[0] * ... * is_last[i]` where `is_last[i] = tuple_inverse[i] * (tuple[i] - (bus.sizes[i] - 1))`. When N == 2, the prefix product is inlined into the transition constraints to optimize the circuit.
- `mult`: Multiplicity column tracking the number of range checks requested for each tuple

The `sizes` parameter in `RangeTupleCheckerBus` defines the maximum value for each dimension.

For a 2-dimensional tuple with `sizes = [3, 2]`, the preprocessed trace contains these 6 combinations in lexicographic order:
```
(0,0)
(1,0)
(2,0)
(0,1)
(1,1)
(2,1)
```

## Circuit Constraints

### Boundary Constraints

We enforce that the trace starts at `(0, 0, ..., 0)` and ends at `(sizes[0]-1, sizes[1]-1, ..., sizes[N-1]-1)`:

### tuple_inverse Constraints

For each `0 <= i < N-1`, we enforce,
```
tuple_inverse[i] * (tuple[i] - (sizes[i] - 1)) * (tuple[i] - (sizes[i] - 1)) = tuple[i] - (sizes[i] - 1)
```

This enforces:
- `tuple_inverse[i] = inv(tuple[i] - (sizes[i] - 1))` when `tuple[i] != sizes[i] - 1`
- `tuple_inverse[i] = 0` when `tuple[i] = sizes[i] - 1`

### prefix_product Constraints (N > 2 only)

For `N > 2` and `1 <= i < N-1`:
```
prefix_product[i] = prefix_product[i-1] * (1 - tuple_inverse[i] * (tuple[i] - (sizes[i] - 1)))
```

where `prefix_product[0] = 1 - tuple_inverse[0] * (tuple[0] - (sizes[0] - 1))`.

### Transition Constraints

The transition constraints enforce lexicographic ordering between consecutive rows. The behavior differs for `N == 2` and `N > 2`:

#### N == 2 Case

When `N == 2`, the `prefix_product` column is eliminated and its computation is inlined directly into the transition constraints:

**For `tuple[0]` (leftmost component):**
```
next.tuple[0] = (local.tuple[0] + 1) * (local.tuple_inverse[0] * (local.tuple[0] - (sizes[0] - 1)))
```

This means:
- If `tuple[0] < sizes[0] - 1`: `next.tuple[0] = tuple[0] + 1` (increment)
- If `tuple[0] = sizes[0] - 1`: `next.tuple[0] = 0` (wrap to zero)

**For `tuple[1]` (rightmost component):**
```
next.tuple[1] = local.tuple[1] + (1 - local.tuple_inverse[0] * (local.tuple[0] - (sizes[0] - 1)))
```

This means:
- If `tuple[0] < sizes[0] - 1`: `next.tuple[1] = tuple[1]` (no change)
- If `tuple[0] = sizes[0] - 1`: `next.tuple[1] = tuple[1] + 1` (increment)

This creates the lexicographic ordering: `(0,0) → (1,0) → (2,0) → (0,1) → (1,1) → (2,1)` for `sizes = [3, 2]`.

#### N > 2 Case

For `N > 2`, the transition constraints use `prefix_product` to determine when each component should increment:

**For `tuple[0]` (leftmost component):**
```
next.tuple[0] = (local.tuple[0] + 1) * (1 - local.prefix_product[0])
```

- Increments every row, wrapping to `0` when it reaches `sizes[0] - 1`

**For `tuple[i]` where `1 <= i < N-1` (middle components):**
```
next.tuple[i] = (local.tuple[i] + local.prefix_product[i-1]) * (1 - local.prefix_product[i])
```

- Increments when all components to the left are at their maximum (`prefix_product[i-1] = 1`)
- Wraps to `0` when it and all components to the left are at their maximum (`prefix_product[i] = 1`)

**For `tuple[N-1]` (rightmost component):**
```
next.tuple[N-1] = local.tuple[N-1] + local.prefix_product[N-2]
```

- Only increments when all other components are at their maximum (`prefix_product[N-2] = 1`)
- Never wraps (it's the last component to increment)
