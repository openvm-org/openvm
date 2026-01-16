# Range Tuple

This chip efficiently range checks tuples of values using a single interaction when the product of their ranges is relatively small (less than ~2^20). For example, when checking pairs `(x, y)` against their respective bit limits, this approach is more efficient than performing separate range checks.

**Note:** This chip requires that each range size is a power of 2 (i.e., each value in `sizes` must be a power of 2).

**Columns:**
- `tuple`: Array of N columns (columns 0 to N-1) containing all possible tuple combinations within the specified ranges
- `is_first`: Array of (N-1) boolean columns (columns N to 2*N-2). `is_first[i]` is 1 if `tuple[i + 1]` has just switched to a new number, 0 otherwise.
- `mult`: Multiplicity column (column 2*N-1) tracking the number of range checks requested for each tuple

The `sizes` parameter in `RangeTupleCheckerBus` defines the maximum value for each dimension.

For a 2-dimensional tuple with `sizes = [4, 2]`, the preprocessed trace contains these 8 combinations in order:
```
(0,0)
(0,1)
(0,2)
(0,3)
(1,0)
(1,1)
(1,2)
(1,3)
```

## Circuit Constraints

### Boundary Constraints

We enforce that the trace starts at `(0, 0, ..., 0)` and ends at `(sizes[0]-1, sizes[1]-1, ..., sizes[N-1]-1)`.

Additionally, in the first row, all `is_first` columns must be 1.

### is_first Constraints

For each `0 <= i < N-1`:
- `is_first[i]` must be boolean (0 or 1)
- In the first row: `is_first[i] = 1`

### Transition Constraints

The transition constraints enforce lexicographic ordering between consecutive rows using `is_first` to track wrapping behavior.

**For `tuple[0]` (leftmost component):**
- `next.tuple[0] - local.tuple[0]` is boolean (0 or 1)

**For `tuple[i]` where `1 <= i < N-1` (middle components):**
- When `next.is_first[i-1] != 1`: `next.tuple[i] - local.tuple[i]` is boolean (0 or 1)
- When `next.is_first[i-1] == 1`: wrapping occurs (handled by wrapping constraints)

**For `tuple[N-1]` (rightmost component):**
- When `next.is_first[N-2] != 1`: `next.tuple[N-1] - local.tuple[N-1] = 1` (must increment by 1)
- When `next.is_first[N-2] == 1`: wrapping occurs (handled by wrapping constraints)

**is_first Constraints:**
The circuit validates `next.is_first` based on tuple changes:
- `next.is_first[0] = next.tuple[0] - local.tuple[0]`
- For `1 <= i < N-1`:
  - When `next.is_first[i-1] != 1`: `next.is_first[i] = next.tuple[i] - local.tuple[i]`
  - When `next.is_first[i-1] == 1`: `next.is_first[i] = 1`

**Wrapping Constraints:**
When `next.is_first[i] == 1` (indicating `tuple[i+1]` wraps):
- `local.tuple[i+1] = sizes[i+1] - 1` (was at maximum)
- `next.tuple[i+1] = 0` (wraps to zero)
