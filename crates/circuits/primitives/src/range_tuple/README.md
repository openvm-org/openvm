# Range Tuple

This chip efficiently range checks tuples of values using a single interaction when the product of their ranges is relatively small (less than ~2^20). For example, when checking pairs `(x, y)` against their respective bit limits, this approach is more efficient than performing separate range checks.

**Note:** This chip requires that each range size is a power of 2 (i.e., each value in `sizes` must be a power of 2).

**Columns:**
- `tuple`: Array of N columns (columns 0 to N-1) containing all possible tuple combinations within the specified ranges.
- `mult`: Multiplicity column (column N) tracking the number of range checks requested for each tuple.

The `sizes` parameter in `RangeTupleCheckerBus` defines the maximum value for each dimension.

As an example, for a 2-dimensional tuple with `sizes = [2, 4]`, the `tuple` column contains these 8 combinations in order:
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

For consecutive tuples `(local, next)`, we say that,

- Column `i` stays the same if `local[i] == next[i]`.
- Column `i` increments if `local[i] + 1 == next[i]`.
- Column `i` wraps if `local[i] == size[i] - 1` and `next[i] == 0`.

The AIR enforces the following constraints for the `tuple` column:

- (T1): The trace starts with `(0, ..., 0)`.
- (T2): The trace ends with `(size[0]-1, ..., size[N-1]-1)`.
- (T3): Between consecutive tuples, column `N-1` must increment or wrap.
- (T4): Between consecutive tuples, column `0` must stay the same or increment.
- (T5): Between consecutive tuples, all other columns must stay the same, increment, or wrap.
- (T6): Between consecutive tuples, column `i` increments or wraps if and only if column `i+1` wraps.

The constraints imply that, if `local` is a valid tuple (i.e. all values are in range), then `next` is also a valid tuple. The proof of this is as follows:

By (T3), column `N-1` must increment or wrap, and when this is combined with (T4) + (T5) + (T6), it is implied that there exists an `0 <= i <= N-1` where columns `0` to `i-1` stay the same, columns `i+1` to `N-1` wrap, and column `i` increments. By definition, all columns except column `i` are valid and stay in bounds. As such, the only possibly problematic column is column `i`. However, if `next[i] >= size[i]`, then it is impossible for column `i` to ever wrap in any following rows, so the value will never become `size[i]-1`, which is required by (T2).

Additionally, the constraints also imply that `next` is the lexographically next valid tuple after `local`.

Note that the length of the trace will be exactly `size[0] * ... * size[N-1]`, which is valid only if `size[i]` is always a power of 2.

___

Another remaining question is what polynomial constraints can be used to obtain (T6).

Define:

- `x := next[i] - local[i]`
- `y := next[i + 1] - local[i + 1]`
- `a := -(size[i] - 1)`
- `b := -(size[i+1] - 1)`

Note that constraints (T3), (T4), (T5) already force $x \in \{0, 1, a\}$, $y \in \{0, 1, b\}$. As such, to get T6, we only need to constrain the following table:

| (x,y) | valid configuration |
|-------|---------------------|
| (0,0) | yes                 |
| (0,1) | yes                 |
| (0,b) | no                  |
| (1,0) | no                  |
| (1,1) | no                  |
| (1,b) | yes                 |
| (a,0) | no                  |
| (a,1) | no                  |
| (a,b) | yes                 |

Consider the table with columns for the polynomials $(x-1)(x-a)y(y-1)$, $x^2(y-b)^2$, and $(x-1)(x-a)y(y-1) - x^2(y-b)^2$:

| (x,y) | $(x-1)(x-a)y(y-1)$ | $x^2(y-b)^2$ | $(x-1)(x-a)y(y-1) - x^2(y-b)^2$ |
|-------|------------------|------------|-------------------------------|
| (0,0) | 0                | 0          | 0                             |
| (0,1) | 0                | 0          | 0                             |
| (0,b) | nonzero          | 0          | nonzero                       |
| (1,0) | 0                | nonzero    | nonzero                       |
| (1,1) | 0                | nonzero    | nonzero                       |
| (1,b) | 0                | 0          | 0                             |
| (a,0) | 0                | nonzero    | nonzero                       |
| (a,1) | 0                | nonzero    | nonzero                       |
| (a,b) | 0                | 0          | 0                             |

Note that $(x-1)(x-a)y(y-1) - x^2(y-b)^2 = ay^2 - axy^2 - xy^2 - ay - x^2y + 2bx^2y + axy + xy - b^2x^2$ which has degree 3.

Thus, if we add the constraint $ay^2 - axy^2 - xy^2 - ay - x^2y + 2bx^2y + axy + xy - b^2x^2 = 0$, we are able to fully obtain T6.