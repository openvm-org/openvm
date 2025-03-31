# Range Checker

This chip implements range checking using a lookup table approach. It is initialized with a `max` value and gets requests to verify that a number is in the range `[0, max)`. The lookup table contains all possible values within the range and tracks the multiplicity of each value.

**Preprocessed Columns:**
- `counter`: Column containing all possible values within the range `[0, max)`

**IO Columns:**
- `mult`: Multiplicity column that tracks the number of range checks to perform for each element (left unconstrained)

The `RangeCheckerAir` adds interaction constraints:

```
    self.bus.receive(prep_local.counter).eval(builder, local.mult);
```

This adds the constraint that on every row, the AIR will receive the `counter` field with multiplicity `mult`.

Suppose we have another AIR that wants to constrain that the value in column `x` is within `[0, max)` whenever another boolean column `cond` is 1. It will do so by adding constraint

```
    self.bus.send(x).eval(builder, cond);
```

where the bus index must equal that used for `RangeChecker`.

Now during trace generation, every time the requester chip want to range check `x` with non-zero `cond`, it will increment the `mult` trace value in `RangeChecker`'s trace by `cond`. All `mult` trace values start at 0.

If the non-materialized send and receive multisets on the shared bus are equal, then the range check is satisfied.

**Note:** This implementation can also be used to efficiently range check values up to $`2^{\texttt{max\_bits}}`$ (where $`2^{\texttt{max\_bits}} < \texttt{max}`$) by checking both the original value and a shifted value. This works by verifying that both $`x`$ and $`x + (\texttt{max} - 2^{\texttt{max\_bits}})`$ are within the valid range of $`0..\texttt{max}`$. This approach only works when $`2 * \texttt{max}`$ is less than the field modulus to avoid wrap-around issues.

## Example

To give a concrete example, let's say `max` is 8 and the trace of Requester looks like this:

| x    | cond |
| ---- | ---- |
| 4    | 1    |
| 1    | 1    |
| 1    | 1    |
| 1000 | 0    |

Then if the `mult` trace values in `RangeChecker` were properly updated, the `RangeChecker` trace would look like this:

| counter | mult |
| ------- | ---- |
| 0       | 0    |
| 1       | 2    |
| 2       | 0    |
| 3       | 0    |
| 4       | 1    |
| 5       | 0    |
| 6       | 0    |
| 7       | 0    |

In this example, the multisets on the shared bus will be:

- Send multiset: `1 * [4] + 1 * [1] + 1 * [1]`
- Receive multiset: `2 * [1] + 1 * [4]`

These multisets are equal, so the range check is satisfied.

### ⚠️ Caution

We almost always prefer to use the [VariableRangeCheckerChip](../var_range/README.md) instead of this chip.
