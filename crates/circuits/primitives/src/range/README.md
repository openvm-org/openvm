# Range Checker

This chip is initialized with a `max` value and gets requests to verify that a number is in the range `[0, max)`. It has only two columns `counter` and `mult`. The `counter` column is preprocessed and the `mult` is left unconstrained.

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
