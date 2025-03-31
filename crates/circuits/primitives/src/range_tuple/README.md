# Range Tuple

This chip efficiently range checks tuples of values using a single interaction when the product of their ranges is relatively small (less than ~2^20). For example, when checking pairs `(x, y)` against their respective bit limits, this approach is more efficient than performing separate range checks.

**Preprocessed Columns:**
- `tuple`: Column containing all possible tuple combinations within the specified ranges

**IO Columns:**
- `mult`: Multiplicity column tracking the number of range checks requested for each tuple

The implementation creates a preprocessed table with all possible value combinations within the specified ranges. The `sizes` parameter in `RangeTupleCheckerBus` defines the maximum value for each dimension.

For a 2-dimensional tuple with `sizes = [3, 2]`, the preprocessed trace contains these 6 combinations in lexicographic order:
```
(0,0)
(1,0)
(2,0)
(0,1)
(1,1)
(2,1)
```

During circuit execution, each row corresponds to a specific tuple from the preprocessed trace, with the `mult` column recording how many times that tuple was requested for range checking.
