# IsEqualArray

This chip outputs a boolean value `out` that equals 1 if and only if arrays `x` and `y` are equal.

**IO Columns:**
- `x`: Array of input values $`[x_0, x_1, ..., x_{n-1}]`$
- `y`: Array of input values $`[y_0, y_1, ..., y_{n-1}]`$
- `out`: Boolean output indicating whether `x = y`
- `condition`: Activation flag `s` (constraints only apply when `s != 0`)

**Aux Columns:**
- `diff_inv_marker`: Array where only the first index i with `x[i] != y[i]` contains the inverse of `x[i] - y[i]`, all others are 0

**Constraints:**

```math
\begin{align}
\texttt{out} \cdot (x_i - y_i) &= 0 & \forall\ i < N &\hspace{2em} (1)\\
s \cdot \left(\texttt{out} + \sum_{i=0}^{N-1} (x_i - y_i) \cdot \texttt{diff\_inv\_marker}_i - 1\right) &= 0 & &\hspace{2em} (2)\\
\texttt{out} \cdot (\texttt{out} - 1) &= 0 & &\hspace{2em} (3)
\end{align}
```

Two cases to consider:

1. When arrays are equal (`x = y`):
   - All differences `x_i - y_i` are zero, so constraint (1) is satisfied for any `out` value
   - In constraint (2), the sum term becomes zero, forcing `out = 1` when $s$ is active
   - Constraint (3) ensures `out` is boolean

2. When arrays are not equal:
   - Let `k` be the first index where `x_k ≠ y_k`
   - Constraint (1) forces `out = 0` since the product must be zero
   - For constraint (2), the prover sets `diff_inv_marker[k] = (x_k - y_k)^{-1}` and all other entries to zero
   - This makes the sum term equal to 1, satisfying constraint (2) when `out = 0`
   - Constraint (3) is satisfied since `out = 0`
