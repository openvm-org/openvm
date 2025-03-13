# IsEqual

This chip outputs a boolean value `out` that equals 1 if and only if `x` equals `y`.

**IO Columns:**
- `x`: First input value to compare
- `y`: Second input value to compare
- `out`: Boolean output indicating whether `x = y`
- `condition`: Activation flag $s$ (constraints only apply when $s \neq 0$)

**Aux Columns:**
- `inv`: The purported inverse of `x - y` when `x ≠ y`

The chip leverages the `IsZeroSubAir` by checking if the difference `x - y` is zero.

**Constraints:**

The IsEqualSubAir applies the following constraint:

```math
\begin{align}
\texttt{IsZero}(x - y,\ \texttt{out},\ s,\ \texttt{inv}) & &\hspace{2em} (1)
\end{align}
```

where the IsZeroSubAir implements constraints to ensure that `out` is 1 if and only if `x - y` is zero (which happens exactly when `x = y`).
